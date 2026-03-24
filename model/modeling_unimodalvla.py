# coding=utf-8
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import os
import torch
import torch.utils.checkpoint
from torch import nn
from torch.linalg import inv
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from transformers.cache_utils import Cache, HybridCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput, logging
from transformers import AutoModel, ZoeDepthForDepthEstimation

from .configuration_unimodalvla import UniModalVLAConfig
from .modeling_gemma2 import Gemma2ForCausalLM


SIGLIP_MEAN, SIGLIP_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
ZOE_MEAN, ZOE_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

logger = logging.get_logger(__name__)


# =============================================================================
# 与原 SpatialVLA 的结构差异（Phase 1）
# - 原 SpatialVLA: 2D 视觉特征 X 与 3D 位置嵌入 P' 直接相加后走同一个投影层。
# - 本文件 UniModalVLA: RGB 与 Depth 是两条独立 token 流，分别走独立 projector，
#   最终通过不同 special token 位置注入到 LLM 输入序列。
# - 本阶段不包含 MoE（MoE 放在后续 Phase 2）。
# =============================================================================


class Ego3DPositionEmbeddingMLP(nn.Module):
    def __init__(self, in_channels=3, num_pos_feats=768, n_freqs=8, logscale=True):
        super().__init__()
        self.n_freqs = n_freqs
        self.freq_out_channels = in_channels * (2 * n_freqs + 1)
        if logscale:
            freq_bands = 2 ** torch.linspace(0, n_freqs - 1, n_freqs)
        else:
            freq_bands = torch.linspace(1, 2 ** (n_freqs - 1), n_freqs)

        center = torch.tensor([0.0, 0.0, 2.0]).repeat(in_channels // 3)
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        self.register_buffer("center", center, persistent=False)

        self.position_embedding_head = nn.Sequential(
            nn.Linear(self.freq_out_channels, num_pos_feats),
            nn.LayerNorm(num_pos_feats),
            nn.ReLU(),
            nn.Linear(num_pos_feats, num_pos_feats),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.01)

    @torch.no_grad()
    def frequency_encoding(self, xyz):
        xyz_n = ((xyz - self.center) / 2.0).to(self.freq_bands.dtype)
        xyz_feq = xyz_n.unsqueeze(-1) * self.freq_bands
        sin_xyz, cos_xyz = torch.sin(xyz_feq), torch.cos(xyz_feq)
        encoding = torch.cat([xyz_n.unsqueeze(-1), sin_xyz, cos_xyz], -1).reshape(*xyz.shape[:2], -1)
        return encoding

    def forward(self, xyz):
        freq_encoding = self.frequency_encoding(xyz)
        position_embedding = self.position_embedding_head(freq_encoding)
        return position_embedding


def process_zoe(pixel_values, pad_mode="reflect"):
    ph, pw = 31, 31
    images = F.pad(pixel_values, (pw, pw, ph, ph), mode=pad_mode)
    images = F.interpolate(images, size=(384, 384), mode="bicubic", align_corners=True)
    images = TF.normalize(images, mean=ZOE_MEAN, std=ZOE_STD)
    return images, ph, pw


@dataclass
class UniModalVLACausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    depth_hidden_states: Optional[torch.FloatTensor] = None


class UniModalProjector(nn.Module):
    def __init__(self, config: UniModalVLAConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, features):
        return self.linear(features)


class UniModalVLAPreTrainedModel(PreTrainedModel): #下面给UniModalVLAForConditionalGeneration继承
    config_class = UniModalVLAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )
        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class UniModalVLAForConditionalGeneration(UniModalVLAPreTrainedModel, GenerationMixin):
    """Phase 1: no-MoE baseline with independent RGB/depth streams.  主程序"""

    def __init__(self, config: UniModalVLAConfig, vision_model=None, vision_zoe_model=None, language_model=None):
        super().__init__(config)

        # 与原 SpatialVLA 不同：保留同一个视觉塔，但拆成两个投影头。
        # RGB 特征 -> rgb_projector，Depth(P') 特征 -> depth_projector。
        self.vision_tower = vision_model or AutoModel.from_config(config=config.vision_config) # 共享编码器：RGB 与深度 token 后续都在同一视觉特征空间工作（Phase 1 先保持简单）。？ 这个好像就在rgb_features里用到
        self.rgb_projector = UniModalProjector(config)
        self.depth_projector = UniModalProjector(config)

        if language_model is None:
            language_model = Gemma2ForCausalLM(config=config.text_config)
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]
        self.language_model = language_model #挂载语言模型

        if config.use_depth_modality and config.use_vision_zoe:
            # 与原 SpatialVLA 相同点：深度估计 + 反投影 + 高频位置编码。
            # 与原 SpatialVLA 不同点：这里输出作为独立深度 token 流，不与 RGB 相加。
            self.vision_zoe_model = vision_zoe_model or ZoeDepthForDepthEstimation(config.vision_zoe_config) #创建 ZoeDepth 网络，用 RGB 估计深度图
            self.position_embedding_3d = Ego3DPositionEmbeddingMLP(
                config.ego3d_patch_reso ** 2 * 3,
                num_pos_feats=config.vision_config.hidden_size,
                n_freqs=config.n_freqs,
            ) # 创建 3D 位置编码器：把反投影后的 patch 3D 点变成深度 token embedding（即 P'）。
            patch_size, reso, image_size = (
                config.vision_config.patch_size,
                config.ego3d_patch_reso,
                config.vision_config.image_size,
            )
            y, x = torch.meshgrid(
                torch.arange(0, image_size, patch_size // reso),
                torch.arange(0, image_size, patch_size // reso),
                indexing="ij",
            )
            y, x = y + patch_size / reso / 2, x + patch_size / reso / 2
            uv_h = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(3, -1)
            self.register_buffer("uv_h", uv_h, persistent=False)

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        vocab_size = model_embeds.weight.shape[0]
        self.config.text_config.vocab_size = vocab_size
        self.config._vocab_size = vocab_size
        self.tie_weights()
        return model_embeds

    def backproject_patch(self, K: torch.Tensor, depth: torch.Tensor, patch_size=14, reso=2) -> torch.Tensor:
        b, c, h, w = depth.shape
        hp, wp = h // patch_size, w // patch_size
        sub_hp = sub_wp = reso
        patch_depth = F.interpolate(depth, size=(hp * reso, wp * reso), mode="area").reshape(b, c, -1)
        p_cam = (inv(K.float()) @ self.uv_h.float()) * patch_depth
        patch_p_cam = p_cam.reshape(b, 3, hp, sub_hp, wp, sub_wp).permute(0, 2, 4, 3, 5, 1).reshape(b, hp * wp, -1)
        return patch_p_cam

    def _update_causal_mask(
        self,
        attention_mask,
        token_type_ids,
        past_key_values,
        cache_position,
        input_ids=None,
        inputs_embeds=None,
        is_training: bool = False,
    ):
        # 用于在自回归生成或训练时正确屏蔽未来 token 的注意力
        if self.config.text_config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        using_static_cache = isinstance(past_key_values, StaticCache)
        min_dtype = torch.finfo(self.dtype).min
        inputs_lead_dim = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        sequence_length = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        elif isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask

        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=self.dtype,
            device=cache_position.device,
        )
        if sequence_length != 1:
            if is_training:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            else:
                causal_mask[:, :sequence_length] = 0.0

        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
            if is_training and token_type_ids is not None:
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    token_type_ids[:, None, None, :].to(causal_mask.device) == 0,
                    0,
                )
        return causal_mask

    def get_rgb_features(self, pixel_values: torch.FloatTensor):
        # 主要的处理已经在processer里处理过了，这里直接过视觉塔 + 投影头得到 RGB 模态特征，作为 <image> token 的替换内容注入 LLM。
        # RGB 独立分支：只编码视觉语义，不和深度分支做逐元素融合。
        # 输入是图像张量 pixel_values，通常形状 (B, 3, H, W)。
        siglip_pixel_values = TF.normalize(pixel_values, mean=SIGLIP_MEAN, std=SIGLIP_STD)
        image_outputs = self.vision_tower(siglip_pixel_values)
        rgb_features = self.rgb_projector(image_outputs.last_hidden_state) # 用 RGB 专用投影层把视觉维度映射到 LLM 输入维度，输出 (B, N_patch, D_text)。
        rgb_features = rgb_features / (self.config.text_config.hidden_size ** 0.5) #尺度归一
        return rgb_features

    def get_depth_features(self, pixel_values: torch.FloatTensor, intrinsic: torch.FloatTensor, depth_values: Optional[torch.Tensor] = None):
        # Depth 独立分支：得到 P' 后走 depth_projector，作为深度模态 token。
        # 注意：这里不执行 X + P'，这是和原 SpatialVLA 的关键区别。
        if depth_values is None: #如果外部没给深度图，就走“RGB -> ZoeDepth”自动估计。
            if not hasattr(self, "vision_zoe_model"):
                raise ValueError("depth_values is None and use_vision_zoe is disabled.")
            zoe_pixel_values, ph, pw = process_zoe(pixel_values, pad_mode="reflect")
            with torch.no_grad(): #深度估计不用梯度
                pvh, pvw = pixel_values.shape[-2:]
                depth = self.vision_zoe_model(pixel_values=zoe_pixel_values).predicted_depth
                depth = F.interpolate(
                    depth.unsqueeze(1),
                    size=(pvh + 2 * ph, pvw + 2 * pw),
                    mode="bicubic",
                    align_corners=True,
                )[..., ph:-ph, pw:-pw]
        else:
            depth = depth_values if depth_values.dim() == 4 else depth_values.unsqueeze(1)

        xyz = self.backproject_patch(
            intrinsic,
            depth,
            patch_size=self.config.vision_config.patch_size,
            reso=self.config.ego3d_patch_reso,
        )
        depth_tokens = self.position_embedding_3d(xyz) #把 3D 点做频率编码 + MLP，得到深度模态 token
        depth_features = self.depth_projector(depth_tokens) #用 depth 专用投影头映射到 LLM 维度 (B, N_patch, D_text)。
        depth_features = depth_features / (self.config.text_config.hidden_size ** 0.5)# 尺度归一
        return depth_features

    @staticmethod
    def _replace_special_tokens_with_features(input_ids, inputs_embeds, token_id, features, token_name):
        # 将某一类 special token (<image> 或 <depth>) 的 embedding 批量替换为对应模态特征。
        # 这就是“拼接到 LLM 输入空间”的具体落点。
        special_mask = (input_ids == token_id).unsqueeze(-1)
        special_mask = special_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        if inputs_embeds[special_mask].numel() != features.numel():
            token_count = torch.sum(input_ids == token_id)
            raise ValueError(
                f"Number of {token_name} features does not match number of special tokens. "
                f"Got {token_count} tokens but {features.shape[0] * features.shape[1]} feature tokens."
            )
        return inputs_embeds.masked_scatter(special_mask, features.to(inputs_embeds.device, inputs_embeds.dtype))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        depth_values: Optional[torch.FloatTensor] = None,
        intrinsic: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, UniModalVLACausalLMOutputWithPast]:
        # Phase 1 前向逻辑：
        # 1) RGB token 位替换为 rgb_features（可选）
        # 2) Depth token 位替换为 depth_features（可选）
        # 3) 两路都可缺失（由输入与占位 token 决定），实现缺模态推理路径
        output_attentions = output_attentions or self.config.output_attentions
        output_hidden_states = output_hidden_states or self.config.output_hidden_states
        return_dict = return_dict or self.config.use_return_dict

        is_training = token_type_ids is not None and labels is not None

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids).clone()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0) + 1

        rgb_features = None
        depth_features = None

        if pixel_values is not None:
            rgb_features = self.get_rgb_features(pixel_values)
            inputs_embeds = self._replace_special_tokens_with_features(
                input_ids,
                inputs_embeds,
                self.config.image_token_index,
                rgb_features,
                token_name="image",
            )

        use_depth = depth_values is not None or (
            self.config.use_depth_modality and pixel_values is not None and intrinsic is not None
        )
        # 与原 SpatialVLA 不同：depth 分支是否启用由独立条件控制，不依赖 RGB+P' 融合。
        if use_depth:
            if intrinsic is None:
                raise ValueError("intrinsic is required when depth branch is used.")
            depth_features = self.get_depth_features(pixel_values, intrinsic, depth_values=depth_values)
            inputs_embeds = self._replace_special_tokens_with_features(
                input_ids,
                inputs_embeds,
                self.config.depth_token_index,
                depth_features,
                token_name="depth",
            )

        if labels is not None and self.pad_token_id in labels:
            labels = torch.where(input_ids == self.pad_token_id, self.config.ignore_index, labels)

        causal_mask = self._update_causal_mask(
            attention_mask,
            token_type_ids,
            past_key_values,
            cache_position,
            input_ids,
            inputs_embeds,
            is_training,
        )

        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs.logits
        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            loss_fct = nn.CrossEntropyLoss()
            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return UniModalVLACausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=rgb_features,
            depth_hidden_states=depth_features,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        depth_values=None,
        intrinsic=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            num_logits_to_keep=num_logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        if model_inputs.get("position_ids") is not None:
            model_inputs["position_ids"] += 1

        if cache_position[0] == 0:
            # 生成时首步同时注入 RGB/Depth，后续 step 依赖 KV cache，避免重复编码。
            model_inputs["pixel_values"] = pixel_values
            model_inputs["depth_values"] = depth_values

        is_training = token_type_ids is not None and labels is not None
        if cache_position[0] == 0 and isinstance(past_key_values, HybridCache):
            causal_mask = self._update_causal_mask(
                attention_mask,
                token_type_ids,
                past_key_values,
                cache_position,
                input_ids,
                inputs_embeds,
                is_training,
            )
            model_inputs["attention_mask"] = causal_mask

        model_inputs["intrinsic"] = intrinsic
        return model_inputs

    @torch.no_grad()
    def predict_action(self, model_inputs) -> torch.Tensor:
        model_inputs = model_inputs.to(torch.bfloat16).to(self.device)
        input_len = model_inputs["input_ids"].shape[-1]
        generation_outputs = self.generate(**model_inputs, max_new_tokens=256, do_sample=False)
        return generation_outputs[:, input_len:] #只返回“新生成部分”

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ):
        #加载预训练模型
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs,
        )
        return model
