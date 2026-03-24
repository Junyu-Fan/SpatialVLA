# coding=utf-8
from transformers.configuration_utils import PretrainedConfig
from transformers import CONFIG_MAPPING, AutoConfig


# =============================================================================
# 与原 SpatialVLA 配置的核心差异（Phase 1）
# 1) 新增 depth_token_index：把深度序列作为独立占位 token 流，而不是和 RGB 特征相加。
# 2) 新增 use_depth_modality：显式控制深度分支是否启用，便于做缺模态训练/推理。
# 3) num_depth_tokens 与 num_image_tokens 对齐：深度 token 长度与 patch 数一致。
# =============================================================================


class UniModalVLAConfig(PretrainedConfig):
    model_type = "unimodalvla"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig, "vision_zoe_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        depth_token_index=256001,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        vision_zoe_config=None,
        ego3d_patch_reso=4,
        n_freqs=8,
        use_vision_zoe=True,
        use_depth_modality=True,
        **kwargs,
    ):
        self._ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.depth_token_index = depth_token_index
        self._vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.is_encoder_decoder = False

        self.vision_config = vision_config
        if isinstance(self.vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "siglip_vision_model"
            )
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"](
                intermediate_size=4096,
                hidden_size=1152,
                patch_size=14,
                image_size=224,
                num_hidden_layers=27,
                num_attention_heads=16,
                vocab_size=vocab_size,
                vision_use_head=False,
            )

        self.text_config = text_config
        if isinstance(self.text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "gemma2"
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["gemma2"](
                hidden_size=2048,
                num_hidden_layers=18,
                intermediate_size=16384,
                num_attention_heads=8,
                num_key_value_heads=1,
                is_encoder_decoder=False,
                vocab_size=vocab_size,
            )

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.text_config.num_depth_tokens = self.text_config.num_image_tokens
        self.vision_config.projection_dim = projection_dim

        self.vision_zoe_config = vision_zoe_config
        if isinstance(self.vision_zoe_config, dict):
            vision_zoe_config["model_type"] = (
                vision_zoe_config["model_type"] if "model_type" in vision_zoe_config else "zoedepth"
            )
            self.vision_zoe_config = CONFIG_MAPPING[vision_zoe_config["model_type"]](**vision_zoe_config)

        self.ego3d_patch_reso = ego3d_patch_reso
        self.n_freqs = n_freqs
        self.use_vision_zoe = use_vision_zoe
        self.use_depth_modality = use_depth_modality

        super().__init__(**kwargs)

    @property
    def ignore_index(self):
        return self._ignore_index

    @classmethod
    def from_spatialvla_config(cls, spatial_config, **overrides):
        # 兼容从原 SpatialVLA 配置迁移到 UniModalVLA 的最小入口。
        # 这里默认补齐深度独立模态所需字段，不影响原配置文件本身。
        data = spatial_config.to_dict()
        data["depth_token_index"] = overrides.pop("depth_token_index", 256001)
        data["use_depth_modality"] = overrides.pop("use_depth_modality", True)
        data.update(overrides)
        return cls(**data)
