# 数据总入口   coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from typing import List, Optional, Union, Dict
import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, is_valid_image
from transformers.processing_utils import Unpack, _validate_images_text_input_order, ProcessorMixin
from transformers.tokenization_utils_base import AddedToken, PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.models.paligemma.processing_paligemma import (
    make_batched_images, 
    build_string_from_input, 
    _is_str_or_image, 
    PaliGemmaProcessorKwargs,
    IMAGE_TOKEN,
    EXTRA_TOKENS
)
from .action_tokenizer import SpatialActionTokenizer
logger = logging.get_logger(__name__)

class SpatialVLAProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = ("GemmaTokenizer", "GemmaTokenizerFast")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        statistics: Optional[dict] = None,
        bin_policy=None,
        intrinsic_config=None,
        action_config=None,
        num_obs_steps=1,
        obs_delta=1,
        action_chunk_size=1,
        min_sigma=0.0,
        **kwargs,
    ):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        if not hasattr(image_processor, "image_seq_length"):
            raise ValueError("Image processor is missing an `image_seq_length` attribute.")

        self.image_seq_length = image_processor.image_seq_length #图像 token 展开长度

        if not hasattr(tokenizer, "image_token"):  #如果 tokenizer 没有 image_token 属性，则添加一个特殊 token 作为 image_token
            image_token = AddedToken(IMAGE_TOKEN, normalized=False, special=True)
            tokens_to_add = {"additional_special_tokens": [image_token]}
            tokenizer.add_special_tokens(tokens_to_add)
            self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        else:
            self.image_token_id = tokenizer.image_token_id

        tokenizer.add_tokens(EXTRA_TOKENS) # PaliGemma 预留的一组“额外特殊词元” EXTRA_TOKENS 是一个列表，包含 PaliGemma 模型预留的一组额外特殊 token（如 <image>、<bos>、<eos> 等）
        tokenizer.add_bos_token = False # 取消 bos token，因为我们会在输入字符串中手动添加 bos token 来确保它在 image_token 之前 （Beginning of Sequence token）
        tokenizer.add_eos_token = False # 取消 eos token，因为我们会在输入字符串中手动添加 eos token 来确保它在 image_token 之后 （End of Sequence token）

        super().__init__(image_processor, tokenizer, chat_template=chat_template)  # 调用父类构造

        # action tokenizer
        self.statistics = statistics if statistics else {}
        self.bin_policy = bin_policy
        self.min_sigma = min_sigma
        self.intrinsic_config = intrinsic_config
        self.action_config = action_config
        self.num_obs_steps = num_obs_steps
        self.obs_delta = obs_delta
        self.action_chunk_size = action_chunk_size
        self.dataset_intrinsics = {}
        height, width = image_processor.size["height"], image_processor.size["width"] # 读取图像处理器的输入图像大小

        # scale intrinsic matrix 这里是根据输入图像大小和内参矩阵原始大小的比例来缩放内参矩阵，以适应输入图像的大小。因为内参矩阵中的焦距和主点位置是基于原始图像大小的（内参和图像大小有关），所以需要进行缩放以保持正确的几何关系。
        for k, v in intrinsic_config.items():
            K = torch.tensor(v["intrinsic"]).float()
            K[:2] *= torch.tensor([width / v["width"], height / v["height"]])[:, None]
            self.dataset_intrinsics[k] = K
        
        self.action_tokenizer = SpatialActionTokenizer(
            tokenizer=tokenizer, num_bins=action_config["num_bins"], 
            bin_policy=bin_policy, use_spherical=action_config["use_spherical"],
            min_sigma=min_sigma,
        )

    def __call__( # 把外部样本变成模型输入
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        unnorm_key: Optional[str] = None,
        suffix_actions: Optional[np.array] = None, # (t e) 模型输出的动作 token ids，形状为 (t, e)，其中 t 是动作 token 的时间步数，e 是每个动作 token 的维度（通常是 3，分别对应平移、旋转和夹爪状态）。这个参数用于在输入字符串中添加一个后缀，表示连续动作序列，以便模型在训练时能够学习从图像和文本输入到动作输出的映射关系。  将动作 token 作为后缀追加到输入序列的末尾，并在 tokenizer 中设置 return_token_type_ids=True 来区分文本 token 和动作 token，从而在计算损失时只关注动作 token 的预测。
        **kwargs: Unpack[PaliGemmaProcessorKwargs],
    ) -> BatchFeature:
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs( #Keyword Arguments 合并，主要是把用户输入的 kwargs 和默认的 PaliGemmaProcessorKwargs 合并成一个字典，方便后续使用
            PaliGemmaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if suffix_actions is not None:
            action_tokens = self.action_tokenizer(suffix_actions) # (n,3) 把连续动作离散化成 token
            suffix="".join(action_tokens.flatten()) #展平后拼成一个字符串 suffix
        else:
            suffix = output_kwargs["text_kwargs"].pop("suffix", None)

        return_token_type_ids = True if suffix is not None else False #如果有 suffix，就要求 tokenizer 返回 token_type_ids（后面用于构建 labels mask）。

        if images is None:
            raise ValueError("`images` are expected as arguments to a `PaliGemmaProcessor` instance.")
        if text is None:
            logger.warning_once( "You are using PaliGemma without a text prefix. It will perform as a picture-captioning model.")
            text = ""

        if _is_str_or_image(text): # 若 text 是单个字符串/图像对象，包装成列表
            text = [text]
        elif isinstance(text, list) and _is_str_or_image(text[0]):
            pass

        if text is not None and images is not None: #文本和图像都存在时处理多模态拼接
            if not any(IMAGE_TOKEN in sample for sample in text): # 如果文本中没有 IMAGE_TOKEN 占位符，则默认在文本末尾添加一个 IMAGE_TOKEN 占位符，并根据图像数量调整占位符的重复次数（如果有多个图像，则重复 IMAGE_TOKEN 占位符以匹配图像数量）。如果文本中没有 IMAGE_TOKEN 占位符，说明用户没有明确指定图像应该放在文本的哪个位置，那么我们默认把图像放在文本的末尾。为了让模型知道图像的位置，我们在文本末尾添加一个 IMAGE_TOKEN 占位符，并且如果有多个图像，我们会重复这个占位符来表示每个图像的位置。这样模型在处理输入时就能正确地识别出图像的位置和数量，从而更好地理解多模态输入。
                if isinstance(text, List) and isinstance(images, List):
                    if len(images) != len(text):
                        raise ValueError(
                            f"Received {len(images)} images for {len(text)} prompts. Each prompt should be associated with an image or list of images."
                        )
                if is_valid_image(images):
                    images = [[images]]
                elif isinstance(images, list) and is_valid_image(images[0]):
                    images = [[image] for image in images]
                elif not (isinstance(images, list) and isinstance(images[0], list) and is_valid_image(images[0][0])):
                    raise ValueError("images must be an image, list of images or list of list of images")
                if suffix is not None and _is_str_or_image(suffix): suffix = [suffix]
                if suffix is not None: suffix = [sfx + self.tokenizer.eos_token for sfx in suffix]
                input_strings = [ # 为每个文本和对应的图像列表构建输入字符串，使用 build_string_from_input 函数来处理文本和图像列表，插入 bos_token 和 image_token，并根据图像数量调整 image_token 的重复次数
                    build_string_from_input(
                        prompt=prompt,
                        bos_token=self.tokenizer.bos_token,
                        image_seq_len=self.image_seq_length,
                        image_token=IMAGE_TOKEN,
                        num_images=len(image_list) if isinstance(image_list, list) else 1,
                    )
                    for prompt, image_list in zip(text, images)
                ]
                images = make_batched_images(images)
            else: # 如果文本中已经包含 IMAGE_TOKEN 占位符，则假设用户已经正确地将占位符放置在文本中，并且不进行任何修改。直接使用文本中的 IMAGE_TOKEN 占位符来处理图像输入。对于每个文本样本，找到最后一个 IMAGE_TOKEN 占位符的位置，并在该位置之前插入 bos_token，以确保 bos_token 在 image_token 之前。然后根据图像数量调整 image_token 的重复次数（如果有多个图像，则重复 IMAGE_TOKEN 占位符以匹配图像数量）。最后为每个文本样本构建输入字符串。
                expanded_samples = []
                for sample in text:
                    expanded_sample = sample.replace(IMAGE_TOKEN, IMAGE_TOKEN * self.image_seq_length)
                    bos_rfind_index = expanded_sample.rfind(IMAGE_TOKEN)
                    bos_index = bos_rfind_index + len(IMAGE_TOKEN) if bos_rfind_index != -1 else 0
                    expanded_sample = (
                        expanded_sample[:bos_index] + self.tokenizer.bos_token + expanded_sample[bos_index:]
                    )
                    expanded_samples.append(expanded_sample)
                input_strings = [f"{sample}\n" for sample in expanded_samples]
        pixel_values = self.image_processor(images, **output_kwargs["images_kwargs"])["pixel_values"]

        if output_kwargs["text_kwargs"].get("max_length", None) is not None:
            output_kwargs["text_kwargs"]["max_length"] += self.image_seq_length

        inputs = self.tokenizer(
            input_strings,
            text_pair=suffix,
            return_token_type_ids=return_token_type_ids,
            **output_kwargs["text_kwargs"],
        )

        intrinsic = self.dataset_intrinsics[unnorm_key] if unnorm_key in self.dataset_intrinsics else self.dataset_intrinsics["default"]
        return_data = {**inputs, "pixel_values": pixel_values, "intrinsic": intrinsic}

        if return_token_type_ids:
            labels = inputs["input_ids"].masked_fill(inputs["token_type_ids"] == 0, -100)
            return_data.update({"labels": labels})
        return BatchFeature(data=return_data)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Gemma
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Gemma
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def decode_actions( # 把模型输出的 action token ids 解码成连续动作
        self,
        generation_outputs: torch.Tensor,
        unnorm_key: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        action_token_num = 3  # translation + rotation + gripper
        predicted_action_token_ids = generation_outputs[0, : action_token_num * self.action_chunk_size].detach().cpu().long().numpy()
        assert self.tokenizer.eos_token != predicted_action_token_ids[-1], "[error] actions contain EOS token, please check you truncation settings!"

        if predicted_action_token_ids.shape[0] < action_token_num * self.action_chunk_size:  # pad with zeros
            logger.warning(f"Padding zero action!")
            predicted_action_token_ids = np.concatenate(
                [
                    predicted_action_token_ids,
                    np.zeros(action_token_num * self.action_chunk_size - predicted_action_token_ids.shape[0], dtype=np.longlong),
                ]
            )
        predicted_action_token_ids = predicted_action_token_ids.reshape(-1, action_token_num)
        normalized_action_chunks = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids)

        if unnorm_key is None:
            logger.warning(f"unnorm_key {unnorm_key} is not in statistics, use next one")
            unnorm_key = next(self.statistics.keys())
        action_norm_stats = self.statistics[unnorm_key]["action"]

        action_dim = len(action_norm_stats["q01"])
        mask = np.array(action_norm_stats.get("mask", np.ones(action_dim)), dtype=bool)
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])

        actions = []
        for normalized_actions in normalized_action_chunks:
            action = np.where(
                mask,
                0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
                normalized_actions,
            )
            actions.append(action)
        actions = np.stack(actions)
        return {"actions": actions, "action_ids": predicted_action_token_ids}