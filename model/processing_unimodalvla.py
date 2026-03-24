# coding=utf-8
from typing import List, Optional, Union

import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin, _validate_images_text_input_order, Unpack
from transformers.tokenization_utils_base import AddedToken, PreTokenizedInput, TextInput
from transformers.models.paligemma.processing_paligemma import (
    make_batched_images,
    build_string_from_input,
    _is_str_or_image,
    PaliGemmaProcessorKwargs,
    IMAGE_TOKEN,
    EXTRA_TOKENS,
)


DEPTH_TOKEN = "<depth>"


# =============================================================================
# 与原 processing_spatialvla.py 的核心差异（Phase 1）
# 1) 新增 <depth> 占位 token，并与 <image> 并列管理。
# 2) 输入可按 use_rgb/use_depth 动态构建，天然支持缺模态。
# 3) depth_values 可直接外部传入（若已有深度），不强制在 processor 内估计。
# =============================================================================


class UniModalVLAProcessor(ProcessorMixin):
    """
    Phase 1 processor:
    - keeps RGB token placeholder (<image>)
    - adds depth placeholder (<depth>)
    - supports missing RGB/depth by deciding placeholders dynamically
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = ("GemmaTokenizer", "GemmaTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        if not hasattr(image_processor, "image_seq_length"):
            raise ValueError("Image processor is missing an `image_seq_length` attribute.")

        self.image_seq_length = image_processor.image_seq_length

        image_token = AddedToken(IMAGE_TOKEN, normalized=False, special=True)
        depth_token = AddedToken(DEPTH_TOKEN, normalized=False, special=True)
        # 与原 SpatialVLA 不同：同时注册 <image>/<depth> 两类 special token。
        tokenizer.add_special_tokens({"additional_special_tokens": [image_token, depth_token]})
        tokenizer.add_tokens(EXTRA_TOKENS)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        self.depth_token_id = tokenizer.convert_tokens_to_ids(DEPTH_TOKEN)

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        depth_values: Optional[torch.Tensor] = None,
        intrinsic: Optional[torch.Tensor] = None,
        use_rgb: bool = True,
        use_depth: bool = True,
        **kwargs: Unpack[PaliGemmaProcessorKwargs],
    ) -> BatchFeature:
        # __call__ 是处理器的主要入口，接收各种输入，返回 BatchFeature。
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
            PaliGemmaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if text is None:
            text = ""
        if _is_str_or_image(text):
            text = [text]

        if use_rgb and images is None:
            raise ValueError("use_rgb=True requires `images`.")
        if use_depth and intrinsic is None:
            raise ValueError("use_depth=True requires `intrinsic`.")

        # 根据模态开关动态插入占位序列：
        # - use_rgb=False: 不插 <image>
        # - use_depth=False: 不插 <depth>
        # 这样 forward 阶段就不会替换对应模态 token，形成缺模态路径。
        rgb_placeholder = IMAGE_TOKEN * self.image_seq_length if use_rgb else ""
        depth_placeholder = DEPTH_TOKEN * self.image_seq_length if use_depth else ""
        prefix = f"{rgb_placeholder}{depth_placeholder}"

        input_strings = [
            build_string_from_input(
                prompt=f"{prefix}{prompt}",
                bos_token=self.tokenizer.bos_token,
                image_seq_len=1,
                image_token="",
                num_images=0,
            )
            for prompt in text
        ]
        # 当启用 RGB 时，调用图像处理器将图像转换为 pixel_values；当启用深度时，depth_values 直接透传（外部提供或内部估计）。
        if use_rgb:
            if isinstance(images, list) and len(images) > 0 and not isinstance(images[0], list):
                images = [[image] for image in images]
            elif not isinstance(images, list):
                images = [[images]]
            images = make_batched_images(images)
            pixel_values = self.image_processor(images, **output_kwargs["images_kwargs"])["pixel_values"]
        else:
            pixel_values = None

        inputs = self.tokenizer(input_strings, **output_kwargs["text_kwargs"])

        return_data = {**inputs, "intrinsic": intrinsic}
        if pixel_values is not None:
            return_data["pixel_values"] = pixel_values
        if depth_values is not None:
            # 直接透传 depth_values，供模型 depth 分支优先使用（跳过 Zoe 估计）。
            return_data["depth_values"] = depth_values
        return BatchFeature(data=return_data)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names + ["depth_values", "intrinsic"]))
