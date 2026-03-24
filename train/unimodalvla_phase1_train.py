import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed

from model.configuration_unimodalvla import UniModalVLAConfig
from model.modeling_unimodalvla import UniModalVLAForConditionalGeneration
from model import SpatialVLAProcessor, SpatialActionTokenizer
from data.dataset import build_datasets
from train.monkey_patch import concat_pad_data_collator


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Base checkpoint path"})
    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    use_vision_zoe: bool = field(default=True)
    use_depth_modality: bool = field(default=True)


@dataclass
class DataArguments:
    data_root_dir: Optional[str] = field(
        default="datasets/open-x-embodiment",
        metadata={"help": "Root directory of OXE/RLDS datasets."},
    )
    data_mix: Optional[str] = field(
        default="bridge",
        metadata={"help": "Dataset mixture name (e.g., bridge, fractal20220817_data)."},
    )
    max_seq_length: Optional[int] = field(default=2048)
    shuffle_buffer_size: Optional[int] = field(default=1000_000)
    tsfm_thread_muti: Optional[int] = field(default=1)
    read_thread_muti: Optional[int] = field(default=1)
    obs_backward_steps: Optional[int] = field(default=0)
    obs_backward_delta: Optional[int] = field(default=1)
    action_forward_steps: Optional[int] = field(default=3)
    fix_raw_length: Optional[int] = field(default=None)
    use_raw_dataloader: Optional[bool] = field(default=False)
    use_dummy_dataset: bool = field(
        default=False,
        metadata={"help": "Use random dummy dataset for smoke test instead of OXE pipeline."},
    )


class DummyPhase1Dataset(torch.utils.data.Dataset):
    def __init__(self, length=8, seq=64, vocab_size=257200, image_size=224, patch=14):
        self.length = length
        self.seq = seq
        self.vocab_size = vocab_size
        self.image_size = image_size
        self.patch_tokens = (image_size // patch) ** 2

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = torch.randint(low=100, high=min(self.vocab_size, 5000), size=(self.seq,), dtype=torch.long)

        image_token_index = 256000
        depth_token_index = 256001
        input_ids[: self.patch_tokens] = image_token_index
        input_ids[self.patch_tokens : 2 * self.patch_tokens] = depth_token_index

        attention_mask = torch.ones(self.seq, dtype=torch.long)
        labels = input_ids.clone()

        pixel_values = torch.randn(3, self.image_size, self.image_size)
        intrinsic = torch.eye(3)
        intrinsic[0, 0] = 300.0
        intrinsic[1, 1] = 300.0
        intrinsic[0, 2] = self.image_size / 2
        intrinsic[1, 2] = self.image_size / 2

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "intrinsic": intrinsic,
        }


def collate_batch(batch):
    keys = batch[0].keys()
    result = {}
    for key in keys:
        result[key] = torch.stack([item[key] for item in batch], dim=0)
    return result


def maybe_freeze(model, model_args: ModelArguments):
    if model_args.freeze_vision_tower:
        for param in model.vision_tower.parameters():
            param.requires_grad = False
    if model_args.freeze_llm:
        for param in model.language_model.parameters():
            param.requires_grad = False


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training args: {training_args}")

    set_seed(training_args.seed)

    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    if not data_args.use_dummy_dataset and not model_args.model_name_or_path:
        raise ValueError(
            "For OXE dataset mode, `model_name_or_path` is required so processor/tokenizer can be loaded. "
            "Set `--use_dummy_dataset True` for smoke test without checkpoint."
        )

    if model_args.model_name_or_path:
        config = UniModalVLAConfig.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch_dtype,
            use_vision_zoe=model_args.use_vision_zoe,
            use_depth_modality=model_args.use_depth_modality,
            local_files_only=True,
        )
        model = UniModalVLAForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            local_files_only=True,
            ignore_mismatched_sizes=True,
        )
    else:
        config = UniModalVLAConfig(
            use_vision_zoe=model_args.use_vision_zoe,
            use_depth_modality=model_args.use_depth_modality,
        )
        model = UniModalVLAForConditionalGeneration(config)

    maybe_freeze(model, model_args)

    if data_args.use_dummy_dataset:
        train_dataset = DummyPhase1Dataset(
            length=16,
            seq=2 * model.config.text_config.num_image_tokens + 16,
            vocab_size=model.config.text_config.vocab_size,
            image_size=model.config.vision_config.image_size,
            patch=model.config.vision_config.patch_size,
        )
        tokenizer = None
        data_collator = collate_batch
    else:
        logger.info(f"Building OXE dataset with data_mix={data_args.data_mix}")
        train_dataset, _ = build_datasets(
            data_args,
            training_args.output_dir,
            vla_processor=None,
        )

        base_processor = SpatialVLAProcessor.from_pretrained(
            model_args.model_name_or_path,
            local_files_only=True,
        )
        tokenizer = base_processor.tokenizer
        action_tokenizer = SpatialActionTokenizer(
            tokenizer,
            num_bins=base_processor.action_config["num_bins"],
            bin_policy=base_processor.action_tokenizer.bin_policy,
            use_spherical=base_processor.action_config["use_spherical"],
            min_sigma=base_processor.action_config.get("min_sigma", 0.0),
        )

        processor = SpatialVLAProcessor(
            image_processor=base_processor.image_processor,
            tokenizer=tokenizer,
            statistics=base_processor.statistics,
            bin_policy=action_tokenizer.bin_policy,
            intrinsic_config=base_processor.intrinsic_config,
            action_config=base_processor.action_config,
            num_obs_steps=data_args.obs_backward_steps + 1,
            obs_delta=data_args.obs_backward_delta,
            action_chunk_size=data_args.action_forward_steps + 1,
        )
        train_dataset.vla_processor = processor

        # 当前 Phase1 使用 SpatialVLA 的 processor 输入协议进行可比测试。
        # 由于该协议默认不生成 <depth> token，占位替换只走 image 分支。
        # 为避免深度分支强制替换报错，这里关闭 depth 分支。
        model.config.use_depth_modality = False
        data_collator = concat_pad_data_collator

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model()


if __name__ == "__main__":
    main()
