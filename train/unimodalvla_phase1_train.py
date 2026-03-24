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
    """
    Placeholder for project-specific dataloader integration.
    Phase 1 focuses on architecture wiring + smoke run.
    """

    train_json: Optional[str] = field(default=None, metadata={"help": "Optional local json for custom dataset"})


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

    dummy_train = DummyPhase1Dataset(
        length=16,
        seq=2 * model.config.text_config.num_image_tokens + 16,
        vocab_size=model.config.text_config.vocab_size,
        image_size=model.config.vision_config.image_size,
        patch=model.config.vision_config.patch_size,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dummy_train if training_args.do_train else None,
        data_collator=collate_batch,
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model()


if __name__ == "__main__":
    main()
