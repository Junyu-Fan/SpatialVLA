import torch

from model.configuration_unimodalvla import UniModalVLAConfig
from model.modeling_unimodalvla import UniModalVLAForConditionalGeneration


def _make_batch(config, batch_size=2, use_rgb=True, use_depth=True):
    image_tokens = config.text_config.num_image_tokens
    seq = image_tokens * (int(use_rgb) + int(use_depth)) + 16

    input_ids = torch.randint(200, min(config.text_config.vocab_size, 10000), (batch_size, seq), dtype=torch.long)
    cursor = 0
    if use_rgb:
        input_ids[:, cursor : cursor + image_tokens] = config.image_token_index
        cursor += image_tokens
    if use_depth:
        input_ids[:, cursor : cursor + image_tokens] = config.depth_token_index

    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    pixel_values = None
    intrinsic = None
    if use_rgb:
        image_size = config.vision_config.image_size
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        intrinsic = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsic[:, 0, 0] = 300.0
        intrinsic[:, 1, 1] = 300.0
        intrinsic[:, 0, 2] = image_size / 2
        intrinsic[:, 1, 2] = image_size / 2
    elif use_depth:
        # depth-only path still needs intrinsic
        image_size = config.vision_config.image_size
        intrinsic = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsic[:, 0, 0] = 300.0
        intrinsic[:, 1, 1] = 300.0
        intrinsic[:, 0, 2] = image_size / 2
        intrinsic[:, 1, 2] = image_size / 2

    depth_values = None
    if use_depth:
        image_size = config.vision_config.image_size
        depth_values = torch.rand(batch_size, 1, image_size, image_size) * 2.0 + 0.1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "depth_values": depth_values,
        "intrinsic": intrinsic,
    }


def run_case(name, model, config, use_rgb, use_depth):
    batch = _make_batch(config, use_rgb=use_rgb, use_depth=use_depth)
    outputs = model(**batch)
    assert outputs.logits.shape[0] == batch["input_ids"].shape[0], f"{name}: bad batch dim"
    assert outputs.logits.shape[1] == batch["input_ids"].shape[1], f"{name}: bad seq dim"
    assert outputs.loss is not None, f"{name}: loss is None"
    print(f"[PASS] {name}: loss={outputs.loss.item():.4f}")


def main():
    config = UniModalVLAConfig(
        use_vision_zoe=False,
        use_depth_modality=True,
        # shrink model for fast smoke test
        vision_config={
            "model_type": "siglip_vision_model",
            "hidden_size": 256,
            "intermediate_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "image_size": 224,
            "patch_size": 14,
            "vision_use_head": False,
            "vocab_size": 257152,
        },
        text_config={
            "model_type": "gemma2",
            "hidden_size": 256,
            "intermediate_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "vocab_size": 257152,
        },
        projection_dim=256,
        hidden_size=256,
    )
    model = UniModalVLAForConditionalGeneration(config)
    model.eval()

    with torch.no_grad():
        run_case("rgb+depth", model, config, use_rgb=True, use_depth=True)
        run_case("rgb-only", model, config, use_rgb=True, use_depth=False)
        run_case("depth-only", model, config, use_rgb=False, use_depth=True)


if __name__ == "__main__":
    main()
