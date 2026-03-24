import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional
import json
import torch
import torch.distributed as dist
from train.dist_utils import init_dist
from train.monkey_patch import (
    replace_train_dataloader,
    replace_compute_loss,
    concat_pad_data_collator,
    replace_train_sampler,
    SaveProcessorCallback
)
import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    set_seed,
    TrainingArguments,
)
from peft import get_peft_model, LoraConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format,
    set_verbosity,
)
from data.dataset import build_datasets
from model import (
    SpatialVLAConfig,
    SpatialVLAForConditionalGeneration,
    SpatialVLAProcessor,
    SpatialActionTokenizer,
)

# -----------------------------------------------------------------------------
# Trainer 行为补丁（Monkey Patch）
# 这些函数会替换 HuggingFace Trainer 的默认行为：
# 1) 自定义 DataLoader（适配本项目 IterableDataset）
# 2) 自定义 compute_loss（记录动作 token 指标等）
# 3) 自定义 sampler（长度分组等）
# -----------------------------------------------------------------------------
replace_train_dataloader()
replace_compute_loss()
replace_train_sampler()

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    # 预训练/微调模型路径（必须能被 SpatialVLAProcessor + SpatialVLAForConditionalGeneration 加载）
    model_name_or_path: Optional[str] = field(default=None,
        metadata={"help": "Path to pretrained model or identifier for resume training."},
    )
    # 是否冻结 LLM token embedding（常见于动作 token 微调，避免词表基础语义漂移）
    freeze_llm_embed: bool = field(
        default=True, metadata={"help": "Set to True to freeze the LLM embeddings."},
    )
    # 是否冻结视觉主干（冻结后只训练语言/投影/LoRA 等模块）
    freeze_vision_tower: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the vision backbone of the model."},
    )
    # LoRA rank；0 表示不启用 LoRA（全参数或按冻结策略训练）
    lora: int = field(
        default=0,
        metadata={"help": "Set the LoRA adapter rank for the LLM. Default is 0."},
    )
    # LoRA alpha（缩放系数）
    lora_alpha: int = field(
        default=8,
        metadata={"help": "Set the LoRA adapter rank for the LLM. Default is 0."},
    )
    # LoRA 目标模块预设：linear / linear+emb / linear+emb+h
    lora_target: Optional[str] = field(
        default="linear",
        metadata={"help": "Set the LoRA adapter rank for the LLM. Default is linear."},
    )
    # 需要在 LoRA 外额外保存的模块（+ 分隔）
    modules_to_save: Optional[str] = field(
        default=None,
        metadata={"help": "Set the LoRA adapter rank for the LLM. Default is none."},
    )
    # 是否启用梯度检查点（省显存，通常会增加训练时长）
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "Set to True to use gradient checkpointing."},
    )
    # 是否启用 FlashAttention2
    flash_attn: bool = field(
        default=True,
        metadata={"help": "Set to True to use Flash Attention 2.0."},
    )
    # 动作空间 embedding 自适应（给一个高斯参数 json）
    adapt_emb: Optional[str] = field(
        default=None,
        metadata={"help": "Set to True to adapt the spatial embeddings with new gaussian config."},
    )
    # 在 adapt_emb 时，是否同时自适应 embedding 特征值本身
    adpt_feature: bool = field(
        default=False,
        metadata={"help": "Set to True to adapt the feature embeddings."},
    )
    # 高斯分箱最小 sigma（防止过窄分布导致分箱不稳定）
    min_sigma: float = field(
        default=0.0,
        metadata={"help": "Set the minimum sigma for creating action grids."},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    # RLDS/OXE 数据根目录
    data_root_dir: Optional[str] = field(
        default="datasets/open-x-embodiment",
        metadata={"help": "The root directory of the dataset. Default is `data`."},
    )
    # 数据混合配方名（在 data/oxe/mixtures.py 里定义）
    data_mix: Optional[str] = field(
        default="bridge",
        metadata={"help": "The name of the dataset mixture. Default is `bridge`."},
    )
    # 文本 token 最大长度（processor 侧会结合 image token 一起处理）
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum total input sequence length after tokenization. "},
    )
    # 数据混洗缓冲区（越大随机性越好，CPU/内存开销也越高）
    shuffle_buffer_size: Optional[int] = field(
        default=1000_000,
        metadata={"help": "The shuffle buffer size for the dataset. Default is 1000000."},
    )
    # RLDS 轨迹变换线程倍率
    tsfm_thread_muti: Optional[int] = field(
        default=1,
        metadata={"help": "The threads number of rlds transfom. Default is 1."},
    )
    # RLDS 读取线程倍率
    read_thread_muti: Optional[int] = field(
        default=1,
        metadata={"help": "The threads number of rlds reader. Default is 1."},
    )
    # 观测窗口：往回取多少步
    obs_backward_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Number of backward steps in observation. 0 indicates current"},
    )
    # 观测步长间隔
    obs_backward_delta: Optional[int] = field(
        default=1, metadata={"help": "Backward delta in observation."}
    )
    # 动作窗口：往前预测多少步（action chunk）
    action_forward_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Number of forward steps in action. 0 indicates current"},
    )
    # 固定 iterable dataset 长度（用于避免 resume 时提前结束）
    fix_raw_length: Optional[int] = field(
        default=None, metadata={"help": "fix the iterable dataset iter length."}
    )
    # 是否使用原始 DataLoader（不走 accelerator.prepare）
    use_raw_dataloader: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use raw dataloader"}
    )

def main():
    # -------------------------------------------------------------------------
    # 0) 初始化分布式
    # -------------------------------------------------------------------------
    launcher = os.environ.get("LAUNCHER", "slurm")
    init_dist(launcher=launcher, backend="nccl")  # 设置 torch.distributed 环境
    
    # 解析三组参数：模型参数、数据参数、HF Training 参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # 支持两种启动方式：
    # 1) python xxx.py config.json
    # 2) python xxx.py --arg1 ...
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # -------------------------------------------------------------------------
    # 1) 日志系统初始化
    # -------------------------------------------------------------------------
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()  # 让 transformers 输出更详细日志

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # -------------------------------------------------------------------------
    # 2) 检查输出目录与断点恢复
    # -------------------------------------------------------------------------
    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        ckpt_files = list(filter(lambda x: x.startswith("checkpoint"), os.listdir(training_args.output_dir)))
        if last_checkpoint is None and len(ckpt_files) > 0:
            ckpt_files = list(filter(lambda x: x.startswith("checkpoint"), os.listdir(training_args.output_dir)))
        if last_checkpoint is None and len(ckpt_files) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)  # 固定随机种子，便于复现实验

    # -------------------------------------------------------------------------
    # 3) 加载 processor/tokenizer/config/model
    # -------------------------------------------------------------------------
    # 1. initializing models and load tokenizer
    _processor = SpatialVLAProcessor.from_pretrained(model_args.model_name_or_path, local_files_only=True)  # 先拿已有processor
    tokenizer = _processor.tokenizer  # tokenizer 复用checkpoint中的配置
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32  # 与训练精度保持一致
    
    logger.info("Loading SpatialVLA Model...")
    config = SpatialVLAConfig.from_pretrained(model_args.model_name_or_path, torch_dtype=torch_dtype, local_files_only=True)  # 读取模型配置
    model = SpatialVLAForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        local_files_only=True
    )  # 读取模型权重
    if model_args.flash_attn:
        model.language_model.config._attn_implementation = model.config.text_config._attn_implementation_internal = "flash_attention_2"  # LLM 使用 FA2
        model.vision_tower.config._attn_implementation = model.config.vision_config._attn_implementation_internal = "flash_attention_2"  # Vision tower 使用 FA2

    # -------------------------------------------------------------------------
    # 4) 构建数据集
    # -------------------------------------------------------------------------
    # 2. build datasets
    train_dataset, eval_dataset = build_datasets(
        data_args,
        training_args.output_dir,
        vla_processor=None,
    )

    # -------------------------------------------------------------------------
    # 5) 重建 action tokenizer（与当前数据/分箱策略一致）
    # -------------------------------------------------------------------------
    # 3. build action tokenizer from current project
    action_tokenizer = SpatialActionTokenizer(
        tokenizer,
        num_bins=_processor.action_config["num_bins"],
        bin_policy=_processor.action_tokenizer.bin_policy,
        use_spherical=_processor.action_config["use_spherical"],
        min_sigma=_processor.action_config.get("min_sigma", 0.0),
    )
    
    # 可选：根据新高斯配置自适应空间 embedding（迁移不同动作分布）
    if model_args.adapt_emb and config.use_spatial_token:
        logger.info(f"adapt spatial embeddings with guassian distribution {model_args.adapt_emb}")
        gs_params = json.load(open(model_args.adapt_emb))
        action_tokenizer.spatial_embedding_adaption(gs_params, model.spatial_embed_tokens, model_args.min_sigma, model_args.adpt_feature)
        logger.info(f"new adaptation embedding {model.spatial_embed_tokens.weight.data}")

        if model_args.adpt_feature:
            model_args.lora_target="linear"  # 特征自适应时，强制 LoRA 目标回到 linear 方案
            model_args.modules_to_save="spatial_embed_tokens"  # 确保新 embedding 被保存
            logger.info(f"reset lora_target to {model_args.lora_target} and modules_to_save {model_args.modules_to_save}")

    # -------------------------------------------------------------------------
    # 6) 覆盖模型运行期属性 + 梯度检查点
    # -------------------------------------------------------------------------
    # overwrite attributes
    model.action_token_begin_idx = model.config.action_token_begin_idx = action_tokenizer.action_token_begin_idx  # 统一 action token 起始索引
    model.vision_tower.gradient_checkpointing = True  # vision tower 开启 gc

    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()  # LLM 开启 gc
    
    # -------------------------------------------------------------------------
    # 7) 冻结策略（控制可训练参数）
    # -------------------------------------------------------------------------
    # set freeze params
    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_llm_embed:
        model.language_model.model.embed_tokens.weight.requires_grad = False  # 冻结词嵌入

    if model_args.freeze_vision_tower:
        model.vision_tower = model.vision_tower.eval()  # 关闭 BN/Dropout 的训练行为
        _freeze_params(model.vision_tower)  # 冻结 vision tower 参数

    model.vision_zoe_model = model.vision_zoe_model.eval()  # ZoeDepth 一般只做特征提取
    _freeze_params(model.vision_zoe_model)  # 默认冻结 ZoeDepth

    # -------------------------------------------------------------------------
    # 8) 可选 LoRA 注入
    # -------------------------------------------------------------------------
    if model_args.lora:
        # peft https://github.com/huggingface/peft/blob/c1fe8105a5a4a612a6178699e1def5c66c2638d2/src/peft/tuners/tuners_utils.py#L1027
        if model_args.lora_target == "linear":
            target_modules=[
                "q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj", # com
                "fc1", "fc2", "out_proj", # siglip
                "linear", # projector
                "position_embedding_head.0", "position_embedding_head.3" # ego3d
            ]
        elif model_args.lora_target == "linear+emb":
            target_modules=[
                "q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj", # com
                "fc1", "fc2", "out_proj", # siglip
                "linear", # projector
                "position_embedding_head.0", "position_embedding_head.3", # ego3d
                "spatial_embed_tokens",
            ]
        elif model_args.lora_target == "linear+emb+h":
            target_modules=[
                "q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj", "lm_head", # com
                "fc1", "fc2", "out_proj", # siglip
                "linear", # projector
                "position_embedding_head.0", "position_embedding_head.3", # ego3d
                "spatial_embed_tokens",
            ]
        else:
            raise ValueError(f"don't support lora targets {model_args.lora_target}")
        
        # modules_to_save: https://github.com/huggingface/peft/issues/334#issuecomment-1786449397
        modules_to_save = model_args.modules_to_save.split("+") if model_args.modules_to_save else []  # "a+b+c" -> [a,b,c]
        lora_config = LoraConfig(
            r=model_args.lora,
            lora_alpha=model_args.lora_alpha,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            init_lora_weights="gaussian",
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, lora_config)  # 返回 PEFT 包装后的模型
        logger.info(f"use Lora ... with {model_args.lora_target} and modules {modules_to_save} ...")
        model.print_trainable_parameters()  # 打印可训练参数占比

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad: logger.info(name)

    set_seed(training_args.seed)  # 再次设置，确保 trainer 内部阶段可复现
    SpatialVLAConfig.register_for_auto_class() # 注册 auto class，便于 save/load 时自动映射
    SpatialVLAForConditionalGeneration.register_for_auto_class()
    SpatialVLAProcessor.register_for_auto_class()

    # -------------------------------------------------------------------------
    # 9) 重建 processor（合并本次训练统计信息）
    # -------------------------------------------------------------------------
    # build processor
    statistic = train_dataset.ds_stats_pc  # 本次数据集统计（用于反归一化）
    _processor.statistics.update(statistic)  # 合并旧统计与新统计
    processor = SpatialVLAProcessor(
        image_processor=_processor.image_processor,
        tokenizer=tokenizer,
        statistics=_processor.statistics,
        bin_policy=action_tokenizer.bin_policy,
        intrinsic_config=_processor.intrinsic_config,
        action_config=_processor.action_config,
        num_obs_steps=data_args.obs_backward_steps + 1,
        obs_delta=data_args.obs_backward_delta,
        action_chunk_size=data_args.action_forward_steps + 1,
    )

    model.action_tokenizer = action_tokenizer  # 让模型在日志/解码时可访问 action tokenizer
    train_dataset.vla_processor = processor  # dataset 在 __iter__ 时会调用 processor 组装输入

    # -------------------------------------------------------------------------
    # 10) Trainer 构建与训练
    # -------------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=concat_pad_data_collator,
        callbacks=[SaveProcessorCallback(processor=processor)],  # 保存 checkpoint 时同步保存 processor
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)  # 真正开始训练
        # trainer.save_model()

        metrics = train_result.metrics  # 训练返回指标
        metrics["train_samples"] = len(train_dataset)  # 补充样本量统计

        trainer.log_metrics("train", metrics)   # 打印到日志
        trainer.save_metrics("train", metrics)  # 保存到 json
        trainer.save_state()                     # 保存 trainer state（优化器/调度器等）

if __name__ == "__main__":
    main()
