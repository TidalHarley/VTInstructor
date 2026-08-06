#!/usr/bin/env python3
"""
Panorama 格式的 NIG 训练脚本（过滤版本）

与原版 train_nig_panorama.py 的关键区别：
- 使用 PanoramaFilteredDataset，自动跳过 filtered_ult.json 中不存在的低质量样本
- 需要额外参数 --filtered_json 指定过滤结果文件

使用方法：
    python train_nig_panorama_merged.py \
        --data_dir /path/to/data \
        --filtered_json /path/to/filtered_ult.json \
        --output_dir /path/to/output

    # 多 GPU 训练
    python -m torch.distributed.run --nproc_per_node=8 train_nig_panorama_merged.py ...
"""
import argparse
import os
import torch
from torch.utils.data import ConcatDataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments

from nig_dataset_panorama_merged import PanoramaFilteredDataset, PanoramaCollator


def main():
    parser = argparse.ArgumentParser(description="Train NIG model on filtered Panorama data")

    parser.add_argument(
        "--model_name",
        default="PATH/TO/models_cache/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        help="预训练模型路径"
    )
    parser.add_argument(
        "--processor_name",
        default="",
        help="Processor/Tokenizer 路径（为空时与 model_name 相同）"
    )
    parser.add_argument(
        "--data_dir",
        default="PATH/TO/outputs/R2RCE_visual/r2rce_train_visual",
        help="兼容旧参数：R2RCE visual 训练数据目录"
    )
    parser.add_argument(
        "--r2rce_data_dir",
        default="",
        help="R2RCE 训练数据目录（为空则回退到 --data_dir）"
    )
    parser.add_argument(
        "--rxrce_data_dir",
        default="",
        help="RXRCE 训练数据目录（为空则不加载 RXRCE）"
    )
    parser.add_argument(
        "--filtered_json",
        default="",
        help="过滤结果 JSON 文件路径（由 filter_results.py 生成）"
    )
    parser.add_argument(
        "--train_mode",
        choices=["mixed", "r2r_only", "rxr_only"],
        default="mixed",
        help="训练数据模式：mixed=R2RCE+RXRCE，r2r_only=仅R2RCE，rxr_only=仅RXRCE"
    )
    parser.add_argument(
        "--output_dir",
        default="PATH/TO/outputs/nig_ft_r2rce_rxrce_mixed_panorama_visual",
        help="模型输出目录"
    )
    parser.add_argument(
        "--logging_dir",
        default="PATH/TO/outputs/nig_ft_logs/r2rce_rxrce_mixed_panorama_visual",
        help="日志目录"
    )

    parser.add_argument("--max_frames", type=int, default=30, help="最大帧数")
    parser.add_argument("--max_samples", type=int, default=0, help="最大样本数（0=全部）")
    parser.add_argument("--max_length", type=int, default=0, help="最大序列长度（0=不限制）")
    parser.add_argument("--shuffle", action="store_true", default=True, help="是否打乱数据")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    parser.add_argument("--batch_size", type=int, default=1, help="单卡 batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--num_epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--lr_scheduler_type", default="cosine", help="学习率调度器类型")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup 比例")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--save_steps", type=int, default=200, help="保存间隔")
    parser.add_argument("--save_total_limit", type=int, default=1, help="最多保留的 checkpoint 数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志间隔")
    parser.add_argument(
        "--resume_from_checkpoint",
        default="",
        help="断点续训的 checkpoint 路径；为空则从头开始"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Panorama NIG Training With Visual Prompt (Filtered Labels)")
    print("=" * 60)
    print(f"Model:          {args.model_name}")
    r2rce_data_dir = args.r2rce_data_dir if args.r2rce_data_dir else args.data_dir
    use_rxrce = bool(args.rxrce_data_dir)
    print(f"Train mode:     {args.train_mode}")
    print(f"R2RCE data dir: {r2rce_data_dir}")
    print(f"RXRCE data dir: {args.rxrce_data_dir if use_rxrce else '(disabled)'}")
    print(f"Filtered JSON:  {args.filtered_json}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Max frames:     {args.max_frames}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Grad accum:     {args.grad_accum}")
    print(f"Epochs:         {args.num_epochs}")
    print(f"Learning rate:  {args.lr}")
    print(f"Save steps:     {args.save_steps}")
    print(f"Save limit:     {args.save_total_limit}")
    print(f"Resume ckpt:    {args.resume_from_checkpoint if args.resume_from_checkpoint else '(none)'}")
    print("=" * 60)

    processor_name = args.processor_name if args.processor_name else args.model_name

    print(f"[INFO] Loading processor from {processor_name}")
    processor = AutoProcessor.from_pretrained(
        processor_name, trust_remote_code=True, local_files_only=True,
    )

    print(f"[INFO] Loading model from {args.model_name}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True, local_files_only=True,
    )

    datasets = []
    if args.train_mode in ("mixed", "r2r_only"):
        if not args.filtered_json or (not os.path.exists(args.filtered_json)):
            raise FileNotFoundError(
                f"R2RCE 模式需要有效 filtered_json: {args.filtered_json}"
            )
        print(f"[INFO] Loading R2RCE dataset from {r2rce_data_dir}")
        if not os.path.exists(r2rce_data_dir):
            raise FileNotFoundError(f"R2RCE data dir 不存在: {r2rce_data_dir}")
        dataset_r2rce = PanoramaFilteredDataset(
            root_dir=r2rce_data_dir,
            filtered_json=args.filtered_json,
            max_frames=args.max_frames,
            max_samples=args.max_samples,
            shuffle=args.shuffle,
            seed=args.seed,
            dataset_type="r2rce",
        )
        datasets.append(dataset_r2rce)

    if args.train_mode in ("mixed", "rxr_only"):
        if not args.rxrce_data_dir:
            raise ValueError("rxr_only/mixed 模式需要提供 --rxrce_data_dir")
        if not os.path.exists(args.rxrce_data_dir):
            raise FileNotFoundError(f"RXRCE data dir 不存在: {args.rxrce_data_dir}")
        print(f"[INFO] Loading RXRCE dataset from {args.rxrce_data_dir}")
        dataset_rxrce = PanoramaFilteredDataset(
            root_dir=args.rxrce_data_dir,
            filtered_json="",
            max_frames=args.max_frames,
            max_samples=args.max_samples,
            shuffle=args.shuffle,
            seed=args.seed,
            dataset_type="rxrce",
        )
        datasets.append(dataset_rxrce)

    if not datasets:
        raise RuntimeError("没有可用训练数据集，请检查 train_mode 与数据路径")

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    max_length = args.max_length if args.max_length > 0 else None
    collator = PanoramaCollator(processor, max_length=max_length)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_dir=args.logging_dir,
        fp16=False,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("[INFO] Starting training...")
    print(f"[INFO] Dataset size: {len(dataset)} samples")
    print(f"[INFO] Effective batch size: {args.batch_size * args.grad_accum * max(1, torch.cuda.device_count())}")
    if args.resume_from_checkpoint:
        print(f"[INFO] Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    print(f"[INFO] Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    print("[INFO] Training complete!")


if __name__ == "__main__":
    main()
