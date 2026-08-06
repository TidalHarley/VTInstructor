#!/usr/bin/env python3
"""
SFT training script with VP-Adapter v3.0.

Changes from v2.0:
  - Cross-attention → content-adaptive spatial modulation
  - Each ViT token modulated only by its aligned VP token (1:1, no dilution)
  - Gate MLP sees both image and VP content for conditional gating

Usage:
    python -m torch.distributed.run --nproc_per_node=8 \
        vp_adapter/train_sft_with_vp_adapter.py \
        --r2rce_data_dir /path/to/r2rce_train_visual \
        --rxrce_data_dir /path/to/rxrce_train_visual \
        --filtered_json /path/to/filtered_ult_gt6.json \
        --output_dir /path/to/output
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

# ── VP-Adapter imports ──
_VP_ADAPTER_DIR = os.path.dirname(os.path.abspath(__file__))    # turn to the absolute path and pick the dir
if _VP_ADAPTER_DIR not in sys.path:
    sys.path.insert(0, _VP_ADAPTER_DIR)

from vp_config import VPAdapterConfig
from vp_model_wrapper import (
    attach_vp_adapter,
    set_vp_features,
    clear_vp_features,
    save_vp_modules,
    print_vp_summary,
    get_vp_trainable_params,
)
from vp_dataset import VPDataset, VPCollator


def _get_vp_encoder(model):
    """Get VPEncoder from model, handling DDP / DeepSpeed wrappers. When single GPU, model is the model
        itself, so we don't need to unwrap it. When multi-GPU, model is the DDP wrapper, so we need to unwrap it.
    """
    unwrapped = getattr(model, "module", model)
    return getattr(unwrapped, "_vp_encoder", None)


class VPTrainer(Trainer):
    """
    Custom Trainer that handles VP mask → VPEncoder → set_vp_features
    before each forward pass, and uses separate LR for VP modules.

    VPEncoder is registered as model._vp_encoder by attach_vp_adapter,
    so its parameters are included in the Trainer's optimizer and DDP
    gradient sync automatically.
    """

    def __init__(self, vp_module_lr: float = 2e-5, **kwargs):
        self.vp_module_lr = vp_module_lr
        super().__init__(**kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model

        # Classify params: VP vs non-VP, decay vs no-decay
        vp_param_ids = set()
        unwrapped = getattr(model, "module", model)
        vp_enc = getattr(unwrapped, "_vp_encoder", None)
        if vp_enc is not None:
            for p in vp_enc.parameters():
                vp_param_ids.add(id(p))
        visual = unwrapped.model.visual
        if hasattr(visual, "_vp_adapters"):
            for p in visual._vp_adapters.parameters():
                vp_param_ids.add(id(p))

        no_decay_keywords = ("bias", "layer_norm", "layernorm", "norm")
        vp_decay, vp_no_decay, lm_decay, lm_no_decay = [], [], [], []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_no_decay = any(kw in name.lower() for kw in no_decay_keywords)
            is_vp = id(param) in vp_param_ids

            if is_vp:
                (vp_no_decay if is_no_decay else vp_decay).append(param)
            else:
                (lm_no_decay if is_no_decay else lm_decay).append(param)

        wd = self.args.weight_decay
        base_lr = self.args.learning_rate

        optimizer_grouped_parameters = []
        if vp_decay:
            optimizer_grouped_parameters.append(
                {"params": vp_decay, "lr": self.vp_module_lr, "weight_decay": wd})
        if vp_no_decay:
            optimizer_grouped_parameters.append(
                {"params": vp_no_decay, "lr": self.vp_module_lr, "weight_decay": 0.0})
        if lm_decay:
            optimizer_grouped_parameters.append(
                {"params": lm_decay, "lr": base_lr, "weight_decay": wd})
        if lm_no_decay:
            optimizer_grouped_parameters.append(
                {"params": lm_no_decay, "lr": base_lr, "weight_decay": 0.0})

        from torch.optim import AdamW
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=base_lr, betas=(0.9, 0.999), eps=1e-8)

        n_vp = len(vp_decay) + len(vp_no_decay)
        n_lm = len(lm_decay) + len(lm_no_decay)
        print(f"[VP] Optimizer: {n_vp} VP param tensors (lr={self.vp_module_lr}), "
              f"{n_lm} LM param tensors (lr={base_lr})")

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        vp_masks = inputs.pop("vp_masks", None)

        vp_encoder = _get_vp_encoder(model)
        if vp_masks is not None and vp_encoder is not None:
            grid_thw = inputs.get("image_grid_thw")
            if grid_thw is not None:
                device = next(model.parameters()).device
                vp_features = vp_encoder(vp_masks, grid_thw.to(device))
                set_vp_features(model, vp_features)

        loss = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        clear_vp_features(model)
        return loss

    def save_model(self, output_dir=None, _internal_call=False):
        """Save both the base model and VP modules."""
        super().save_model(output_dir, _internal_call)
        save_dir = output_dir or self.args.output_dir
        vp_encoder = _get_vp_encoder(self.model)
        if vp_encoder is not None:
            save_vp_modules(vp_encoder, self.model, save_dir)
            print(f"[VP] Saved VP modules to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="SFT with VP-Adapter")

    # Model
    parser.add_argument("--model_name", default=(
        "PATH/TO/models_cache/models--Qwen--Qwen3-VL-8B-Instruct"
        "/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"))
    parser.add_argument("--processor_name", default="")

    # Data
    parser.add_argument("--r2rce_data_dir", required=True)
    parser.add_argument("--rxrce_data_dir", default="")
    parser.add_argument("--filtered_json", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--logging_dir", default="")

    # VP-Adapter config (v2.0 defaults)
    parser.add_argument("--vp_dim", type=int, default=384)
    parser.add_argument("--adapter_layers", default="7",
                        help="Comma-separated ViT layer indices for adapter insertion")
    parser.add_argument("--adapter_num_heads", type=int, default=4)
    parser.add_argument("--freeze_vit", action="store_true", default=False)
    parser.add_argument("--no_freeze_vit", action="store_true")
    parser.add_argument("--vp_module_lr", type=float, default=2e-5)

    # Training
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=12)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--logging_steps", type=int, default=10)

    args = parser.parse_args()

    if args.no_freeze_vit:
        args.freeze_vit = False

    processor_name = args.processor_name or args.model_name
    if not args.logging_dir:
        args.logging_dir = os.path.join(args.output_dir, "logs")

    # ── VP-Adapter Config ──
    adapter_layers = tuple(int(x) for x in args.adapter_layers.split(","))
    vp_cfg = VPAdapterConfig(
        vp_dim=args.vp_dim,
        adapter_layers=adapter_layers,
        adapter_num_heads=args.adapter_num_heads,
        freeze_vit_backbone=args.freeze_vit,
        vp_module_lr=args.vp_module_lr,
        lm_lr=args.lr,
    )

    print("=" * 60)
    print("SFT with VP-Adapter")
    print("=" * 60)
    print(f"Model:           {args.model_name}")
    print(f"R2RCE data:      {args.r2rce_data_dir}")
    print(f"RXRCE data:      {args.rxrce_data_dir or '(none)'}")
    print(f"Adapter layers:  {adapter_layers}")
    print(f"VP dim:          {args.vp_dim}")
    print(f"Freeze ViT:      {args.freeze_vit}")
    print(f"VP module LR:    {args.vp_module_lr}")
    print(f"LM LR:           {args.lr}")
    print("=" * 60)

    # ── Load processor ──
    processor = AutoProcessor.from_pretrained(
        processor_name, trust_remote_code=True, local_files_only=True)

    # ── Load model ──
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True, local_files_only=True,
    )

    # ── Attach VP-Adapter (registers encoder + adapters on the model) ──
    vp_encoder, vp_adapters = attach_vp_adapter(model, vp_cfg)
    print_vp_summary(vp_encoder, model)

    # ── Datasets ──
    datasets = []
    if args.r2rce_data_dir and os.path.isdir(args.r2rce_data_dir):
        ds_r2r = VPDataset(
            root_dir=args.r2rce_data_dir,
            filtered_json=args.filtered_json,
            max_frames=args.max_frames,
            max_samples=args.max_samples,
            shuffle=args.shuffle, seed=args.seed,
            dataset_type="r2rce",
        )
        datasets.append(ds_r2r)

    if args.rxrce_data_dir and os.path.isdir(args.rxrce_data_dir):
        ds_rxr = VPDataset(
            root_dir=args.rxrce_data_dir,
            filtered_json="",
            max_frames=args.max_frames,
            max_samples=args.max_samples,
            shuffle=args.shuffle, seed=args.seed,
            dataset_type="rxrce",
        )
        datasets.append(ds_rxr)

    if not datasets:
        raise RuntimeError("No datasets found")

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    max_length = args.max_length if args.max_length > 0 else None
    collator = VPCollator(processor, max_length=max_length)

    # ── Training args ──
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
        dataloader_num_workers=0,
        report_to=["tensorboard"],
    )

    # ── Trainer ──
    trainer = VPTrainer(
        vp_module_lr=args.vp_module_lr,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print(f"[INFO] Dataset size: {len(dataset)}")
    print(f"[INFO] Effective batch: "
          f"{args.batch_size * args.grad_accum * max(1, torch.cuda.device_count())}")
    print("[INFO] Starting SFT with VP-Adapter ...")
    trainer.train()

    print(f"[INFO] Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)

    # Print final gate values
    print_vp_summary(vp_encoder, model)
    print("[DONE] SFT with VP-Adapter complete!")


if __name__ == "__main__":
    main()
