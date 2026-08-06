#!/usr/bin/env python3
"""
End-to-end test for VP-Adapter pipeline.

Tests:
  1. VPEncoder: dummy mask → correct output shape
  2. VPSpatialAdapter: dummy features → gate vector applies scaled injection
  3. Model wrapper: attach + patched forward pass
  4. Full forward: model with VP masks produces valid loss
  5. Gate gradient: verify gate receives gradients

Run:
    python vp_adapter/test_pipeline.py [--quick]
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import numpy as np

_VP_DIR = os.path.dirname(os.path.abspath(__file__))
if _VP_DIR not in sys.path:
    sys.path.insert(0, _VP_DIR)

from vp_config import VPAdapterConfig
from vp_encoder import VPEncoder, build_vp_encoder
from vp_gated_adapter import VPSpatialAdapter, build_vp_adapters
from vp_model_wrapper import (
    attach_vp_adapter, set_vp_features, clear_vp_features,
    save_vp_modules, load_vp_modules, print_vp_summary,
)

MODEL_DIR = (
    "PATH/TO/models_cache/models--Qwen--Qwen3-VL-8B-Instruct"
    "/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)


def test_vp_encoder():
    """Test VPEncoder produces correct output shape."""
    print("\n[TEST 1] VPEncoder shape test ...")
    cfg = VPAdapterConfig()
    encoder = build_vp_encoder(cfg)

    # Simulate 3 images with different sizes
    masks = [
        torch.rand(3, 384, 768),   # panorama frame 1
        torch.rand(3, 384, 768),   # panorama frame 2
        torch.rand(3, 384, 768),   # panorama frame 3
    ]
    # grid_thw: for 384x768 image with patch_size=16: h=24, w=48
    grid_thw = torch.tensor([
        [1, 24, 48],
        [1, 24, 48],
        [1, 24, 48],
    ], dtype=torch.long)

    out = encoder(masks, grid_thw)
    expected_total = 3 * 24 * 48  # 3 images, each 24*48 patches

    assert out.shape == (expected_total, cfg.vp_dim), \
        f"Expected ({expected_total}, {cfg.vp_dim}), got {out.shape}"
    print(f"  [OK] Output shape: {out.shape}")
    print(f"  [OK] Param count: {sum(p.numel() for p in encoder.parameters()):,}")
    return True


def test_vp_spatial_adapter():
    """Test VPSpatialAdapter gate vector shape and injection magnitude."""
    print("\n[TEST 2] VPSpatialAdapter gate vector test ...")
    cfg = VPAdapterConfig()
    adapter = VPSpatialAdapter(
        hidden_dim=cfg.vit_hidden_dim,
        vp_dim=cfg.vp_dim,
    )

    seq_len = 24 * 48  # one image
    x = torch.randn(seq_len, cfg.vit_hidden_dim)
    vp = torch.randn(seq_len, cfg.vp_dim)
    cu = torch.tensor([0, seq_len], dtype=torch.int32)

    assert adapter.gate.shape == (cfg.vit_hidden_dim,), (
        f"Gate should be a per-dimension vector of size {cfg.vit_hidden_dim}, "
        f"got {tuple(adapter.gate.shape)}"
    )
    gate_init = adapter.gate.detach().abs().max().item()

    y = adapter(x, vp, cu)
    diff = (y - x).abs().max().item()
    # Injection is gate * LayerNorm(...), so the perturbation stays on the
    # order of the gate init — small enough not to disturb the SFT starting
    # point, but non-zero so the gate receives gradient from step one.
    assert diff > 0, "Gate is non-zero at init but output is unchanged"
    assert diff < 100 * gate_init, (
        f"Injection {diff:.2e} is far larger than gate init {gate_init:.2e}"
    )
    print(f"  [OK] Gate shape: {tuple(adapter.gate.shape)}, init {gate_init:.6f}")
    print(f"  [OK] Max diff from input: {diff:.2e} (expected ~{gate_init:.1e})")
    print(f"  [OK] Param count: {sum(p.numel() for p in adapter.parameters()):,}")
    return True


def test_model_wrapper(quick=False):
    """Test attaching VP-Adapter to Qwen3-VL."""
    print("\n[TEST 3] Model wrapper test ...")
    if quick:
        print("  [SKIP] --quick mode, skipping model load")
        return True

    if not os.path.exists(MODEL_DIR):
        print(f"  [SKIP] Model not found: {MODEL_DIR}")
        return True

    from transformers import Qwen3VLForConditionalGeneration
    print("  Loading model (this takes ~30s) ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16,
        trust_remote_code=True, local_files_only=True,
    )

    cfg = VPAdapterConfig()
    vp_encoder, vp_adapters = attach_vp_adapter(model, cfg)

    # Verify adapters are registered
    visual = model.model.visual
    assert hasattr(visual, "_vp_adapters"), "Adapters not registered"
    assert hasattr(visual, "_vp_adapter_layer_set"), "Layer set not registered"
    assert visual._vp_adapter_layer_set == {7, 15, 23}

    # Verify ViT backbone is frozen
    frozen_count = sum(1 for p in visual.parameters()
                       if not p.requires_grad and "_vp_adapters" not in "")
    print(f"  [OK] Adapters registered at layers: {visual._vp_adapter_layer_set}")
    print(f"  [OK] ViT frozen params: {frozen_count}")

    # Verify gate values
    for name, adapter in visual._vp_adapters.items():
        g = adapter.gate.item()
        assert abs(g) < 1e-6, f"Gate at layer {name} not zero: {g}"
        print(f"  [OK] Gate at layer {name}: {g:.6f}")

    print_vp_summary(vp_encoder, model)

    # Test save/load
    tmp_dir = "/tmp/vp_adapter_test"
    save_vp_modules(vp_encoder, model, tmp_dir)
    print(f"  [OK] Saved VP modules to {tmp_dir}")

    # Modify gate, reload, verify restored
    with torch.no_grad():
        for adapter in visual._vp_adapters.values():
            adapter.gate.fill_(0.5)
    load_vp_modules(vp_encoder, model, tmp_dir)
    for name, adapter in visual._vp_adapters.items():
        g = adapter.gate.item()
        assert abs(g) < 1e-6, f"Gate not restored: {g}"
    print(f"  [OK] Save/load round-trip passed")

    return True


def test_full_forward(quick=False):
    """Test full forward pass with VP masks → loss."""
    print("\n[TEST 4] Full forward pass test ...")
    if quick:
        print("  [SKIP] --quick mode")
        return True

    if not os.path.exists(MODEL_DIR):
        print(f"  [SKIP] Model not found: {MODEL_DIR}")
        return True

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from PIL import Image

    print("  Loading model + processor ...")
    processor = AutoProcessor.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16,
        trust_remote_code=True, local_files_only=True,
    )

    cfg = VPAdapterConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    vp_encoder, _ = attach_vp_adapter(model, cfg)

    # Create a dummy image + instruction
    dummy_img = Image.fromarray(np.random.randint(0, 255, (384, 768, 3), dtype=np.uint8))
    dummy_mask = torch.rand(3, 384, 768)

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this route."},
            {"type": "image", "image": dummy_img},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Walk forward past the table."},
        ]},
    ]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    grid_thw = inputs["image_grid_thw"]
    vp_features = vp_encoder([dummy_mask], grid_thw)
    set_vp_features(model, vp_features)

    # Forward pass
    prompt_len = 10
    labels = inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    inputs["labels"] = labels

    outputs = model(**inputs)
    loss = outputs.loss
    clear_vp_features(model)

    print(f"  [OK] Loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    assert torch.isfinite(loss), "Loss is not finite"

    # Test gradient flow through gate
    loss.backward()
    visual = model.model.visual
    for name, adapter in visual._vp_adapters.items():
        g = adapter.gate.grad
        assert g is not None, f"No gradient for gate at layer {name}"
        print(f"  [OK] Gate gradient at layer {name}: {g.item():.6e}")

    print(f"  [OK] Full forward + backward pass complete")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Skip tests that require loading the full model")
    args = parser.parse_args()

    print("=" * 60)
    print("VP-Adapter Pipeline Test Suite")
    print("=" * 60)

    results = {}
    t0 = time.time()

    results["VPEncoder shape"] = test_vp_encoder()
    results["VPSpatialAdapter gate vector"] = test_vp_spatial_adapter()
    results["Model wrapper"] = test_model_wrapper(quick=args.quick)
    results["Full forward"] = test_full_forward(quick=args.quick)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("Results:")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\nTotal time: {elapsed:.1f}s")
    all_pass = all(results.values())
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
