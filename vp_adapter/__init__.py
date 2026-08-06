"""
VP-Adapter: Visual Prompt-Guided Gated Cross-Attention Adapter for Qwen3-VL.

Injects structured VP spatial information (ribbon / arrow / endpoint masks)
into the vision encoder via lightweight gated cross-attention modules,
enabling the model to explicitly ground instruction generation in
navigation visual prompts.
"""
