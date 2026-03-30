from __future__ import annotations
"""Export a PyTorch checkpoint to ONNX format."""

from pathlib import Path

import torch
import onnx

from .config import _BaseConfig
from .model import HexChessNet, build_model


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    config: _BaseConfig | None = None,
) -> Path:
    """
    Export a PyTorch model checkpoint to ONNX.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        output_path: Where to write the .onnx file.
        config: Configuration (for model architecture).

    Returns:
        The output_path on success.
    """
    cfg = config or _BaseConfig()

    # Load the model
    model = build_model(cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    # Dummy input matching the engine's board encoding shape
    dummy_input = torch.randn(
        1, cfg.board_channels, cfg.board_height, cfg.board_width
    )

    # Export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["board"],
        output_names=["policy", "value"],
        dynamic_axes={
            "board": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"},
        },
        opset_version=17,
    )

    # Re-save with all weights embedded in a single file (torch.onnx.export
    # may create a separate .data file for large models).
    model_proto = onnx.load(str(output_path))
    onnx.save_model(model_proto, str(output_path), save_as_external_data=False)

    # Clean up any leftover external data file
    data_file = output_path.parent / (output_path.name + ".data")
    if data_file.exists():
        data_file.unlink()

    print(f"Exported ONNX model to {output_path}")

    # Verify the exported model
    verify_onnx(output_path)

    return output_path


def verify_onnx(onnx_path: Path) -> None:
    """Load and check the exported ONNX model."""
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print(f"ONNX model verification passed: {onnx_path}")

    # Optional: run a quick inference test with onnxruntime
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(str(onnx_path))
        cfg = _BaseConfig()
        dummy = np.random.randn(1, cfg.board_channels, cfg.board_height, cfg.board_width).astype(np.float32)
        outputs = session.run(None, {"board": dummy})
        policy, value = outputs
        print(f"  Inference test: policy shape={policy.shape}, WDL shape={value.shape}")
        print(f"  WDL logits: {value[0]}")
    except ImportError:
        print("  (onnxruntime not installed, skipping inference test)")
