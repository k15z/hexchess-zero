from __future__ import annotations
"""Export a PyTorch checkpoint to ONNX format."""

from pathlib import Path

import torch
import onnx

from .config import Config
from .model import HexChessNet, build_model


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    config: Config | None = None,
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
    cfg = config or Config()

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
        dummy = np.random.randn(1, 16, 11, 11).astype(np.float32)
        outputs = session.run(None, {"board": dummy})
        policy, value = outputs
        print(f"  Inference test: policy shape={policy.shape}, value shape={value.shape}")
        print(f"  Value: {value[0][0]:.4f}")
    except ImportError:
        print("  (onnxruntime not installed, skipping inference test)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint (default: models/checkpoints/latest.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .onnx path (default: models/latest.onnx)",
    )
    args = parser.parse_args()

    cfg = Config()
    checkpoint = Path(args.checkpoint) if args.checkpoint else cfg.checkpoint_dir / "latest.pt"
    output = Path(args.output) if args.output else cfg.model_dir / "latest.onnx"

    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        print("Train a model first with: python -m training.trainer")
    else:
        export_to_onnx(checkpoint, output, cfg)
