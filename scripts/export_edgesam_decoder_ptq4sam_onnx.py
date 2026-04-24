from __future__ import annotations

import argparse
import os
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def ensure_wandb_stub() -> None:
    if "wandb" not in sys.modules:
        sys.modules["wandb"] = types.ModuleType("wandb")


ensure_wandb_stub()

from edge_sam import sam_model_registry
from edge_sam.quantization.decoder_ptq_compare import _build_decoder_surface, collect_decoder_sample_triplets
from ptq4sam.quantization.fake_quant import AdaptiveGranularityQuantize
from scripts.edgesam_decoder_ptq4sam_uint8 import (
    calibrate_decoder,
    make_qconfig,
    move_samples_to_device,
    quantize_decoder_surface,
)
from ptq4sam.quantization.state import enable_quantization


def resolve_output_path(checkpoint: str, output: str | None, scope: str) -> Path:
    if output is not None:
        return Path(output)
    checkpoint_path = Path(checkpoint)
    stem = checkpoint_path.stem
    return checkpoint_path.with_name(f"{stem}_decoder_ptq4sam_uint8_{scope}.onnx")


def export_pe_gaussian_matrix(checkpoint: str, output_path: Path) -> Path:
    sam = sam_model_registry["edge_sam"](checkpoint=checkpoint, upsample_mode="bilinear")
    sam.eval()
    pe_matrix = sam.prompt_encoder.pe_layer.positional_encoding_gaussian_matrix.cpu().numpy()
    pe_path = output_path.parent / "pe_gaussian_matrix.bin"
    pe_matrix.astype(np.float32).tofile(pe_path)
    return pe_path


def _onnx_export(*args, **kwargs) -> None:
    parts = torch.__version__.split(".")
    major, minor = int(parts[0]), int(parts[1])
    if major > 2 or (major == 2 and minor >= 6):
        kwargs["dynamo"] = False
    torch.onnx.export(*args, **kwargs)


def prepare_quantized_decoder_for_onnx_export(decoder: torch.nn.Module) -> torch.nn.Module:
    for module in decoder.modules():
        if isinstance(module, AdaptiveGranularityQuantize) and not hasattr(module, "zero_point"):
            module.register_buffer("zero_point", torch.tensor([0], dtype=torch.int))
    return decoder


def _print_onnx_summary(onnx_path: Path) -> None:
    try:
        import onnx
    except ImportError:
        print("Cannot print summary: onnx not installed")
        return

    dtype_names = {
        1: "FLOAT",
        2: "UINT8",
        3: "INT8",
        5: "INT16",
        6: "INT32",
        7: "INT64",
        9: "BOOL",
        10: "FLOAT16",
    }
    model = onnx.load(str(onnx_path))
    ops = Counter(node.op_type for node in model.graph.node)

    dtypes = set()
    for init in model.graph.initializer:
        dtypes.add(dtype_names.get(init.data_type, "?"))
    for inp in model.graph.input:
        dtypes.add(dtype_names.get(inp.type.tensor_type.elem_type, "?"))
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.name == "value" and hasattr(attr, "t") and attr.t.data_type:
                dtypes.add(dtype_names.get(attr.t.data_type, "?"))

    print(f"\n{'=' * 60}")
    print(f"  {onnx_path.name}")
    print(f"{'=' * 60}")
    print(f"  Nodes: {len(model.graph.node)},  Op types: {len(ops)}")
    print(f"  Ops: {', '.join(f'{op}({cnt})' for op, cnt in sorted(ops.items()))}")
    print(f"  Dtypes: {', '.join(sorted(dtypes))}")
    for inp in model.graph.input:
        shape = [dim.dim_value or dim.dim_param for dim in inp.type.tensor_type.shape.dim]
        dt = dtype_names.get(inp.type.tensor_type.elem_type, "?")
        print(f"  Input  {inp.name}: {dt} {shape}")
    for out in model.graph.output:
        shape = [dim.dim_value or dim.dim_param for dim in out.type.tensor_type.shape.dim]
        dt = dtype_names.get(out.type.tensor_type.elem_type, "?")
        print(f"  Output {out.name}: {dt} {shape}")

    known_npu_bad = {"Sin", "Cos", "Abs", "Gather", "GatherND", "ScatterND", "NonZero", "Greater"}
    likely_quant_bad = {"Round", "Clip", "Log"}
    found = (known_npu_bad | likely_quant_bad) & set(ops.keys())
    if found:
        print(f"  WARNING: ops needing review for NPU compilation: {', '.join(sorted(found))}")
    if "INT64" in dtypes:
        print("  WARNING: INT64 data type present")
    print(f"{'=' * 60}")


def build_quantized_decoder_for_export(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    checkpoint_path = Path(args.checkpoint).resolve()
    calibration_paths = collect_decoder_sample_triplets(args.calibration_list, limit=args.calibration_count)
    if len(calibration_paths) < args.calibration_count:
        raise ValueError(
            f"Requested {args.calibration_count} calibration samples but found {len(calibration_paths)}"
        )

    config_quant = make_qconfig(bit=args.bit)
    config_quant.ptq4sam.BIG = not args.disable_big
    config_quant.ptq4sam.AGQ = not args.disable_agq

    decoder = _build_decoder_surface(checkpoint_path, use_stability_score=args.use_stability_score).to(device).eval()
    decoder = quantize_decoder_surface(decoder, config_quant, scope=args.scope).to(device).eval()

    calibration_samples = move_samples_to_device(calibration_paths, device)
    calibrate_decoder(decoder, calibration_samples, config_quant.ptq4sam.BIG)
    enable_quantization(decoder)
    return decoder


def export_quantized_decoder_to_onnx(
    decoder: torch.nn.Module,
    output_path: Path,
    *,
    num_points: int,
    opset_version: int,
    dummy_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> None:
    decoder = prepare_quantized_decoder_for_onnx_export(decoder)
    if dummy_inputs is None:
        embed_dim = getattr(decoder, "dense_embedding").shape[1]
        embed_h = int(getattr(decoder, "embed_h", getattr(decoder, "dense_embedding").shape[2]))
        embed_w = int(getattr(decoder, "embed_w", getattr(decoder, "dense_embedding").shape[3]))
        dummy_embeddings = torch.randn(1, embed_dim, embed_h, embed_w, dtype=torch.float32)
        dummy_pe = torch.randn(1, num_points, embed_dim, dtype=torch.float32)
        dummy_labels = torch.randint(0, 4, (1, num_points), dtype=torch.int64).to(torch.float32)
    else:
        dummy_embeddings, dummy_pe, dummy_labels = dummy_inputs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        _onnx_export(
            decoder.cpu().eval(),
            (dummy_embeddings, dummy_pe, dummy_labels),
            str(output_path),
            input_names=["image_embeddings", "point_embedding_pe", "point_labels"],
            output_names=["scores", "masks"],
            opset_version=opset_version,
            verbose=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PTQ4SAM-style EdgeSAM decoder fake-quant ONNX with optional BIG/AGQ."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="/home/kitemanul/project/EdgeSAM/weights/edge_sam.pth",
        help="Path to EdgeSAM checkpoint (.pth)",
    )
    parser.add_argument(
        "--calibration-list",
        default="/home/kitemanul/project/EdgeSAM/weights/one_pipeline/calib/decoder_datalist_20.txt",
        help="Decoder calibration datalist",
    )
    parser.add_argument("--calibration-count", type=int, default=20)
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument(
        "--scope",
        choices=("transformer", "full"),
        default="transformer",
        help="Quantization boundary to export",
    )
    parser.add_argument("--disable-big", action="store_true")
    parser.add_argument("--disable-agq", action="store_true")
    parser.add_argument("--use-stability-score", action="store_true", default=True)
    parser.add_argument("--num-points", type=int, default=5, help="Static prompt count for exported ONNX")
    parser.add_argument("--opset-version", type=int, default=11)
    parser.add_argument("--output", type=str, default=None, help="Output ONNX path")
    parser.add_argument(
        "--check-ops-only",
        action="store_true",
        help="Export, print ONNX summary, then delete the file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because PTQ4SAM BIG/AGQ calibration code uses CUDA tensors")

    device = torch.device("cuda:0")
    output_path = resolve_output_path(args.checkpoint, args.output, args.scope)
    decoder = build_quantized_decoder_for_export(args, device=device)
    export_quantized_decoder_to_onnx(
        decoder,
        output_path,
        num_points=args.num_points,
        opset_version=args.opset_version,
    )
    pe_path = export_pe_gaussian_matrix(args.checkpoint, output_path)
    print(f"Exported quantized decoder ONNX to {output_path}")
    print(f"Exported PE gaussian matrix to {pe_path}")
    print("Note: this is a PTQ4SAM fake-quant/emulated uint8 ONNX, not a QDQ-typed INT8 graph.")
    _print_onnx_summary(output_path)
    if args.check_ops_only:
        os.remove(output_path)
        print(f"\n(Removed {output_path} — check-ops-only mode)")


if __name__ == "__main__":
    main()
