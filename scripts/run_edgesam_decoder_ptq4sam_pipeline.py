from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from edge_sam.quantization.decoder_ptq_compare import _build_decoder_surface, collect_decoder_sample_triplets
from scripts.edgesam_decoder_ptq4sam_uint8 import (
    calibrate_decoder,
    evaluate_decoder,
    make_qconfig,
    move_samples_to_device,
    quantize_decoder_surface,
)
from scripts.export_edgesam_decoder_ptq4sam_onnx import (
    _print_onnx_summary,
    export_pe_gaussian_matrix,
    export_quantized_decoder_to_onnx,
    resolve_output_path,
)
from ptq4sam.quantization.state import enable_quantization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-shot EdgeSAM decoder PTQ4SAM pipeline: load -> quantize -> evaluate -> export ONNX."
    )
    parser.add_argument(
        "--checkpoint",
        default="/home/kitemanul/project/EdgeSAM/weights/edge_sam.pth",
        help="Path to EdgeSAM checkpoint (.pth)",
    )
    parser.add_argument(
        "--calibration-list",
        default="/home/kitemanul/project/EdgeSAM/weights/one_pipeline/calib/decoder_datalist_20.txt",
        help="Decoder calibration datalist",
    )
    parser.add_argument(
        "--eval-list",
        default="/home/kitemanul/project/EdgeSAM/weights/one_pipeline/calib/decoder_test_5.txt",
        help="Decoder evaluation datalist",
    )
    parser.add_argument("--calibration-count", type=int, default=20)
    parser.add_argument("--eval-count", type=int, default=5)
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument(
        "--scope",
        choices=("transformer", "full"),
        default="transformer",
        help="Quantization boundary",
    )
    parser.add_argument("--disable-big", action="store_true")
    parser.add_argument("--disable-agq", action="store_true")

    parser.add_argument(
        "--summary-output-dir",
        default=None,
        help="Directory to save summary.json; defaults to PTQ4SAM/results/<timestamp>",
    )
    parser.add_argument(
        "--onnx-output",
        default=None,
        help="Output ONNX path; defaults to <checkpoint>_decoder_ptq4sam_uint8_<scope>.onnx",
    )
    parser.add_argument("--num-points", type=int, default=5, help="Static prompt count for exported ONNX")
    parser.add_argument("--opset-version", type=int, default=11)
    parser.add_argument(
        "--check-ops-only",
        action="store_true",
        help="Export and print ONNX summary, then remove the ONNX file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because PTQ4SAM BIG/AGQ calibration code uses CUDA tensors")

    checkpoint_path = Path(args.checkpoint).resolve()
    calibration_paths = collect_decoder_sample_triplets(args.calibration_list, limit=args.calibration_count)
    eval_paths = collect_decoder_sample_triplets(args.eval_list, limit=args.eval_count)

    if len(calibration_paths) < args.calibration_count:
        raise ValueError(f"Requested {args.calibration_count} calibration samples but found {len(calibration_paths)}")
    if len(eval_paths) < args.eval_count:
        raise ValueError(f"Requested {args.eval_count} eval samples but found {len(eval_paths)}")

    config_quant = make_qconfig(bit=args.bit)
    config_quant.ptq4sam.BIG = not args.disable_big
    config_quant.ptq4sam.AGQ = not args.disable_agq

    device = torch.device("cuda:0")

    fp_model = _build_decoder_surface(checkpoint_path, use_stability_score=True).to(device).eval()
    quant_model = _build_decoder_surface(checkpoint_path, use_stability_score=True).to(device).eval()
    quant_model = quantize_decoder_surface(quant_model, config_quant, scope=args.scope).to(device).eval()

    calibration_samples = move_samples_to_device(calibration_paths, device)
    calibrate_decoder(quant_model, calibration_samples, config_quant.ptq4sam.BIG)
    enable_quantization(quant_model)

    bundle = evaluate_decoder(fp_model, quant_model, eval_paths)

    if args.summary_output_dir is not None:
        summary_output_dir = Path(args.summary_output_dir)
    else:
        summary_output_dir = (
            Path("results")
            / "edgesam_decoder_ptq4sam_uint8"
            / args.scope
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    summary_output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint": str(checkpoint_path),
        "calibration_list": str(Path(args.calibration_list).resolve()),
        "eval_list": str(Path(args.eval_list).resolve()),
        "calibration_count": args.calibration_count,
        "eval_count": args.eval_count,
        "bit": args.bit,
        "scope": args.scope,
        "big": config_quant.ptq4sam.BIG,
        "agq": config_quant.ptq4sam.AGQ,
        "mean_metrics": bundle.mean_metrics,
        "per_sample": bundle.per_sample,
    }
    summary_path = summary_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["mean_metrics"], indent=2))
    print(f"Saved summary to {summary_path}")

    onnx_path = resolve_output_path(str(checkpoint_path), args.onnx_output, args.scope)
    export_quantized_decoder_to_onnx(
        quant_model,
        output_path=onnx_path,
        num_points=args.num_points,
        opset_version=args.opset_version,
    )
    pe_path = export_pe_gaussian_matrix(str(checkpoint_path), onnx_path)
    print(f"Exported quantized decoder ONNX to {onnx_path}")
    print(f"Exported PE gaussian matrix to {pe_path}")
    _print_onnx_summary(onnx_path)

    if args.check_ops_only and onnx_path.exists():
        onnx_path.unlink()
        print(f"Removed {onnx_path} (check-ops-only mode)")


if __name__ == "__main__":
    main()
