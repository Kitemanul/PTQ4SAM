from __future__ import annotations

import argparse
import json
import sys
import types
from datetime import datetime
from pathlib import Path
import time

import torch

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def ensure_wandb_stub() -> None:
    if "wandb" not in sys.modules:
        sys.modules["wandb"] = types.ModuleType("wandb")


ensure_wandb_stub()

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


def log_progress(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


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
    parser.add_argument(
        "--image-embeddings-shape",
        type=int,
        nargs=4,
        metavar=("B", "C", "H", "W"),
        required=True,
        help="Dummy input shape for image_embeddings",
    )
    parser.add_argument(
        "--point-embedding-pe-shape",
        type=int,
        nargs=3,
        metavar=("B", "N", "C"),
        required=True,
        help="Dummy input shape for point_embedding_pe",
    )
    parser.add_argument(
        "--point-labels-shape",
        type=int,
        nargs=2,
        metavar=("B", "N"),
        required=True,
        help="Dummy input shape for point_labels",
    )
    parser.add_argument("--opset-version", type=int, default=11)
    parser.add_argument(
        "--check-ops-only",
        action="store_true",
        help="Export and print ONNX summary, then remove the ONNX file",
    )
    return parser.parse_args()


def build_dummy_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_b, image_c, image_h, image_w = args.image_embeddings_shape
    pe_b, pe_n, pe_c = args.point_embedding_pe_shape
    labels_b, labels_n = args.point_labels_shape

    if image_b != pe_b or image_b != labels_b:
        raise ValueError("Batch size mismatch across image_embeddings, point_embedding_pe, and point_labels")
    if pe_n != labels_n:
        raise ValueError("Point count mismatch between point_embedding_pe and point_labels")
    if image_c != pe_c:
        raise ValueError("Channel mismatch between image_embeddings and point_embedding_pe")

    dummy_embeddings = torch.randn(image_b, image_c, image_h, image_w, dtype=torch.float32)
    dummy_pe = torch.randn(pe_b, pe_n, pe_c, dtype=torch.float32)
    dummy_labels = torch.randint(0, 4, (labels_b, labels_n), dtype=torch.int64).to(torch.float32)
    return dummy_embeddings, dummy_pe, dummy_labels


def adapt_decoder_dense_embedding_size(decoder: torch.nn.Module, image_h: int, image_w: int) -> None:
    dense_embedding = getattr(decoder, "dense_embedding", None)
    if dense_embedding is None:
        raise AttributeError("Decoder has no dense_embedding; cannot adapt export input spatial size")

    current_h = int(dense_embedding.shape[2])
    current_w = int(dense_embedding.shape[3])
    if current_h == image_h and current_w == image_w:
        return

    resized = torch.nn.functional.interpolate(
        dense_embedding.detach(),
        size=(image_h, image_w),
        mode="bilinear",
        align_corners=False,
    )
    resized = resized.to(device=dense_embedding.device, dtype=dense_embedding.dtype).contiguous()

    # Buffer shape changes from e.g. [1, C, 1, 1] to [1, C, H, W], so we must replace
    # the buffer instead of using in-place copy_ (which requires identical shapes).
    if isinstance(getattr(decoder, "_buffers", None), dict) and "dense_embedding" in decoder._buffers:
        decoder._buffers["dense_embedding"] = resized
    else:
        setattr(decoder, "dense_embedding", resized)

    setattr(decoder, "embed_h", image_h)
    setattr(decoder, "embed_w", image_w)


def main() -> None:
    start_time = time.perf_counter()
    log_progress("Starting EdgeSAM decoder PTQ4SAM pipeline")
    args = parse_args()
    log_progress(
        "Args loaded: "
        f"scope={args.scope}, bit={args.bit}, calibration_count={args.calibration_count}, eval_count={args.eval_count}"
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because PTQ4SAM BIG/AGQ calibration code uses CUDA tensors")
    log_progress("CUDA available")

    checkpoint_path = Path(args.checkpoint).resolve()
    log_progress(f"Collecting calibration samples from {args.calibration_list}")
    calibration_paths = collect_decoder_sample_triplets(args.calibration_list, limit=args.calibration_count)
    log_progress(f"Collected calibration samples: {len(calibration_paths)}")
    log_progress(f"Collecting eval samples from {args.eval_list}")
    eval_paths = collect_decoder_sample_triplets(args.eval_list, limit=args.eval_count)
    log_progress(f"Collected eval samples: {len(eval_paths)}")

    if len(calibration_paths) < args.calibration_count:
        raise ValueError(f"Requested {args.calibration_count} calibration samples but found {len(calibration_paths)}")
    if len(eval_paths) < args.eval_count:
        raise ValueError(f"Requested {args.eval_count} eval samples but found {len(eval_paths)}")

    config_quant = make_qconfig(bit=args.bit)
    config_quant.ptq4sam.BIG = not args.disable_big
    config_quant.ptq4sam.AGQ = not args.disable_agq

    device = torch.device("cuda:0")
    log_progress(f"Using device: {device}")

    log_progress("Building FP model")
    fp_model = _build_decoder_surface(checkpoint_path, use_stability_score=True).to(device).eval()
    log_progress("Building quant model")
    quant_model = _build_decoder_surface(checkpoint_path, use_stability_score=True).to(device).eval()
    log_progress("Applying quantization wrappers")
    quant_model = quantize_decoder_surface(quant_model, config_quant, scope=args.scope).to(device).eval()

    log_progress("Loading calibration tensors to device")
    calibration_samples = move_samples_to_device(calibration_paths, device)
    log_progress("Running calibration")
    calibrate_decoder(quant_model, calibration_samples, config_quant.ptq4sam.BIG)
    log_progress("Enabling quantization")
    enable_quantization(quant_model)

    log_progress("Running decoder evaluation")
    bundle = evaluate_decoder(fp_model, quant_model, eval_paths)
    log_progress("Evaluation complete")

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
    log_progress(f"Summary written to {summary_path}")

    onnx_path = resolve_output_path(str(checkpoint_path), args.onnx_output, args.scope)
    log_progress(f"Preparing ONNX export to {onnx_path}")
    dummy_inputs = build_dummy_inputs(args)
    _, _, image_h, image_w = args.image_embeddings_shape
    log_progress(f"Adapting decoder dense_embedding to export size HxW={image_h}x{image_w}")
    adapt_decoder_dense_embedding_size(quant_model, image_h=image_h, image_w=image_w)
    log_progress("Exporting quantized decoder to ONNX")
    export_quantized_decoder_to_onnx(
        quant_model,
        output_path=onnx_path,
        num_points=args.point_embedding_pe_shape[1],
        opset_version=args.opset_version,
        dummy_inputs=dummy_inputs,
    )
    pe_path = export_pe_gaussian_matrix(str(checkpoint_path), onnx_path)
    print(f"Exported quantized decoder ONNX to {onnx_path}")
    print(f"Exported PE gaussian matrix to {pe_path}")
    log_progress("Printing ONNX operator summary")
    _print_onnx_summary(onnx_path)

    if args.check_ops_only and onnx_path.exists():
        onnx_path.unlink()
        print(f"Removed {onnx_path} (check-ops-only mode)")
        log_progress("ONNX removed due to --check-ops-only")

    elapsed = time.perf_counter() - start_time
    log_progress(f"Pipeline finished in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
