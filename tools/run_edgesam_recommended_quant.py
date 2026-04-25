#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RECOMMENDED_SCOPE = "recommended_mixed_maskhead_w8a16pc_signed"


def ensure_wandb_stub() -> None:
    if "wandb" not in sys.modules:
        sys.modules["wandb"] = types.ModuleType("wandb")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def recommended_metadata() -> dict[str, Any]:
    return {
        "scope": RECOMMENDED_SCOPE,
        "transformer_bit": 8,
        "transformer_big": True,
        "transformer_agq": True,
        "mask_head_weight_bit": 8,
        "mask_head_weight_signed": False,
        "mask_head_activation_bit": 16,
        "mask_head_activation_signed": True,
        "mask_head_activation_granularity": "per_channel",
        "feature_or_token_ch_axis": 1,
        "hyper_in_ch_axis": 2,
        "tail_observer": "MinMaxObserver",
        "tail_quantizer": "FixedFakeQuantize",
        "mask_head_modules": [
            "output_upscaling",
            "output_hypernetworks",
            "mask_projection",
            "score_head",
        ],
        "deployment_note": (
            "This reproduces the recommended PyTorch fake-quant policy. Vanilla onecc can not "
            "represent the per-channel INT16 activation part without explicit Q/DQ preservation or "
            "ONE quantizer changes."
        ),
    }


def make_recommended_qconfigs() -> dict[str, Any]:
    ensure_wandb_stub()
    from scripts.edgesam_decoder_ptq4sam_uint8 import AttrDict, clone_qconfig, make_qconfig

    transformer_config = make_qconfig(bit=8)
    transformer_config.ptq4sam.BIG = True
    transformer_config.ptq4sam.AGQ = True

    mask_weight_qconfig = clone_qconfig(
        transformer_config.w_qconfig,
        bit=8,
        symmetric=False,
        ch_axis=0,
    )
    feature_activation_qconfig = AttrDict(
        quantizer="FixedFakeQuantize",
        observer="MinMaxObserver",
        bit=16,
        symmetric=True,
        ch_axis=1,
    )
    hyper_activation_qconfig = clone_qconfig(feature_activation_qconfig, ch_axis=2)

    return {
        "transformer": transformer_config,
        "mask_weight": mask_weight_qconfig,
        "mask_feature_activation": feature_activation_qconfig,
        "mask_hyper_activation": hyper_activation_qconfig,
    }


def build_recommended_decoder_surface(org_module: Any) -> Any:
    ensure_wandb_stub()
    import torch
    import torch.nn as nn

    from ptq4sam.quantization.quantized_module import QuantizedMatMul, WeightQuantizer
    from scripts.edgesam_decoder_ptq4sam_uint8 import (
        QuantDecoderHypernetworkStack,
        QuantDecoderLabelEmbedding,
        QuantDecoderScoreHead,
        QuantEdgeTwoWayTransformer,
        QuantOutputUpscaling,
    )

    qconfigs = make_recommended_qconfigs()
    transformer_config = qconfigs["transformer"]
    mask_weight_qconfig = qconfigs["mask_weight"]
    feature_activation_qconfig = qconfigs["mask_feature_activation"]
    hyper_activation_qconfig = qconfigs["mask_hyper_activation"]

    class RecommendedMaskProjection(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.matmul = QuantizedMatMul(
                feature_activation_qconfig,
                quantize_a_input=True,
                quantize_b_input=True,
                a_input_qconfig=hyper_activation_qconfig,
                b_input_qconfig=feature_activation_qconfig,
            )

        def forward(self, hyper_in: torch.Tensor, upscaled: torch.Tensor) -> torch.Tensor:
            batch_size = upscaled.size(0)
            channels = upscaled.size(1)
            spatial = upscaled.size(2) * upscaled.size(3)
            flat = upscaled.reshape(batch_size, channels, spatial)
            return self.matmul((hyper_in, flat))

    class RecommendedMixedDecoderSurface(nn.Module):
        def __init__(self, source: Any) -> None:
            super().__init__()
            self.label_embedding = QuantDecoderLabelEmbedding(
                source.label_embedding,
                transformer_config.w_qconfig,
                transformer_config.a_qconfig,
            )
            self.transformer = QuantEdgeTwoWayTransformer(
                source.transformer,
                transformer_config.w_qconfig,
                transformer_config.a_qconfig,
                transformer_config.ptq4sam,
            )
            self.iou_token = source.iou_token
            self.mask_tokens = source.mask_tokens
            self.iou_token_weight_fake_quantize = WeightQuantizer(copy.deepcopy(transformer_config.w_qconfig))
            self.mask_tokens_weight_fake_quantize = WeightQuantizer(copy.deepcopy(transformer_config.w_qconfig))
            self.register_buffer("dense_embedding", source.dense_embedding.clone())
            self.register_buffer("image_pe", source.image_pe.clone())
            self.output_upscaling = QuantOutputUpscaling(
                source.output_upscaling,
                mask_weight_qconfig,
                feature_activation_qconfig,
            )
            self.output_hypernetworks = QuantDecoderHypernetworkStack(
                source.output_hypernetworks,
                mask_weight_qconfig,
                feature_activation_qconfig,
            )
            self.mask_projection = RecommendedMaskProjection()
            self.score_head = QuantDecoderScoreHead(
                source.score_head,
                mask_weight_qconfig,
                feature_activation_qconfig,
            )
            self.num_mask_tokens = source.num_mask_tokens
            self.embed_h = source.embed_h
            self.embed_w = source.embed_w

        def forward(
            self,
            image_embeddings: torch.Tensor,
            point_embedding_pe: torch.Tensor,
            point_labels: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            sparse_embedding = self.label_embedding(point_embedding_pe, point_labels)
            iou_token = self.iou_token_weight_fake_quantize(self.iou_token.weight)
            mask_tokens = self.mask_tokens_weight_fake_quantize(self.mask_tokens.weight)
            output_tokens = torch.cat([iou_token, mask_tokens], dim=0)
            output_tokens = output_tokens.unsqueeze(0).expand(sparse_embedding.size(0), -1, -1)
            tokens = torch.cat((output_tokens, sparse_embedding), dim=1)

            src = image_embeddings + self.dense_embedding
            hs, src = self.transformer(src, self.image_pe, tokens)

            iou_token_out = hs[:, 0, :]
            mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

            batch_size = src.size(0)
            embed_dim = src.size(2)
            src = src.transpose(1, 2).reshape(batch_size, embed_dim, self.embed_h, self.embed_w)

            upscaled = self.output_upscaling(src)
            hyper_in = self.output_hypernetworks(mask_tokens_out)
            masks = self.mask_projection(hyper_in, upscaled).reshape(
                batch_size,
                -1,
                upscaled.size(2),
                upscaled.size(3),
            )
            scores = self.score_head(iou_token_out, masks)
            return scores, masks

    return RecommendedMixedDecoderSurface(org_module)


def build_calibrated_pair(args: argparse.Namespace, device: Any) -> tuple[Any, Any]:
    ensure_wandb_stub()
    import torch

    from edge_sam.quantization.decoder_ptq_compare import _build_decoder_surface, collect_decoder_sample_triplets
    from ptq4sam.quantization.state import enable_quantization
    from scripts.edgesam_decoder_ptq4sam_uint8 import calibrate_decoder, move_samples_to_device

    fp32_decoder = _build_decoder_surface(
        args.checkpoint,
        use_stability_score=args.use_stability_score,
    ).to(device).eval()
    quant_decoder = _build_decoder_surface(
        args.checkpoint,
        use_stability_score=args.use_stability_score,
    )
    quant_decoder = build_recommended_decoder_surface(quant_decoder).to(device).eval()

    calibration_paths = collect_decoder_sample_triplets(args.calibration_list, limit=args.calibration_count)
    if len(calibration_paths) < args.calibration_count:
        raise ValueError(
            f"Requested {args.calibration_count} calibration samples but found {len(calibration_paths)}"
        )
    calibration_samples = move_samples_to_device(calibration_paths, torch.device(device))
    calibrate_decoder(quant_decoder, calibration_samples, big=True)
    enable_quantization(quant_decoder)
    return fp32_decoder, quant_decoder


def run_tensor_metrics(args: argparse.Namespace, output_dir: Path) -> Path:
    ensure_wandb_stub()
    import torch

    from edge_sam.quantization.decoder_ptq_compare import collect_decoder_sample_triplets
    from scripts.edgesam_decoder_ptq4sam_uint8 import evaluate_decoder

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for the default PTQ calibration path")

    device = torch.device(args.device)
    fp32_decoder, quant_decoder = build_calibrated_pair(args, device)
    eval_paths = collect_decoder_sample_triplets(args.eval_list, limit=args.eval_count)
    if len(eval_paths) < args.eval_count:
        raise ValueError(f"Requested {args.eval_count} eval samples but found {len(eval_paths)}")

    bundle = evaluate_decoder(fp32_decoder, quant_decoder, eval_paths)
    payload = {
        "config": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "calibration_list": str(Path(args.calibration_list).resolve()),
            "eval_list": str(Path(args.eval_list).resolve()),
            "calibration_count": args.calibration_count,
            "eval_count": args.eval_count,
            "use_stability_score": args.use_stability_score,
            **recommended_metadata(),
        },
        "mean_metrics": bundle.mean_metrics,
        "per_sample": bundle.per_sample,
    }
    output_path = output_dir / "tensor_metrics_summary.json"
    write_json(output_path, payload)
    return output_path


def build_models_for_real_iou(args: argparse.Namespace, device: str) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    ensure_wandb_stub()
    from edge_sam import SamPredictor, sam_model_registry

    sam = sam_model_registry["edge_sam"](checkpoint=args.checkpoint, upsample_mode="bilinear")
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    fp32_decoder, quant_decoder = build_calibrated_pair(args, device)
    return sam, predictor, fp32_decoder, quant_decoder, recommended_metadata()


def run_real_iou(args: argparse.Namespace, output_dir: Path) -> Path:
    ensure_wandb_stub()
    import numpy as np
    import torch

    import scripts.eval_edgesam_decoder_real_iou as real_iou

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for the default PTQ calibration path")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    original_build_models = real_iou.build_models
    real_iou.build_models = build_models_for_real_iou
    try:
        payload = real_iou.run_evaluation(args)
    finally:
        real_iou.build_models = original_build_models

    payload["summary"].update(recommended_metadata())
    output_path = output_dir / "real_iou_summary.json"
    write_json(output_path, payload)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the recommended EdgeSAM mixed quantization policy and write tensor and real IoU metrics."
    )
    parser.add_argument("--stage", choices=("tensor", "real-iou", "both"), default="both")
    parser.add_argument("--checkpoint", default="/home/kitemanul/project/EdgeSAM/weights/edge_sam.pth")
    parser.add_argument(
        "--calibration-list",
        default="/home/kitemanul/project/EdgeSAM/weights/one_pipeline/calib/decoder_datalist_20.txt",
    )
    parser.add_argument("--calibration-count", type=int, default=20)
    parser.add_argument("--eval-list", default="/home/kitemanul/project/EdgeSAM/weights/one_pipeline/calib/decoder_test_5.txt")
    parser.add_argument("--eval-count", type=int, default=5)
    parser.add_argument("--ann-file", default="/home/kitemanul/dataset/coco2017/annotations/instances_val2017.json")
    parser.add_argument("--img-dir", default="/home/kitemanul/dataset/coco2017/val2017")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--max-masks-per-image", type=int, default=-1)
    parser.add_argument("--num-points", type=int, default=1)
    parser.add_argument("--point-strategy", choices=("random", "center", "bbox_center"), default="center")
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--fp32-mask-threshold", type=float, default=None)
    parser.add_argument("--uint8-mask-threshold", type=float, default=None)
    parser.add_argument("--use-stability-score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-per-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to results/edgesam_recommended_mixed_quant/<timestamp>",
    )
    args = parser.parse_args()

    args.bit = 8
    args.scope = RECOMMENDED_SCOPE
    args.disable_big = False
    args.disable_agq = False
    args.quantize_hypernetwork_output = True
    args.quantize_mask_projection_hyper_input = True
    args.quantize_mask_projection_upscaled_input = True
    args.hypernetwork_output_bit = None
    args.mask_projection_hyper_input_bit = None
    args.mask_projection_upscaled_input_bit = None
    args.output = None
    return args


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        output_dir = Path("results") / "edgesam_recommended_mixed_quant" / datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    if args.stage in ("tensor", "both"):
        outputs.append(run_tensor_metrics(args, output_dir))
    if args.stage in ("real-iou", "both"):
        outputs.append(run_real_iou(args, output_dir))

    print(json.dumps({"outputs": [str(path) for path in outputs], **recommended_metadata()}, indent=2))


if __name__ == "__main__":
    main()
