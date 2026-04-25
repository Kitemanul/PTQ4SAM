from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ptq4sam.selection_metrics import summarize_selection_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate EdgeSAM decoder-only PTQ on COCO-format data and report both "
            "model-selected and oracle-selected mIoU."
        )
    )
    parser.add_argument('--checkpoint', default='/home/kitemanul/project/EdgeSAM/weights/edge_sam.pth')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--img-dir', required=True)
    parser.add_argument(
        '--calibration-list',
        default='/home/kitemanul/project/EdgeSAM/weights/one_pipeline/calib/decoder_datalist_20.txt',
    )
    parser.add_argument('--calibration-count', type=int, default=20)
    parser.add_argument('--bit', type=int, default=8)
    parser.add_argument('--scope', choices=('transformer', 'full'), default='full')
    parser.add_argument('--disable-big', action='store_true')
    parser.add_argument('--disable-agq', action='store_true')
    parser.add_argument('--use-stability-score', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--quantize-hypernetwork-output', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--quantize-mask-projection-hyper-input', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--quantize-mask-projection-upscaled-input', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--hypernetwork-output-bit', type=int, default=None)
    parser.add_argument('--mask-projection-hyper-input-bit', type=int, default=None)
    parser.add_argument('--mask-projection-upscaled-input-bit', type=int, default=None)
    parser.add_argument('--mask-threshold', type=float, default=None)
    parser.add_argument('--fp32-mask-threshold', type=float, default=None)
    parser.add_argument('--uint8-mask-threshold', type=float, default=None)
    parser.add_argument('--num-points', type=int, default=1)
    parser.add_argument('--point-strategy', choices=('random', 'center', 'bbox_center'), default='center')
    parser.add_argument('--max-masks-per-image', type=int, default=-1)
    parser.add_argument('--max-images', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None, help='Optional summary.json output path')
    parser.add_argument('--save-per-mask', action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def decode_mask(segmentation: Any, height: int, width: int) -> np.ndarray | None:
    from pycocotools import mask as mask_utils

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict):
        if isinstance(segmentation.get('counts'), list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
    else:
        return None
    return mask_utils.decode(rle)


def sample_points(
    binary_mask: np.ndarray,
    num_points: int,
    strategy: str,
    *,
    rng: np.random.Generator,
    bbox: Iterable[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        coords = np.zeros((num_points, 2), dtype=np.float32)
        labels = np.ones(num_points, dtype=np.float32)
        return coords, labels

    if strategy == 'bbox_center' and bbox is not None:
        x, y, w, h = [float(v) for v in bbox]
        center = np.array([[x + w / 2.0, y + h / 2.0]], dtype=np.float32)
        coords = np.repeat(center, num_points, axis=0)
    elif strategy == 'center':
        center = np.array([[float(xs.mean()), float(ys.mean())]], dtype=np.float32)
        coords = np.repeat(center, num_points, axis=0)
    else:
        replace = len(xs) < num_points
        indices = rng.choice(len(xs), size=num_points, replace=replace)
        coords = np.stack([xs[indices], ys[indices]], axis=1).astype(np.float32)

    labels = np.ones(num_points, dtype=np.float32)
    return coords, labels


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def summarize_candidate_masks(
    candidate_masks: np.ndarray,
    candidate_scores: np.ndarray,
    gt_mask: np.ndarray,
) -> dict[str, Any]:
    candidate_ious = [compute_iou(mask, gt_mask) for mask in candidate_masks]
    summary = summarize_selection_sample(candidate_ious, candidate_scores.tolist())
    return {
        'num_candidates': int(summary['num_candidates']),
        'selected_index': int(summary['selected_index']),
        'oracle_index': int(summary['oracle_index']),
        'selected_iou': float(summary['selected_miou']),
        'oracle_iou': float(summary['oracle_miou']),
        'oracle_gap': float(summary['oracle_gap']),
        'rank_hit_at_1': float(summary['rank_hit_at_1']),
        'candidate_ious': [float(value) for value in candidate_ious],
        'candidate_scores': [float(value) for value in candidate_scores.tolist()],
    }


def aggregate_prompt_metrics(records: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    if not records:
        raise ValueError('records must not be empty')
    selected = np.array([record[f'{prefix}_selected_iou'] for record in records], dtype=np.float64)
    oracle = np.array([record[f'{prefix}_oracle_iou'] for record in records], dtype=np.float64)
    gap = np.array([record[f'{prefix}_oracle_gap'] for record in records], dtype=np.float64)
    hit = np.array([record[f'{prefix}_rank_hit_at_1'] for record in records], dtype=np.float64)
    return {
        f'{prefix}_selected_miou': float(selected.mean()),
        f'{prefix}_oracle_miou': float(oracle.mean()),
        f'{prefix}_oracle_gap': float(gap.mean()),
        f'{prefix}_rank_hit_at_1': float(hit.mean()),
        f'{prefix}_selected_median_iou': float(np.median(selected)),
        f'{prefix}_selected_iou_std': float(selected.std()),
        f'{prefix}_oracle_median_iou': float(np.median(oracle)),
        f'{prefix}_oracle_iou_std': float(oracle.std()),
    }


def summarize_image_records(image_id: int, file_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        'image_id': image_id,
        'file_name': file_name,
        'num_masks': len(records),
    }
    for prefix in ('fp32', 'uint8'):
        selected = np.array([record[f'{prefix}_selected_iou'] for record in records], dtype=np.float64)
        oracle = np.array([record[f'{prefix}_oracle_iou'] for record in records], dtype=np.float64)
        gap = np.array([record[f'{prefix}_oracle_gap'] for record in records], dtype=np.float64)
        hit = np.array([record[f'{prefix}_rank_hit_at_1'] for record in records], dtype=np.float64)
        result[f'{prefix}_mean_iou'] = float(selected.mean())
        result[f'{prefix}_oracle_mean_iou'] = float(oracle.mean())
        result[f'{prefix}_oracle_gap'] = float(gap.mean())
        result[f'{prefix}_rank_hit_at_1'] = float(hit.mean())
    result['delta_mean_iou'] = float(result['uint8_mean_iou'] - result['fp32_mean_iou'])
    result['delta_oracle_mean_iou'] = float(result['uint8_oracle_mean_iou'] - result['fp32_oracle_mean_iou'])
    return result


def build_summary_payload(
    args: argparse.Namespace,
    per_mask_records: list[dict[str, Any]],
    per_image_records: list[dict[str, Any]],
) -> dict[str, Any]:
    fp32_metrics = aggregate_prompt_metrics(per_mask_records, 'fp32')
    uint8_metrics = aggregate_prompt_metrics(per_mask_records, 'uint8')
    summary: dict[str, Any] = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'ann_file': str(Path(args.ann_file).resolve()),
        'img_dir': str(Path(args.img_dir).resolve()),
        'images_requested': args.max_images,
        'images_processed': len(per_image_records),
        'masks_evaluated': len(per_mask_records),
        'calibration_list': str(Path(args.calibration_list).resolve()),
        'bit': args.bit,
        'scope': args.scope,
        'big': not args.disable_big,
        'agq': not args.disable_agq,
        'use_stability_score': bool(args.use_stability_score),
        'quantize_hypernetwork_output': bool(args.quantize_hypernetwork_output),
        'quantize_mask_projection_hyper_input': bool(args.quantize_mask_projection_hyper_input),
        'quantize_mask_projection_upscaled_input': bool(args.quantize_mask_projection_upscaled_input),
        'hypernetwork_output_bit': args.hypernetwork_output_bit,
        'mask_projection_hyper_input_bit': args.mask_projection_hyper_input_bit,
        'mask_projection_upscaled_input_bit': args.mask_projection_upscaled_input_bit,
        'mask_threshold': args.mask_threshold,
        'fp32_mask_threshold': args.fp32_mask_threshold,
        'uint8_mask_threshold': args.uint8_mask_threshold,
        'point_strategy': args.point_strategy,
        'num_points': args.num_points,
        **fp32_metrics,
        **uint8_metrics,
    }
    summary['delta_selected_miou'] = float(summary['uint8_selected_miou'] - summary['fp32_selected_miou'])
    summary['delta_oracle_miou'] = float(summary['uint8_oracle_miou'] - summary['fp32_oracle_miou'])

    # Backward-compatible aliases for existing selected-mask summary consumers.
    summary['fp32_miou'] = summary['fp32_selected_miou']
    summary['uint8_miou'] = summary['uint8_selected_miou']
    summary['delta_miou'] = summary['delta_selected_miou']
    summary['fp32_median_iou'] = summary['fp32_selected_median_iou']
    summary['uint8_median_iou'] = summary['uint8_selected_median_iou']
    summary['fp32_iou_std'] = summary['fp32_selected_iou_std']
    summary['uint8_iou_std'] = summary['uint8_selected_iou_std']

    payload = {
        'summary': summary,
        'per_image': per_image_records,
    }
    if args.save_per_mask:
        payload['per_mask'] = per_mask_records
    return payload


def load_coco_annotations(ann_file: str, max_images: int) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    with open(ann_file, 'r', encoding='utf-8') as handle:
        coco = json.load(handle)

    images = {image['id']: image for image in coco['images']}
    ann_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco['annotations']:
        if annotation.get('iscrowd', 0):
            continue
        image_id = annotation['image_id']
        if image_id in images:
            ann_by_image[image_id].append(annotation)

    image_ids = [image_id for image_id in ann_by_image if image_id in images]
    if max_images > 0:
        image_ids = image_ids[:max_images]
    return [(images[image_id], ann_by_image[image_id]) for image_id in image_ids]


def build_models(args: argparse.Namespace, device: str):
    import sys
    import types

    import torch

    if 'wandb' not in sys.modules:
        sys.modules['wandb'] = types.ModuleType('wandb')

    from edge_sam import SamPredictor, sam_model_registry
    from edge_sam.quantization.decoder_ptq_compare import _build_decoder_surface, collect_decoder_sample_triplets
    from ptq4sam.quantization.state import enable_quantization
    from scripts.edgesam_decoder_ptq4sam_uint8 import calibrate_decoder, make_qconfig, move_samples_to_device, quantize_decoder_surface

    sam = sam_model_registry['edge_sam'](checkpoint=args.checkpoint, upsample_mode='bilinear')
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    fp32_decoder = _build_decoder_surface(args.checkpoint, use_stability_score=args.use_stability_score).to(device).eval()
    quant_decoder = _build_decoder_surface(args.checkpoint, use_stability_score=args.use_stability_score).to(device).eval()

    config_quant = make_qconfig(bit=args.bit)
    config_quant.ptq4sam.BIG = not args.disable_big
    config_quant.ptq4sam.AGQ = not args.disable_agq
    quant_decoder = quantize_decoder_surface(
        quant_decoder,
        config_quant,
        scope=args.scope,
        quantize_hypernetwork_output=args.quantize_hypernetwork_output,
        quantize_mask_projection_hyper_input=args.quantize_mask_projection_hyper_input,
        quantize_mask_projection_upscaled_input=args.quantize_mask_projection_upscaled_input,
        hypernetwork_output_bit=args.hypernetwork_output_bit,
        mask_projection_hyper_input_bit=args.mask_projection_hyper_input_bit,
        mask_projection_upscaled_input_bit=args.mask_projection_upscaled_input_bit,
    ).to(device).eval()

    calibration_paths = collect_decoder_sample_triplets(args.calibration_list, limit=args.calibration_count)
    if len(calibration_paths) < args.calibration_count:
        raise ValueError(
            f'Requested {args.calibration_count} calibration samples but found {len(calibration_paths)}'
        )
    calibration_samples = move_samples_to_device(calibration_paths, torch.device(device))
    calibrate_decoder(quant_decoder, calibration_samples, config_quant.ptq4sam.BIG)
    enable_quantization(quant_decoder)
    return sam, predictor, fp32_decoder, quant_decoder, config_quant


def compute_point_pe(sam, predictor, point_coords: np.ndarray, original_size: tuple[int, int]):
    import torch

    transformed = predictor.transform.apply_coords(point_coords, original_size)
    coords = torch.as_tensor(transformed, dtype=torch.float32, device=sam.device)[None, :, :]
    labels = torch.ones((1, point_coords.shape[0]), dtype=torch.float32, device=sam.device)
    with torch.no_grad():
        point_embedding_pe = sam.prompt_encoder.pe_layer._pe_encoding((coords + 0.5) / sam.image_encoder.img_size)
    return point_embedding_pe, labels


def resolve_mask_thresholds(args: argparse.Namespace, default_threshold: float) -> tuple[float, float]:
    shared_threshold = default_threshold if args.mask_threshold is None else float(args.mask_threshold)
    fp32_threshold = shared_threshold if args.fp32_mask_threshold is None else float(args.fp32_mask_threshold)
    uint8_threshold = shared_threshold if args.uint8_mask_threshold is None else float(args.uint8_mask_threshold)
    return fp32_threshold, uint8_threshold


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for decoder PTQ evaluation')

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda:0'
    sam, predictor, fp32_decoder, quant_decoder, _ = build_models(args, device=device)
    args.fp32_mask_threshold, args.uint8_mask_threshold = resolve_mask_thresholds(args, float(sam.mask_threshold))
    image_records = load_coco_annotations(args.ann_file, args.max_images)

    per_mask_records: list[dict[str, Any]] = []
    per_image_records: list[dict[str, Any]] = []

    for image_info, annotations in image_records:
        image_path = os.path.join(args.img_dir, image_info['file_name'])
        if not os.path.exists(image_path):
            continue

        image = np.array(Image.open(image_path).convert('RGB'))
        features = predictor.set_image(image)
        original_size = tuple(int(v) for v in predictor.original_size)
        input_size = tuple(int(v) for v in predictor.input_size)
        image_level_records: list[dict[str, Any]] = []

        if args.max_masks_per_image > 0 and len(annotations) > args.max_masks_per_image:
            selected_indices = rng.choice(len(annotations), size=args.max_masks_per_image, replace=False)
            annotations = [annotations[int(index)] for index in selected_indices]

        for ann_index, annotation in enumerate(annotations):
            gt_mask = decode_mask(annotation['segmentation'], image_info['height'], image_info['width'])
            if gt_mask is None or gt_mask.sum() == 0:
                continue

            point_coords, point_labels = sample_points(
                gt_mask,
                args.num_points,
                args.point_strategy,
                rng=rng,
                bbox=annotation.get('bbox'),
            )
            point_embedding_pe, label_tensor = compute_point_pe(sam, predictor, point_coords, original_size)

            with torch.no_grad():
                fp32_scores, fp32_masks = fp32_decoder(features, point_embedding_pe, label_tensor)
                uint8_scores, uint8_masks = quant_decoder(features, point_embedding_pe, label_tensor)
                fp32_masks = sam.postprocess_masks(fp32_masks, input_size, original_size)
                uint8_masks = sam.postprocess_masks(uint8_masks, input_size, original_size)

            fp32_binary = (fp32_masks[0] > args.fp32_mask_threshold).detach().cpu().numpy()
            uint8_binary = (uint8_masks[0] > args.uint8_mask_threshold).detach().cpu().numpy()
            fp32_summary = summarize_candidate_masks(
                fp32_binary,
                fp32_scores[0].detach().cpu().numpy(),
                gt_mask,
            )
            uint8_summary = summarize_candidate_masks(
                uint8_binary,
                uint8_scores[0].detach().cpu().numpy(),
                gt_mask,
            )

            record: dict[str, Any] = {
                'image_id': int(image_info['id']),
                'file_name': image_info['file_name'],
                'ann_index': ann_index,
                'point_coords': point_coords.tolist(),
                'point_labels': point_labels.tolist(),
            }
            for prefix, summary in (('fp32', fp32_summary), ('uint8', uint8_summary)):
                for key, value in summary.items():
                    record[f'{prefix}_{key}'] = value
            image_level_records.append(record)
            per_mask_records.append(record)

        if image_level_records:
            per_image_records.append(
                summarize_image_records(int(image_info['id']), image_info['file_name'], image_level_records)
            )

    return build_summary_payload(args, per_mask_records, per_image_records)


def main() -> None:
    args = parse_args()
    payload = run_evaluation(args)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        print(f'Saved summary to {output_path}')
    print(json.dumps(payload['summary'], indent=2))


if __name__ == '__main__':
    main()
