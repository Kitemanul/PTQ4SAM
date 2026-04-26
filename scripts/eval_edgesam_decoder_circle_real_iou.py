from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval_edgesam_decoder_real_iou import (
    aggregate_prompt_metrics,
    decode_mask,
    load_coco_annotations,
    sample_points,
    summarize_candidate_masks,
)


DEFAULT_ONE_QUANTIZE = '/home/kitemanul/project/ONE/build/py312/compiler/one-cmds/one-quantize'
DEFAULT_CIRCLE_INTERPRETER_LIB = (
    '/home/kitemanul/project/ONE/build/compiler/circle-interpreter/libcircle_interpreter_cffi.so'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate an EdgeSAM NPU decoder Circle model on COCO-format masks. '
            'Quantized Circle models are evaluated through a fake-quantized FP32 '
            'Circle because the local luci interpreter does not execute all uint8 kernels.'
        )
    )
    parser.add_argument('--checkpoint', default='/home/kitemanul/project/EdgeSAM/weights/edge_sam.pth')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--circle-model', default=None, help='Existing fake-quant or FP32 Circle model to evaluate')
    parser.add_argument('--quant-circle-model', default=None, help='Quantized Circle model to convert with --fake_quantize')
    parser.add_argument('--fake-quant-output', default=None, help='Output path for materialized fake-quant Circle')
    parser.add_argument('--reuse-fake-quant', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--one-quantize', default=DEFAULT_ONE_QUANTIZE)
    parser.add_argument('--circle-interpreter-lib', default=DEFAULT_CIRCLE_INTERPRETER_LIB)
    parser.add_argument('--use-stability-score', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--mask-threshold', type=float, default=None)
    parser.add_argument('--fp32-mask-threshold', type=float, default=None)
    parser.add_argument('--circle-mask-threshold', type=float, default=None)
    parser.add_argument('--num-points', type=int, default=1)
    parser.add_argument('--circle-num-points', type=int, default=5)
    parser.add_argument('--point-strategy', choices=('random', 'center', 'bbox_center'), default='center')
    parser.add_argument('--max-masks-per-image', type=int, default=-1)
    parser.add_argument('--max-images', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--save-per-mask', action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def pad_point_prompt(
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    *,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(point_coords, dtype=np.float32)
    labels = np.asarray(point_labels, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f'point_coords must have shape [N, 2], got {coords.shape}')
    if labels.ndim != 1 or labels.shape[0] != coords.shape[0]:
        raise ValueError(f'point_labels must have shape [N], got {labels.shape} for coords {coords.shape}')
    if coords.shape[0] > num_points:
        raise ValueError(f'{coords.shape[0]} prompt points exceed static Circle prompt count {num_points}')

    padded_coords = np.zeros((num_points, 2), dtype=np.float32)
    padded_labels = np.full((num_points,), -1.0, dtype=np.float32)
    padded_coords[: coords.shape[0]] = coords
    padded_labels[: labels.shape[0]] = labels
    return padded_coords, padded_labels


def resolve_mask_thresholds(args: argparse.Namespace, default_threshold: float) -> tuple[float, float]:
    shared_threshold = default_threshold if args.mask_threshold is None else float(args.mask_threshold)
    fp32_threshold = shared_threshold if args.fp32_mask_threshold is None else float(args.fp32_mask_threshold)
    circle_threshold = shared_threshold if args.circle_mask_threshold is None else float(args.circle_mask_threshold)
    return fp32_threshold, circle_threshold


def summarize_image_records(image_id: int, file_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        'image_id': image_id,
        'file_name': file_name,
        'num_masks': len(records),
    }
    for prefix in ('fp32', 'circle'):
        selected = np.array([record[f'{prefix}_selected_iou'] for record in records], dtype=np.float64)
        oracle = np.array([record[f'{prefix}_oracle_iou'] for record in records], dtype=np.float64)
        gap = np.array([record[f'{prefix}_oracle_gap'] for record in records], dtype=np.float64)
        hit = np.array([record[f'{prefix}_rank_hit_at_1'] for record in records], dtype=np.float64)
        result[f'{prefix}_mean_iou'] = float(selected.mean())
        result[f'{prefix}_oracle_mean_iou'] = float(oracle.mean())
        result[f'{prefix}_oracle_gap'] = float(gap.mean())
        result[f'{prefix}_rank_hit_at_1'] = float(hit.mean())
    result['delta_mean_iou'] = float(result['circle_mean_iou'] - result['fp32_mean_iou'])
    result['delta_oracle_mean_iou'] = float(result['circle_oracle_mean_iou'] - result['fp32_oracle_mean_iou'])
    return result


def build_summary_payload(
    args: argparse.Namespace,
    per_mask_records: list[dict[str, Any]],
    per_image_records: list[dict[str, Any]],
) -> dict[str, Any]:
    fp32_metrics = aggregate_prompt_metrics(per_mask_records, 'fp32')
    circle_metrics = aggregate_prompt_metrics(per_mask_records, 'circle')
    summary: dict[str, Any] = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'ann_file': str(Path(args.ann_file).resolve()),
        'img_dir': str(Path(args.img_dir).resolve()),
        'circle_model': str(Path(args.circle_model).resolve()) if args.circle_model else None,
        'quant_circle_model': str(Path(args.quant_circle_model).resolve()) if args.quant_circle_model else None,
        'circle_interpreter_lib': str(Path(args.circle_interpreter_lib).resolve()),
        'images_requested': args.max_images,
        'images_processed': len(per_image_records),
        'masks_evaluated': len(per_mask_records),
        'use_stability_score': bool(args.use_stability_score),
        'mask_threshold': args.mask_threshold,
        'fp32_mask_threshold': args.fp32_mask_threshold,
        'circle_mask_threshold': args.circle_mask_threshold,
        'point_strategy': args.point_strategy,
        'num_points': args.num_points,
        'circle_num_points': args.circle_num_points,
        **fp32_metrics,
        **circle_metrics,
    }
    summary['delta_selected_miou'] = float(summary['circle_selected_miou'] - summary['fp32_selected_miou'])
    summary['delta_oracle_miou'] = float(summary['circle_oracle_miou'] - summary['fp32_oracle_miou'])

    summary['fp32_miou'] = summary['fp32_selected_miou']
    summary['circle_miou'] = summary['circle_selected_miou']
    summary['delta_miou'] = summary['delta_selected_miou']
    summary['fp32_median_iou'] = summary['fp32_selected_median_iou']
    summary['circle_median_iou'] = summary['circle_selected_median_iou']
    summary['fp32_iou_std'] = summary['fp32_selected_iou_std']
    summary['circle_iou_std'] = summary['circle_selected_iou_std']

    payload = {
        'summary': summary,
        'per_image': per_image_records,
    }
    if args.save_per_mask:
        payload['per_mask'] = per_mask_records
    return payload


def materialize_circle_model(args: argparse.Namespace) -> str:
    if args.circle_model:
        return args.circle_model
    if not args.quant_circle_model:
        raise ValueError('Either --circle-model or --quant-circle-model is required')

    quant_model = Path(args.quant_circle_model)
    if args.fake_quant_output:
        fake_quant_model = Path(args.fake_quant_output)
    else:
        fake_quant_model = (
            Path('results/edgesam_decoder_ptq4sam_uint8/circle_fake_quant_20260426')
            / f'{quant_model.parent.name}_{quant_model.stem}.fake_quant.circle'
        )
    fake_quant_model.parent.mkdir(parents=True, exist_ok=True)
    if not (args.reuse_fake_quant and fake_quant_model.exists()):
        subprocess.run(
            [
                args.one_quantize,
                '--fake_quantize',
                '-i',
                str(quant_model),
                '-o',
                str(fake_quant_model),
            ],
            check=True,
        )
    args.circle_model = str(fake_quant_model)
    return args.circle_model


class CircleDecoderRunner:
    def __init__(self, model_path: str, lib_path: str):
        from cffi import FFI

        self.ffi = FFI()
        self.ffi.cdef(
            """
            typedef struct InterpreterWrapper InterpreterWrapper;
            const char *get_last_error(void);
            void clear_last_error(void);
            InterpreterWrapper *Interpreter_new(const uint8_t *data, const size_t data_size);
            void Interpreter_delete(InterpreterWrapper *intp);
            void Interpreter_interpret(InterpreterWrapper *intp);
            void Interpreter_writeInputTensor(InterpreterWrapper *intp, const int input_idx, const void *data, size_t input_size);
            void Interpreter_readOutputTensor(InterpreterWrapper *intp, const int output_idx, void *output, size_t output_size);
            """
        )
        self.lib = self.ffi.dlopen(lib_path)
        self.model_data = bytearray(Path(model_path).read_bytes())
        self.interpreter = self.lib.Interpreter_new(self.ffi.from_buffer(self.model_data), len(self.model_data))
        self._check_errors()

    def _check_errors(self) -> None:
        message = self.ffi.string(self.lib.get_last_error()).decode('utf-8')
        if message:
            self.lib.clear_last_error()
            raise RuntimeError(f'Circle interpreter error: {message}')

    def close(self) -> None:
        if getattr(self, 'interpreter', None) is not None:
            self.lib.Interpreter_delete(self.interpreter)
            self._check_errors()
            self.interpreter = None

    def __enter__(self) -> 'CircleDecoderRunner':
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @staticmethod
    def _contiguous_float32(value: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(value, dtype=np.float32)

    def run(
        self,
        image_embeddings: np.ndarray,
        point_embedding_pe: np.ndarray,
        point_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        inputs = (
            self._contiguous_float32(image_embeddings),
            self._contiguous_float32(point_embedding_pe),
            self._contiguous_float32(point_labels),
        )
        for index, array in enumerate(inputs):
            self.lib.Interpreter_writeInputTensor(
                self.interpreter,
                index,
                self.ffi.from_buffer(array),
                array.nbytes,
            )
            self._check_errors()

        self.lib.Interpreter_interpret(self.interpreter)
        self._check_errors()

        scores = np.empty((1, 4), dtype=np.float32)
        masks = np.empty((1, 4, 256, 256), dtype=np.float32)
        self.lib.Interpreter_readOutputTensor(self.interpreter, 0, self.ffi.from_buffer(scores), scores.nbytes)
        self._check_errors()
        self.lib.Interpreter_readOutputTensor(self.interpreter, 1, self.ffi.from_buffer(masks), masks.nbytes)
        self._check_errors()
        return scores, masks


def build_models(args: argparse.Namespace, device: str):
    import sys
    import types

    import torch

    if 'wandb' not in sys.modules:
        sys.modules['wandb'] = types.ModuleType('wandb')

    from edge_sam import SamPredictor, sam_model_registry
    from edge_sam.quantization.decoder_ptq_compare import _build_decoder_surface

    sam = sam_model_registry['edge_sam'](checkpoint=args.checkpoint, upsample_mode='bilinear')
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    fp32_decoder = _build_decoder_surface(args.checkpoint, use_stability_score=args.use_stability_score).to(device).eval()
    return sam, predictor, fp32_decoder


def compute_padded_point_pe(
    sam: Any,
    predictor: Any,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    original_size: tuple[int, int],
    *,
    num_points: int,
):
    import torch

    padded_coords, padded_labels = pad_point_prompt(point_coords, point_labels, num_points=num_points)
    transformed = predictor.transform.apply_coords(padded_coords, original_size)
    coords = torch.as_tensor(transformed, dtype=torch.float32, device=sam.device)[None, :, :]
    labels = torch.as_tensor(padded_labels, dtype=torch.float32, device=sam.device)[None, :]
    with torch.no_grad():
        point_embedding_pe = sam.prompt_encoder.pe_layer._pe_encoding((coords + 0.5) / sam.image_encoder.img_size)
    return point_embedding_pe, labels, padded_coords, padded_labels


def _maybe_subsample_annotations(
    annotations: list[dict[str, Any]],
    *,
    max_masks_per_image: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if max_masks_per_image > 0 and len(annotations) > max_masks_per_image:
        selected_indices = rng.choice(len(annotations), size=max_masks_per_image, replace=False)
        return [annotations[int(index)] for index in selected_indices]
    return annotations


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for EdgeSAM image feature extraction')

    materialize_circle_model(args)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda:0'
    sam, predictor, fp32_decoder = build_models(args, device=device)
    args.fp32_mask_threshold, args.circle_mask_threshold = resolve_mask_thresholds(args, float(sam.mask_threshold))
    image_records = load_coco_annotations(args.ann_file, args.max_images)

    per_mask_records: list[dict[str, Any]] = []
    per_image_records: list[dict[str, Any]] = []

    with CircleDecoderRunner(args.circle_model, args.circle_interpreter_lib) as circle_decoder:
        for image_info, annotations in image_records:
            image_path = os.path.join(args.img_dir, image_info['file_name'])
            if not os.path.exists(image_path):
                continue

            image = np.array(Image.open(image_path).convert('RGB'))
            features = predictor.set_image(image)
            original_size = tuple(int(v) for v in predictor.original_size)
            input_size = tuple(int(v) for v in predictor.input_size)
            image_level_records: list[dict[str, Any]] = []
            annotations = _maybe_subsample_annotations(
                annotations,
                max_masks_per_image=args.max_masks_per_image,
                rng=rng,
            )

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
                point_embedding_pe, label_tensor, padded_coords, padded_labels = compute_padded_point_pe(
                    sam,
                    predictor,
                    point_coords,
                    point_labels,
                    original_size,
                    num_points=args.circle_num_points,
                )

                with torch.no_grad():
                    fp32_scores, fp32_masks = fp32_decoder(features, point_embedding_pe, label_tensor)
                    fp32_masks = sam.postprocess_masks(fp32_masks, input_size, original_size)

                circle_scores, circle_masks = circle_decoder.run(
                    features.detach().cpu().numpy(),
                    point_embedding_pe.detach().cpu().numpy(),
                    label_tensor.detach().cpu().numpy(),
                )
                circle_masks_tensor = torch.from_numpy(circle_masks)
                circle_masks_tensor = sam.postprocess_masks(circle_masks_tensor, input_size, original_size)

                fp32_binary = (fp32_masks[0] > args.fp32_mask_threshold).detach().cpu().numpy()
                circle_binary = (circle_masks_tensor[0] > args.circle_mask_threshold).detach().cpu().numpy()
                fp32_summary = summarize_candidate_masks(
                    fp32_binary,
                    fp32_scores[0].detach().cpu().numpy(),
                    gt_mask,
                )
                circle_summary = summarize_candidate_masks(
                    circle_binary,
                    circle_scores[0],
                    gt_mask,
                )

                record: dict[str, Any] = {
                    'image_id': int(image_info['id']),
                    'file_name': image_info['file_name'],
                    'ann_index': ann_index,
                    'point_coords': point_coords.tolist(),
                    'point_labels': point_labels.tolist(),
                    'circle_point_coords': padded_coords.tolist(),
                    'circle_point_labels': padded_labels.tolist(),
                }
                for prefix, summary in (('fp32', fp32_summary), ('circle', circle_summary)):
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
        output_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'Saved summary to {output_path}')
    print(json.dumps(payload['summary'], indent=2))


if __name__ == '__main__':
    main()
