from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.eval_edgesam_decoder_real_iou import (
    aggregate_prompt_metrics,
    decode_mask,
    load_coco_annotations,
    sample_points,
    summarize_candidate_masks,
)
from scripts.eval_edgesam_decoder_circle_real_iou import (
    DEFAULT_CIRCLE_INTERPRETER_LIB,
    DEFAULT_ONE_QUANTIZE,
    pad_point_prompt,
)


PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)[None, :, None, None]
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)[None, :, None, None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate two EdgeSAM encoder+decoder Circle pipelines on COCO-format masks.'
    )
    parser.add_argument('--checkpoint', default='/home/kitemanul/project/EdgeSAM/weights/edge_sam.pth')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--fp32-encoder-circle', required=True)
    parser.add_argument('--fp32-decoder-circle', required=True)
    parser.add_argument('--quant-encoder-circle', required=True)
    parser.add_argument('--quant-decoder-circle', required=True)
    parser.add_argument('--fake-quant-dir', required=True)
    parser.add_argument('--one-quantize', default=DEFAULT_ONE_QUANTIZE)
    parser.add_argument('--circle-interpreter-lib', default=DEFAULT_CIRCLE_INTERPRETER_LIB)
    parser.add_argument('--fp32-size', type=int, default=1024)
    parser.add_argument('--quant-size', type=int, default=768)
    parser.add_argument('--num-points', type=int, default=1)
    parser.add_argument('--circle-num-points', type=int, default=5)
    parser.add_argument('--point-strategy', choices=('random', 'center', 'bbox_center'), default='center')
    parser.add_argument('--max-images', type=int, default=10)
    parser.add_argument('--max-masks-per-image', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mask-threshold', type=float, default=0.0)
    parser.add_argument('--output', default=None)
    parser.add_argument('--save-per-mask', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--reuse-fake-quant', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


class CircleRunner:
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

    @staticmethod
    def _contiguous_float32(value: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(value, dtype=np.float32)

    def run(self, inputs: tuple[np.ndarray, ...], output_shapes: tuple[tuple[int, ...], ...]) -> tuple[np.ndarray, ...]:
        for index, value in enumerate(inputs):
            array = self._contiguous_float32(value)
            self.lib.Interpreter_writeInputTensor(
                self.interpreter,
                index,
                self.ffi.from_buffer(array),
                array.nbytes,
            )
            self._check_errors()

        self.lib.Interpreter_interpret(self.interpreter)
        self._check_errors()

        outputs = []
        for index, shape in enumerate(output_shapes):
            output = np.empty(shape, dtype=np.float32)
            self.lib.Interpreter_readOutputTensor(
                self.interpreter,
                index,
                self.ffi.from_buffer(output),
                output.nbytes,
            )
            self._check_errors()
            outputs.append(output)
        return tuple(outputs)

    def close(self) -> None:
        if getattr(self, 'interpreter', None) is not None:
            self.lib.Interpreter_delete(self.interpreter)
            self._check_errors()
            self.interpreter = None

    def __enter__(self) -> 'CircleRunner':
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class Pipeline:
    def __init__(self, encoder: CircleRunner, decoder: CircleRunner, *, model_size: int, prompt_helper: Any):
        self.encoder = encoder
        self.decoder = decoder
        self.model_size = model_size
        self.embed_size = model_size // 16
        self.low_res_size = model_size // 4
        self.prompt_helper = prompt_helper

    def encode(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        input_image, input_size, original_size = preprocess_image(image, self.model_size)
        (features,) = self.encoder.run((input_image,), ((1, 256, self.embed_size, self.embed_size),))
        return features, input_size, original_size

    def decode(
        self,
        features: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        original_size: tuple[int, int],
        *,
        num_points: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        point_pe, labels, padded_coords, padded_labels = self.prompt_helper.compute(
            point_coords,
            point_labels,
            original_size,
            model_size=self.model_size,
            num_points=num_points,
        )
        scores, masks = self.decoder.run(
            (features, point_pe, labels),
            ((1, 4), (1, 4, self.low_res_size, self.low_res_size)),
        )
        return scores, masks, padded_coords, padded_labels


class PromptHelper:
    def __init__(self, checkpoint: str):
        import sys
        import types

        import torch

        if 'wandb' not in sys.modules:
            sys.modules['wandb'] = types.ModuleType('wandb')

        from edge_sam import sam_model_registry
        from edge_sam.utils.transforms import ResizeLongestSide

        self.torch = torch
        self.ResizeLongestSide = ResizeLongestSide
        self.sam = sam_model_registry['edge_sam'](checkpoint=checkpoint, upsample_mode='bilinear')
        self.sam.eval()

    def compute(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        original_size: tuple[int, int],
        *,
        model_size: int,
        num_points: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        padded_coords, padded_labels = pad_point_prompt(point_coords, point_labels, num_points=num_points)
        transformed = self.ResizeLongestSide(model_size).apply_coords(padded_coords, original_size)
        coords = self.torch.as_tensor(transformed, dtype=self.torch.float32)[None, :, :]
        with self.torch.no_grad():
            point_pe = self.sam.prompt_encoder.pe_layer._pe_encoding((coords + 0.5) / model_size)
        labels = padded_labels[None, :].astype(np.float32)
        return point_pe.detach().cpu().numpy(), labels, padded_coords, padded_labels


def preprocess_image(image_np: np.ndarray, model_size: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    from edge_sam.utils.transforms import ResizeLongestSide

    original_size = image_np.shape[:2]
    resized = ResizeLongestSide(model_size).apply_image(image_np)
    input_size = resized.shape[:2]
    x = resized.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
    x = (x - PIXEL_MEAN) / PIXEL_STD
    h, w = x.shape[-2:]
    x = np.pad(x, ((0, 0), (0, 0), (0, model_size - h), (0, model_size - w)), mode='constant')
    return x, tuple(int(v) for v in input_size), tuple(int(v) for v in original_size)


def postprocess_masks(
    low_res_masks: np.ndarray,
    *,
    model_size: int,
    input_size: tuple[int, int],
    original_size: tuple[int, int],
) -> np.ndarray:
    masks = []
    for mask in low_res_masks[0]:
        resized = cv2.resize(mask, (model_size, model_size), interpolation=cv2.INTER_LINEAR)
        cropped = resized[: input_size[0], : input_size[1]]
        full = cv2.resize(cropped, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
        masks.append(full)
    return np.stack(masks, axis=0)


def maybe_fake_quantize(args: argparse.Namespace, quant_path: str, label: str) -> str:
    fake_dir = Path(args.fake_quant_dir)
    fake_dir.mkdir(parents=True, exist_ok=True)
    fake_path = fake_dir / f'{label}.fake_quant.circle'
    if args.reuse_fake_quant and fake_path.exists():
        return str(fake_path)
    subprocess.run(
        [args.one_quantize, '--fake_quantize', '-i', quant_path, '-o', str(fake_path)],
        check=True,
    )
    return str(fake_path)


def maybe_subsample_annotations(
    annotations: list[dict[str, Any]],
    *,
    max_masks_per_image: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if max_masks_per_image > 0 and len(annotations) > max_masks_per_image:
        indices = rng.choice(len(annotations), size=max_masks_per_image, replace=False)
        return [annotations[int(index)] for index in indices]
    return annotations


def summarize_image_records(image_id: int, file_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {'image_id': image_id, 'file_name': file_name, 'num_masks': len(records)}
    for prefix in ('fp32', 'quant'):
        selected = np.array([record[f'{prefix}_selected_iou'] for record in records], dtype=np.float64)
        oracle = np.array([record[f'{prefix}_oracle_iou'] for record in records], dtype=np.float64)
        result[f'{prefix}_mean_iou'] = float(selected.mean())
        result[f'{prefix}_oracle_mean_iou'] = float(oracle.mean())
    result['delta_mean_iou'] = float(result['quant_mean_iou'] - result['fp32_mean_iou'])
    result['delta_oracle_mean_iou'] = float(result['quant_oracle_mean_iou'] - result['fp32_oracle_mean_iou'])
    return result


def build_summary(args: argparse.Namespace, per_mask_records: list[dict[str, Any]], per_image_records: list[dict[str, Any]]) -> dict[str, Any]:
    fp32 = aggregate_prompt_metrics(per_mask_records, 'fp32')
    quant = aggregate_prompt_metrics(per_mask_records, 'quant')
    summary: dict[str, Any] = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'ann_file': str(Path(args.ann_file).resolve()),
        'img_dir': str(Path(args.img_dir).resolve()),
        'fp32_encoder_circle': str(Path(args.fp32_encoder_circle).resolve()),
        'fp32_decoder_circle': str(Path(args.fp32_decoder_circle).resolve()),
        'quant_encoder_circle': str(Path(args.quant_encoder_circle).resolve()),
        'quant_decoder_circle': str(Path(args.quant_decoder_circle).resolve()),
        'fp32_size': args.fp32_size,
        'quant_size': args.quant_size,
        'images_requested': args.max_images,
        'images_processed': len(per_image_records),
        'masks_evaluated': len(per_mask_records),
        'point_strategy': args.point_strategy,
        'num_points': args.num_points,
        'circle_num_points': args.circle_num_points,
        'mask_threshold': args.mask_threshold,
        **fp32,
        **quant,
    }
    summary['delta_selected_miou'] = float(summary['quant_selected_miou'] - summary['fp32_selected_miou'])
    summary['delta_oracle_miou'] = float(summary['quant_oracle_miou'] - summary['fp32_oracle_miou'])
    summary['fp32_miou'] = summary['fp32_selected_miou']
    summary['quant_miou'] = summary['quant_selected_miou']
    summary['delta_miou'] = summary['delta_selected_miou']
    payload: dict[str, Any] = {'summary': summary, 'per_image': per_image_records}
    if args.save_per_mask:
        payload['per_mask'] = per_mask_records
    return payload


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    from PIL import Image

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    prompt_helper = PromptHelper(args.checkpoint)
    quant_encoder = maybe_fake_quantize(args, args.quant_encoder_circle, 'quant_encoder')
    quant_decoder = maybe_fake_quantize(args, args.quant_decoder_circle, 'quant_decoder')
    image_records = load_coco_annotations(args.ann_file, args.max_images)

    per_mask_records: list[dict[str, Any]] = []
    per_image_records: list[dict[str, Any]] = []

    with (
        CircleRunner(args.fp32_encoder_circle, args.circle_interpreter_lib) as fp32_encoder,
        CircleRunner(args.fp32_decoder_circle, args.circle_interpreter_lib) as fp32_decoder,
        CircleRunner(quant_encoder, args.circle_interpreter_lib) as q_encoder,
        CircleRunner(quant_decoder, args.circle_interpreter_lib) as q_decoder,
    ):
        fp32_pipeline = Pipeline(fp32_encoder, fp32_decoder, model_size=args.fp32_size, prompt_helper=prompt_helper)
        quant_pipeline = Pipeline(q_encoder, q_decoder, model_size=args.quant_size, prompt_helper=prompt_helper)

        for image_info, annotations in image_records:
            image_path = os.path.join(args.img_dir, image_info['file_name'])
            if not os.path.exists(image_path):
                continue
            image = np.array(Image.open(image_path).convert('RGB'))
            fp32_features, fp32_input_size, original_size = fp32_pipeline.encode(image)
            quant_features, quant_input_size, _ = quant_pipeline.encode(image)
            image_level_records: list[dict[str, Any]] = []
            annotations = maybe_subsample_annotations(
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
                fp32_scores, fp32_low_res, fp32_padded_coords, fp32_padded_labels = fp32_pipeline.decode(
                    fp32_features,
                    point_coords,
                    point_labels,
                    original_size,
                    num_points=args.circle_num_points,
                )
                quant_scores, quant_low_res, quant_padded_coords, quant_padded_labels = quant_pipeline.decode(
                    quant_features,
                    point_coords,
                    point_labels,
                    original_size,
                    num_points=args.circle_num_points,
                )

                fp32_masks = postprocess_masks(
                    fp32_low_res,
                    model_size=args.fp32_size,
                    input_size=fp32_input_size,
                    original_size=original_size,
                )
                quant_masks = postprocess_masks(
                    quant_low_res,
                    model_size=args.quant_size,
                    input_size=quant_input_size,
                    original_size=original_size,
                )
                fp32_binary = fp32_masks > args.mask_threshold
                quant_binary = quant_masks > args.mask_threshold
                fp32_summary = summarize_candidate_masks(fp32_binary, fp32_scores[0], gt_mask)
                quant_summary = summarize_candidate_masks(quant_binary, quant_scores[0], gt_mask)
                record: dict[str, Any] = {
                    'image_id': int(image_info['id']),
                    'file_name': image_info['file_name'],
                    'ann_index': ann_index,
                    'point_coords': point_coords.tolist(),
                    'point_labels': point_labels.tolist(),
                    'fp32_point_coords': fp32_padded_coords.tolist(),
                    'fp32_point_labels': fp32_padded_labels.tolist(),
                    'quant_point_coords': quant_padded_coords.tolist(),
                    'quant_point_labels': quant_padded_labels.tolist(),
                }
                for prefix, summary in (('fp32', fp32_summary), ('quant', quant_summary)):
                    for key, value in summary.items():
                        record[f'{prefix}_{key}'] = value
                image_level_records.append(record)
                per_mask_records.append(record)

            if image_level_records:
                per_image_records.append(
                    summarize_image_records(int(image_info['id']), image_info['file_name'], image_level_records)
                )

    return build_summary(args, per_mask_records, per_image_records)


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
