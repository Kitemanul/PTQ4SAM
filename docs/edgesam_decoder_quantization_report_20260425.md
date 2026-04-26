# EdgeSAM Decoder Quantization Report, 2026-04-25

## Status

This document is the canonical summary for the EdgeSAM decoder quantization work in this
repository as of 2026-04-25. It consolidates the experiment notes, result artifacts, and
deployment findings into one stable reference.

For the later onecc-first strategy decision and Circle real-IoU checks from 2026-04-26,
see `docs/edgesam_decoder_onecc_quant_strategy_20260426.md`.

The short conclusion is:

- Quantizing the EdgeSAM decoder transformer with UINT8 weights and activations is viable.
- PTQ4SAM BIG and AGQ are useful in the transformer path.
- Full UINT8 quantization of the whole decoder is not viable because the mask head is
  sensitive, especially the `output_hypernetworks -> mask_projection` path.
- The current recommended mixed policy is transformer UINT8 with BIG+AGQ, mask-head
  weights as UINT8, and mask-head activations as signed INT16 per-channel.
- Vanilla `onecc` quantization config cannot express this mixed per-channel INT16
  activation policy. Deployment should preserve explicit Q/DQ in ONNX/Circle or extend ONE
  quantization support.

## Source Artifacts

Primary project files:

- `scripts/edgesam_decoder_ptq4sam_uint8.py`: PTQ4SAM-style EdgeSAM decoder wrappers,
  calibration, and tensor-similarity evaluation.
- `scripts/eval_edgesam_decoder_real_iou.py`: COCO real-IoU evaluation for model-selected
  and oracle-selected masks.
- `tools/run_edgesam_recommended_quant.py`: entry point for the recommended mixed policy.
- `scripts/export_edgesam_decoder_ptq4sam_onnx.py`: FP32 ONNX export from calibrated
  fake-quant models. This is useful for numerical inspection but is not a real UINT8 Q/DQ
  deployment graph.
- `tools/postprocess_edgesam_qdq_for_onecc.py`: ONNX Q/DQ compatibility postprocessor for
  local ONE/circle-mlir import and optimization.
- `docs/edgesam_quant_experiments_20260425.md`: original chronological experiment notes.

Primary result directories:

- `results/edgesam_decoder_ptq4sam_uint8/`: tensor-similarity and real-IoU experiment
  summaries.
- `results/onecc_qdq_smoke_20260425/`: explicit-Q/DQ ONNX and ONE import/opt smoke
  artifacts.
- `results/onecc_npu_safe_decoder_uint8_20260425/`: vanilla onecc UINT8 quantization
  artifacts for the NPU-safe decoder path.

## Evaluation Setup

Model:

- EdgeSAM checkpoint: `/home/kitemanul/project/EdgeSAM/weights/edge_sam.pth`
- Decoder surface from EdgeSAM, using the NPU-safe stability-score path unless an experiment
  explicitly disables it.

Calibration:

- Datalist: `/home/kitemanul/project/EdgeSAM/weights/one_pipeline/calib/decoder_datalist_20.txt`
- Sample count: 20 decoder samples.

Tensor-similarity evaluation:

- Datalist: `/home/kitemanul/project/EdgeSAM/weights/one_pipeline/calib/decoder_test_5.txt`
- Sample count: 5 decoder samples.
- Metrics: score MSE, mask MSE, top-1/top-5 selection agreement, cosine similarity.

Real-IoU evaluation:

- Dataset: COCO val2017.
- Annotation file: `/home/kitemanul/dataset/coco2017/annotations/instances_val2017.json`
- Image directory: `/home/kitemanul/dataset/coco2017/val2017`
- Split subset: first 200 images.
- Masks evaluated: 1893 masks.
- Prompt policy: one center point prompt per mask.
- Mask threshold: 0.0 in the reported real-IoU comparisons.
- Metrics: selected mIoU and oracle mIoU. Selected mIoU uses the decoder score ranking.
  Oracle mIoU picks the candidate mask with the highest IoU to the ground truth and isolates
  segmentation quality from score-ranking behavior.

## Key Real-IoU Results

| Candidate | Selected mIoU | Oracle mIoU | Delta selected | Delta oracle | Result source |
|---|---:|---:|---:|---:|---|
| FP32 EdgeSAM baseline | 0.503478 | 0.652608 | 0.000000 | 0.000000 | reference in real-IoU runs |
| Transformer UINT8 + BIG + AGQ | 0.517905 | 0.647591 | +0.014427 | -0.005017 | `real_iou_coco_val200_transformer_big_agq_uint8_with_oracle_20260425` |
| Full decoder plain UINT8 | 0.472424 | 0.626311 | -0.031053 | -0.026297 | `real_iou_coco_val200_plain_uint8_with_oracle_20260425` |
| Full decoder UINT8 + BIG only | 0.473206 | 0.625698 | -0.030271 | -0.026910 | `real_iou_coco_val200_big_only_uint8_stability_20260425` |
| Full decoder UINT8 + AGQ only | 0.458496 | 0.610643 | -0.044981 | -0.041965 | `real_iou_coco_val200_agq_only_uint8_stability_20260425` |
| Full decoder UINT8 + BIG + AGQ | 0.456875 | 0.611060 | -0.046602 | -0.041548 | `real_iou_coco_val200_big_agq_uint8_with_oracle_20260425` |
| Full BIG+AGQ, hyper output/input kept FP32 | 0.490742 | 0.647764 | -0.012736 | -0.004844 | `real_iou_coco_val200_big_agq_uint8_combo_no_hyper_output_plus_only_upscaled_quantized_20260425` |
| Mixed W8A16 per-channel signed mask head | 0.521188 | 0.632268 | +0.017710 | -0.020341 | `real_iou_coco_val200_mixed_uint8_big_agq_maskhead_w8a16pc_signed_stability_20260425` |
| Mixed W16A16 per-channel signed mask head | 0.521442 | 0.632240 | +0.017964 | -0.020369 | `real_iou_coco_val200_mixed_uint8_big_agq_maskhead_w16a16pc_signed_stability_20260425` |

Interpretation:

- Transformer quantization is safe enough for the current deployment target. The oracle mIoU
  loss is only about 0.005.
- Selected mIoU can improve even when oracle mIoU drops because quantization changes candidate
  mask score ranking. For deployment decisions, real-IoU metrics should be considered before
  tensor MSE or cosine metrics.
- Full decoder UINT8 quantization loses about 0.04 oracle mIoU with BIG+AGQ. The loss is not
  caused by BIG or AGQ alone; it appears when the post-transformer mask head is also forced
  into UINT8.
- Keeping the hypernetwork output and mask-projection hyper input in FP32 almost restores
  oracle mIoU, which localizes the sensitive path.

## Tensor-Similarity Findings

Representative 5-sample tensor-similarity results:

| Candidate | Scores MSE | Scores top1 | Scores cosine | Masks MSE | Masks top1 | Masks cosine |
|---|---:|---:|---:|---:|---:|---:|
| Full BIG+AGQ with hyper input FP32 | 0.001132 | 1.000000 | 0.999344 | 0.201965 | 0.800000 | 0.999536 |
| Full BIG+AGQ hyper input INT16 per-channel | 0.001916 | 0.800000 | 0.998851 | 0.942367 | 0.600000 | 0.998217 |
| Mixed W8A16 signed per-channel mask head | 0.002907 | 0.800000 | 0.996518 | 0.978696 | 0.800000 | 0.998814 |
| Mixed W16A16 signed per-channel mask head | 0.002888 | 0.800000 | 0.996571 | 0.987780 | 0.400000 | 0.998847 |
| Plain PyTorch full UINT8 | 0.006579 | 1.000000 | 0.986764 | 1.174698 | 0.600000 | 0.997712 |

Tensor similarity is useful for catching gross regressions, but it is not sufficient for this
decoder. Mask logits can keep high cosine similarity while thresholded masks lose spatial
quality near object boundaries. The real-IoU run is the deciding evaluation.

## Recommended Quantization Policy

Use this as the current policy for continued experiments:

- Decoder transformer:
  - weights: UINT8
  - activations: UINT8
  - BIG: enabled
  - AGQ: enabled
- Mask head:
  - weights: UINT8 per-channel where available
  - activations: signed INT16 per-channel
  - tail observer: `MinMaxObserver`
  - tail fake quantizer: `FixedFakeQuantize`
- Channel axes:
  - feature/token/mask tensors: channel axis 1
  - `hyper_in` into mask projection: channel axis 2
- Score path:
  - keep the NPU-safe stability-score path for decoder deployment experiments
  - report selected mIoU and oracle mIoU separately

Current best practical candidate:

- `tools/run_edgesam_recommended_quant.py`
- Policy name: `recommended_mixed_maskhead_w8a16pc_signed`
- Real-IoU artifact:
  `results/edgesam_decoder_ptq4sam_uint8/real_iou_coco_val200_mixed_uint8_big_agq_maskhead_w8a16pc_signed_stability_20260425/summary.json`

W16A16 did not improve real IoU enough to justify the heavier activation/weight coverage.

## Reproduction Commands

Run the recommended mixed policy:

```bash
/home/kitemanul/miniconda3/envs/edgesam/bin/python \
  tools/run_edgesam_recommended_quant.py \
  --stage both \
  --output-dir results/edgesam_recommended_mixed_quant/<run-name>
```

Run tensor-similarity evaluation for a basic transformer-only policy:

```bash
/home/kitemanul/miniconda3/envs/edgesam/bin/python \
  scripts/edgesam_decoder_ptq4sam_uint8.py \
  --scope transformer \
  --calibration-count 20 \
  --eval-count 5 \
  --output-dir results/edgesam_decoder_ptq4sam_uint8/<run-name>
```

Run real-IoU evaluation for a full decoder policy:

```bash
/home/kitemanul/miniconda3/envs/edgesam/bin/python \
  scripts/eval_edgesam_decoder_real_iou.py \
  --ann-file /home/kitemanul/dataset/coco2017/annotations/instances_val2017.json \
  --img-dir /home/kitemanul/dataset/coco2017/val2017 \
  --scope full \
  --calibration-count 20 \
  --max-images 200 \
  --point-strategy center \
  --fp32-mask-threshold 0.0 \
  --uint8-mask-threshold 0.0 \
  --output results/edgesam_decoder_ptq4sam_uint8/<run-name>/summary.json
```

Probe the sensitive hypernetwork/mask-projection path by keeping hyper output and hyper input
in FP32:

```bash
/home/kitemanul/miniconda3/envs/edgesam/bin/python \
  scripts/eval_edgesam_decoder_real_iou.py \
  --ann-file /home/kitemanul/dataset/coco2017/annotations/instances_val2017.json \
  --img-dir /home/kitemanul/dataset/coco2017/val2017 \
  --scope full \
  --no-quantize-hypernetwork-output \
  --no-quantize-mask-projection-hyper-input \
  --fp32-mask-threshold 0.0 \
  --uint8-mask-threshold 0.0 \
  --output results/edgesam_decoder_ptq4sam_uint8/<run-name>/summary.json
```

Export a calibrated fake-quant ONNX for inspection:

```bash
/home/kitemanul/miniconda3/envs/edgesam/bin/python \
  scripts/export_edgesam_decoder_ptq4sam_onnx.py \
  /home/kitemanul/project/EdgeSAM/weights/edge_sam.pth \
  --scope full \
  --calibration-count 20 \
  --num-points 5 \
  --output results/edgesam_decoder_ptq4sam_uint8/<run-name>/decoder.onnx
```

Important: the export command above produces an FP32 ONNX traced from the calibrated
fake-quant model. It does not produce a deployment-grade UINT8 or INT16 Q/DQ graph.

## ONE Deployment Status

Vanilla onecc route:

- `onecc` can quantize weights and activations from qconf, but activation quantization is
  scalar per layer, not per-channel.
- This cannot reproduce the recommended W8A16 per-channel activation policy.
- A PyTorch quantize-then-dequantize FP32 ONNX validates fake-quant numerics but does not
  prove real quantized deployment.

Explicit-Q/DQ route:

- The smoke experiment used explicit ONNX Q/DQ and compatibility postprocessing.
- Raw recommended affine-QDQ ONNX with AGQ hit ONE import issues around `Less`,
  `GreaterOrEqual`, dynamic `Pow`, optional `Clip`, constant `Cast`, and one dynamic reshape
  target.
- `tools/postprocess_edgesam_qdq_for_onecc.py` fixes those by:
  - staticizing `Reshape` shapes
  - staticizing `Slice` parameters
  - rewriting boolean comparison ops
  - rewriting dynamic base-2 `Pow` into `Exp`
  - lowering `Clip` to `Max`/`Min`
  - folding constant `Cast`
  - pruning dead nodes

Postprocess command:

```bash
/home/kitemanul/miniconda3/envs/edgesam/bin/python \
  tools/postprocess_edgesam_qdq_for_onecc.py \
  --input results/onecc_qdq_smoke_20260425/edgesam_recommended_affine_qdq_smoke.onnx \
  --output results/onecc_qdq_smoke_20260425/edgesam_recommended_affine_qdq_script_postprocessed.onnx
```

ONE import+opt command:

```bash
cd /home/kitemanul/project/ONE/build/py312/compiler/one-cmds
./onecc -C /home/kitemanul/project/PTQ4SAM/results/onecc_qdq_smoke_20260425/edgesam_recommended_affine_qdq_script_postprocessed_import_opt.cfg
```

Known-good import/opt artifact:

- Config:
  `results/onecc_qdq_smoke_20260425/edgesam_recommended_affine_qdq_script_postprocessed_import_opt.cfg`
- Input ONNX:
  `results/onecc_qdq_smoke_20260425/edgesam_recommended_affine_qdq_script_postprocessed.onnx`
- Imported Circle:
  `results/onecc_qdq_smoke_20260425/edgesam_recommended_affine_qdq_script_postprocessed.circle`
- Optimized Circle:
  `results/onecc_qdq_smoke_20260425/edgesam_recommended_affine_qdq_script_postprocessed.opt.circle`

The optimized Circle retained 130 `ONNXQuantizeLinear` and 130 `ONNXDequantizeLinear`
CustomOps in the smoke experiment. This verifies ONE import+opt compatibility, not final
backend code generation.

Backend status:

- Final backend codegen remains unverified in this repository state.
- The current py312 `one-cmds` setup did not provide a working final codegen path during the
  recorded smoke experiment.
- Do not treat an imported or optimized Circle as deployment-ready until backend artifact
  generation and runtime comparison are verified.

## Follow-Up Work

Recommended next steps:

1. Preserve or formalize the explicit-Q/DQ ONNX generation path for the recommended mixed
   policy. The current reusable postprocessor exists, but the exact Q/DQ exporter path should
   be made reproducible.
2. Add a deployment validation checklist that checks for:
   - explicit Q/DQ nodes in ONNX
   - preserved Q/DQ CustomOps in Circle
   - successful backend codegen artifact
   - numerical comparison between PyTorch fake-quant output and backend output
   - real-IoU spot checks on the final artifact
3. Avoid spending more time on full UINT8 decoder variants unless a new mask-head activation
   representation is available. Existing results already localize the failure mode.
4. Keep reporting selected and oracle mIoU together. Selected mIoU alone can hide score-ranking
   artifacts.
