# EdgeSAM Decoder ONE Quantization Strategy, 2026-04-26

## Status

This document records the current deployable ONE/onecc strategy for the EdgeSAM NPU-safe
decoder. It is separate from `docs/edgesam_decoder_quantization_report_20260425.md`, which
focused on PyTorch/PTQ4SAM fake-quant policies.

Current decision:

- Recommended onecc-first policy: full decoder uint8, channel granularity, percentile
  calibration with 20 samples, stability-score output, static 5 prompt slots.
- Reason: it is the most uint8-heavy policy, passes the local onecc import/opt/quant path,
  and is the best Circle real-IoU result among the deployable mostly-uint8 candidates when
  selected mIoU is the primary metric.
- Important caveat: transformer-int16 plus uint8 elsewhere has much better tensor-level
  MAE/MSE/top-k metrics and slightly better oracle mIoU on the 50-prompt Circle subset, but
  its selected mIoU is worse. Treat it as a tensor/oracle-quality fallback, not the selected
  IoU winner.

## User Question Check

The concern was:

> If transformer-only looks clearly better, why is plain uint8 better than uint8+BIG+AGQ?

The missing control was run on 2026-04-26:

| Scope | BIG | AGQ | Selected mIoU | Oracle mIoU | Result |
|---|---:|---:|---:|---:|---|
| transformer | no | no | 0.514804 | 0.650462 | `real_iou_coco_val200_transformer_plain_uint8_stability_20260426` |
| transformer | yes | yes | 0.517905 | 0.647591 | `real_iou_coco_val200_transformer_big_agq_uint8_with_oracle_20260425` |
| full | no | no | 0.472424 | 0.626311 | `real_iou_coco_val200_plain_uint8_with_oracle_20260425` |
| full | yes | yes | 0.456875 | 0.611060 | `real_iou_coco_val200_big_agq_uint8_with_oracle_20260425` |

Conclusion:

- The premise is only weakly true for real IoU. Transformer BIG+AGQ is only +0.003101
  selected mIoU over transformer plain uint8, and has slightly lower oracle mIoU.
- Full plain uint8 really is better than full uint8+BIG+AGQ in the COCO val200 real-IoU
  run.
- The reversal is not a script mismatch. The transformer-only scope only quantizes
  `model.transformer`; the full scope additionally quantizes label embedding, tokens,
  output upscaling, hypernetworks, mask projection, and score head. The loss appears when
  the post-transformer decoder tail is forced into uint8.

## ONE/onecc Candidate Results

All onecc tensor metrics below use:

- Input model: `/home/kitemanul/project/EdgeSAM/weights/one_pipeline/decoder/edge_sam_decoder.opt.circle`
- Calibration: `decoder_calib_20.h5`
- Test data: `decoder_test_5.h5`
- Quantization: channel granularity, float32 model input/output for the compared candidate
  artifacts unless noted.

| onecc policy | /Div MAE | /Div MSE | /Div top1 | /Reshape_6 MAE | /Reshape_6 MSE | /Reshape_6 top1 | /Reshape_6 top5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full uint8 | 0.059993 | 0.009561 | 0.600 | 1.133120 | 2.122860 | 0.400 | 0.400 |
| transformer int16, rest uint8 | 0.016502 | 0.000481 | 0.800 | 0.343812 | 0.199735 | 1.000 | 0.760 |
| score branch int16, rest uint8 | 0.069125 | 0.011193 | 0.600 | 1.133120 | 2.122860 | 0.400 | 0.400 |
| mask head int16, rest uint8 | 0.059183 | 0.010804 | 0.600 | 1.047360 | 1.809720 | 0.400 | 0.480 |
| full int16 upper bound | 0.001479 | 0.000006 | 0.800 | 0.032319 | 0.002460 | 1.000 | 1.000 |

Interpretation:

- Tensor metrics strongly favor transformer-int16 and full-int16.
- Score-branch-only int16 is rejected: it worsens `/Div` while leaving mask metrics unchanged.
- Mask-head int16 is rejected for now: it improves mask MSE only modestly and leaves score
  ranking metrics poor.
- Full int16 is an accuracy upper bound, but it violates the "use uint8 as much as possible"
  objective.

## Circle Real-IoU Evaluation

`scripts/eval_edgesam_decoder_circle_real_iou.py` was added to evaluate ONE Circle artifacts.

Why it uses fake-quant Circle:

- Direct `circle-interpreter` execution of `quant.circle` fails locally on quantized `Mul`
  kernels.
- `one-quantize --fake_quantize` converts the quantized Circle into an FP32 fake-quant Circle.
  This is the same comparison route onecc uses for `circle-eval-diff`.

Prompt contract:

- The NPU decoder has static prompt shape `[1, 5, 256]` and labels `[1, 5]`.
- For one-point real-IoU, the script pads to labels `[1, -1, -1, -1, -1]`.
- Padding coordinates are zero-filled; label `-1` causes the decoder label embedding to ignore
  their positional encoding.

Runtime caveat:

- The local luci interpreter takes about 9 seconds per decoder prompt.
- Full COCO val200 with 1893 masks would take hours, so Circle real-IoU is currently reported
  on controlled subsets. PyTorch fake-quant real-IoU remains the full val200 reference.

Primary 50-prompt Circle subset:

| onecc policy | FP32 selected | Circle selected | Delta selected | FP32 oracle | Circle oracle | Delta oracle | Circle rank hit@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full uint8 | 0.530146 | 0.501739 | -0.028407 | 0.671402 | 0.665379 | -0.006023 | 0.240 |
| transformer int16, rest uint8 | 0.530146 | 0.491812 | -0.038334 | 0.671402 | 0.668974 | -0.002428 | 0.200 |

Small upper-bound 10-prompt subset:

| onecc policy | Circle selected | Delta selected | Circle oracle | Delta oracle |
|---|---:|---:|---:|---:|
| full uint8 | 0.505679 | -0.021851 | 0.637610 | +0.000441 |
| transformer int16, rest uint8 | 0.519651 | -0.007879 | 0.641569 | +0.004400 |
| full int16 upper bound | 0.520424 | -0.007106 | 0.639782 | +0.002613 |

The 10-prompt subset favored transformer-int16, but the 50-prompt subset reversed selected
mIoU while preserving the oracle pattern. This points to score-ranking sensitivity rather
than mask-quality collapse.

## Final Policy Choice

Use full uint8 as the current onecc-first policy:

- It uses the most uint8.
- It has a known-good onecc import/opt/quant artifact:
  `results/onecc_npu_safe_decoder_uint8_20260425/edge_sam_decoder.quant.circle`
- It matches the c20 tensor metrics in the EdgeSAM `decoder_uint8_c20_t5` artifact.
- It is the better selected real-IoU policy on the larger Circle subset:
  `results/edgesam_decoder_ptq4sam_uint8/circle_real_iou_coco_val50x1_uint8_20260426/summary.json`

Keep transformer-int16 as fallback:

- Artifact:
  `/home/kitemanul/project/EdgeSAM/weights/one_pipeline/decoder_transformer_int16_c20_t5/edge_sam_decoder.quant.circle`
- Use it when tensor similarity or oracle mask quality is more important than selected mIoU.
- Do not call it the selected-IoU winner until a larger/faster backend real-IoU run reverses
  the 50-prompt result.

Reject for now:

- full uint8+BIG+AGQ as a onecc strategy: BIG/AGQ are PTQ4SAM fake-quant features and are not
  represented by vanilla onecc quantization.
- score-branch-only int16: tensor score error worsened.
- learned-IoU/no-stability-score path: PyTorch real-IoU selected mIoU was much worse than
  stability-score output.
- full int16: useful upper bound, not aligned with the uint8 objective.

## Reproduction Commands

Run the missing transformer plain real-IoU control:

```bash
env PYTHONPATH=. /home/kitemanul/miniconda3/envs/edgesam/bin/python \
  scripts/eval_edgesam_decoder_real_iou.py \
  --ann-file /home/kitemanul/dataset/coco2017/annotations/instances_val2017.json \
  --img-dir /home/kitemanul/dataset/coco2017/val2017 \
  --scope transformer \
  --disable-big \
  --disable-agq \
  --calibration-count 20 \
  --max-images 200 \
  --point-strategy center \
  --fp32-mask-threshold 0.0 \
  --uint8-mask-threshold 0.0 \
  --output results/edgesam_decoder_ptq4sam_uint8/real_iou_coco_val200_transformer_plain_uint8_stability_20260426/summary.json
```

Run Circle real-IoU on full uint8:

```bash
env PYTHONPATH=. /home/kitemanul/miniconda3/envs/edgesam/bin/python \
  scripts/eval_edgesam_decoder_circle_real_iou.py \
  --ann-file /home/kitemanul/dataset/coco2017/annotations/instances_val2017.json \
  --img-dir /home/kitemanul/dataset/coco2017/val2017 \
  --quant-circle-model /home/kitemanul/project/EdgeSAM/weights/one_pipeline/decoder_uint8_c20_t5/edge_sam_decoder.quant.circle \
  --max-images 50 \
  --max-masks-per-image 1 \
  --point-strategy center \
  --fp32-mask-threshold 0.0 \
  --circle-mask-threshold 0.0 \
  --output results/edgesam_decoder_ptq4sam_uint8/circle_real_iou_coco_val50x1_uint8_20260426/summary.json
```

Run Circle real-IoU on transformer-int16 fallback:

```bash
env PYTHONPATH=. /home/kitemanul/miniconda3/envs/edgesam/bin/python \
  scripts/eval_edgesam_decoder_circle_real_iou.py \
  --ann-file /home/kitemanul/dataset/coco2017/annotations/instances_val2017.json \
  --img-dir /home/kitemanul/dataset/coco2017/val2017 \
  --quant-circle-model /home/kitemanul/project/EdgeSAM/weights/one_pipeline/decoder_transformer_int16_c20_t5/edge_sam_decoder.quant.circle \
  --max-images 50 \
  --max-masks-per-image 1 \
  --point-strategy center \
  --fp32-mask-threshold 0.0 \
  --circle-mask-threshold 0.0 \
  --output results/edgesam_decoder_ptq4sam_uint8/circle_real_iou_coco_val50x1_transformer_int16_20260426/summary.json
```

Run the rejected score-int16 onecc probe:

```bash
/home/kitemanul/project/ONE/build/py312/compiler/one-cmds/onecc \
  -C /home/kitemanul/project/PTQ4SAM/results/onecc_decoder_uint8_score_int16_20260426/decoder_score_int16_onecc_quant.cfg
```

## Open Risk

The final backend/runtime path is still not a full deployment proof:

- onecc import/opt/quant is verified locally.
- Quantized Circle direct execution in `circle-interpreter` is blocked by unsupported quantized
  kernels, so real-IoU uses fake-quant Circle.
- Backend code generation/runtime execution should be verified separately when a real target
  backend or a faster quantized Circle runner is available.
