# EdgeSAM decoder quantization experiments, 2026-04-25

## Setup

- Model: EdgeSAM decoder surface from `/home/kitemanul/project/EdgeSAM/weights/edge_sam.pth`.
- Calibration: `decoder_datalist_20.txt`, 20 decoder samples.
- Tensor similarity eval: `decoder_test_5.txt`, 5 decoder samples.
- Real IoU eval: COCO val2017 first 200 images, 1893 masks, center point prompt, mask threshold 0.0 unless noted.
- Metrics kept: score/mask MSE, top1, top5, cosine similarity, model selected mIoU, and oracle mIoU. Oracle mIoU means choosing the candidate mask with the highest IoU to GT, so it isolates segmentation quality from score ranking.

## Real IoU summary

| Candidate | Selected mIoU | Oracle mIoU | Delta selected | Delta oracle | Notes |
|---|---:|---:|---:|---:|---|
| FP32 EdgeSAM baseline | 0.503478 | 0.652608 | 0.000000 | 0.000000 | Reference decoder |
| Transformer UINT8 + BIG + AGQ | 0.517905 | 0.647591 | +0.014427 | -0.005017 | Transformer quantization is mostly safe |
| Full decoder plain UINT8 | 0.472424 | 0.626311 | -0.031053 | -0.026297 | Full tail UINT8 hurts masks |
| Full decoder UINT8 + BIG only | 0.473206 | 0.625698 | -0.030271 | -0.026910 | BIG alone helps little in full tail |
| Full decoder UINT8 + AGQ only | 0.458496 | 0.610643 | -0.044981 | -0.041965 | AGQ alone is worse here |
| Full decoder UINT8 + BIG + AGQ | 0.456875 | 0.611060 | -0.046602 | -0.041548 | Full tail quantization dominates the loss |
| Full BIG+AGQ with hyper output and hyper input kept FP32 | 0.490742 | 0.647764 | -0.012736 | -0.004844 | Confirms mask projection hyper path is the sensitive area |
| Mixed: transformer UINT8 BIG+AGQ, mask head W8A16 signed per-channel | 0.521188 | 0.632268 | +0.017710 | -0.020341 | Best practical mixed candidate |
| Mixed: transformer UINT8 BIG+AGQ, mask head W16A16 signed per-channel | 0.521442 | 0.632240 | +0.017964 | -0.020369 | No meaningful gain over W8A16 |

## Tensor similarity summary

| Candidate | Scores MSE | Scores top1 | Scores top5 | Scores cosine | Masks MSE | Masks top1 | Masks top5 | Masks cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full BIG+AGQ with hyper input FP32 | 0.001132 | 1.000000 | 1.000000 | 0.999344 | 0.201965 | 0.800000 | 0.800000 | 0.999536 |
| Full BIG+AGQ hyper input INT16 per-channel | 0.001916 | 0.800000 | 1.000000 | 0.998851 | 0.942367 | 0.600000 | 0.680000 | 0.998217 |
| Mixed W8A16 signed per-channel mask head | 0.002907 | 0.800000 | 1.000000 | 0.996518 | 0.978696 | 0.800000 | 0.800000 | 0.998814 |
| Mixed W16A16 signed per-channel mask head | 0.002888 | 0.800000 | 1.000000 | 0.996571 | 0.987780 | 0.400000 | 0.800000 | 0.998847 |
| Plain PyTorch full UINT8 | 0.006579 | 1.000000 | 1.000000 | 0.986764 | 1.174698 | 0.600000 | 0.560000 | 0.997712 |

## Diagnosis

- BIG and AGQ are useful for the transformer. The transformer-only experiment loses only 0.005 oracle mIoU and even improves selected mIoU because score ranking changes slightly.
- The regression in full BIG+AGQ is not caused by BIG or AGQ themselves. It appears when the post-transformer mask head is also forced into UINT8, especially the hypernetwork output and mask projection inputs.
- MAE and cosine can look good while real IoU drops because masks are thresholded spatial logits. Small logit shifts near the boundary can flip many pixels, and cosine does not measure topology or threshold sensitivity.
- Keeping hyper output and mask projection hyper input in FP32 nearly recovers oracle mIoU, which localizes the sensitive path.
- The best current mixed strategy is W8A16 signed per-channel for mask head activations. W16A16 does not improve real IoU enough to justify more INT16 weight coverage.

## Deployment conclusion

Recommended quantization policy:

- Transformer: UINT8 weights and activations, BIG enabled, AGQ enabled.
- Mask head weights: UINT8 per-channel where available.
- Mask head activations: signed INT16 per-channel.
- Channel axes: feature/token/mask tensors use channel axis 1; hyper_in into mask projection uses channel axis 2.
- Score path: keep stability score for the NPU-safe decoder path, while reporting selected and oracle mIoU separately.

ONE and onecc caveat:

- Vanilla onecc quantizes both weights and activations, but activation quantization is per-layer scalar, not per-channel.
- Therefore the current recommended W8A16 per-channel activation policy cannot be reproduced exactly by just setting onecc quantized_dtype.
- PyTorch quantize then dequantize to plain FP32 can validate fake-quant numerics, but it is not a real UINT8 or INT16 deployment. For deployment, preserve explicit Q/DQ quantization in ONNX/Circle or extend ONE quantization support, then verify the final Circle or backend artifact really contains quantized execution.
