import unittest
import sys
import types

import torch
import torch.nn as nn

if "wandb" not in sys.modules:
    sys.modules["wandb"] = types.ModuleType("wandb")

from edge_sam.modeling.common import LayerNorm2d
from edge_sam.modeling.mask_decoder import MLP
from edge_sam.modeling.transformer import TwoWayTransformer
from edge_sam.quantization.decoder_ptq_compare import (
    DecoderHypernetworkStack,
    DecoderLabelEmbedding,
    DecoderScoreHead,
)
from ptq4sam.quantization.state import enable_calibration_woquantization
from scripts.edgesam_decoder_ptq4sam_uint8 import (
    QuantDecoderHypernetworkStack,
    QuantDecoderLabelEmbedding,
    QuantDecoderScoreHead,
    QuantEdgeTwoWayTransformer,
    QuantFullDecoderSurface,
    QuantOutputUpscaling,
    make_qconfig,
    quantize_decoder_surface,
)


class _DummyDecoderSurface(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        embed_dim = 8
        self.label_embedding = DecoderLabelEmbedding(
            nn.ModuleList(nn.Embedding(1, embed_dim) for _ in range(4)),
            nn.Embedding(1, embed_dim),
        )
        self.transformer = TwoWayTransformer(
            depth=1,
            embedding_dim=embed_dim,
            num_heads=2,
            mlp_dim=16,
        )
        self.iou_token = nn.Embedding(1, embed_dim)
        self.mask_tokens = nn.Embedding(4, embed_dim)
        self.register_buffer("dense_embedding", torch.randn(1, embed_dim, 1, 1))
        self.register_buffer("image_pe", torch.randn(1, embed_dim, 1, 1))
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.output_hypernetworks = DecoderHypernetworkStack(
            nn.ModuleList(MLP(embed_dim, embed_dim, embed_dim // 4, 3) for _ in range(4))
        )
        self.score_head = DecoderScoreHead(
            MLP(embed_dim, embed_dim, 4, 3),
            use_stability_score=True,
            mask_threshold=0.0,
        )
        self.num_mask_tokens = 4
        self.embed_h = 1
        self.embed_w = 1


class QuantizeDecoderSurfaceScopeTest(unittest.TestCase):
    def test_transformer_scope_keeps_non_transformer_modules(self) -> None:
        model = _DummyDecoderSurface()
        original_label_embedding = model.label_embedding
        original_output_upscaling = model.output_upscaling
        original_output_hypernetworks = model.output_hypernetworks
        original_score_head = model.score_head

        quantized = quantize_decoder_surface(model, make_qconfig(), scope="transformer")

        self.assertIs(quantized, model)
        self.assertIsInstance(quantized.transformer, QuantEdgeTwoWayTransformer)
        self.assertIs(quantized.label_embedding, original_label_embedding)
        self.assertIs(quantized.output_upscaling, original_output_upscaling)
        self.assertIs(quantized.output_hypernetworks, original_output_hypernetworks)
        self.assertIs(quantized.score_head, original_score_head)

    def test_full_scope_wraps_npu_safe_decoder_stages(self) -> None:
        model = _DummyDecoderSurface()

        quantized = quantize_decoder_surface(model, make_qconfig(), scope="full")

        self.assertIsInstance(quantized, QuantFullDecoderSurface)
        self.assertIsInstance(quantized.label_embedding, QuantDecoderLabelEmbedding)
        self.assertIsInstance(quantized.transformer, QuantEdgeTwoWayTransformer)
        self.assertIsInstance(quantized.output_upscaling, QuantOutputUpscaling)
        self.assertIsInstance(quantized.output_hypernetworks, QuantDecoderHypernetworkStack)
        self.assertIsInstance(quantized.score_head, QuantDecoderScoreHead)

    def test_full_scope_can_keep_selected_tail_modules_in_fp32(self) -> None:
        model = _DummyDecoderSurface()

        quantized = quantize_decoder_surface(
            model,
            make_qconfig(),
            scope="full",
            full_exclude=("output_upscaling", "output_hypernetworks", "score_head"),
        )

        self.assertIsInstance(quantized, QuantFullDecoderSurface)
        self.assertIsInstance(quantized.label_embedding, QuantDecoderLabelEmbedding)
        self.assertIsInstance(quantized.transformer, QuantEdgeTwoWayTransformer)
        self.assertIs(quantized.output_upscaling, model.output_upscaling)
        self.assertIs(quantized.output_hypernetworks, model.output_hypernetworks)
        self.assertIs(quantized.score_head, model.score_head)

    def test_full_scope_can_disable_hypernetwork_output_quantization(self) -> None:
        model = _DummyDecoderSurface()

        quantized = quantize_decoder_surface(
            model,
            make_qconfig(),
            scope="full",
            quantize_hypernetwork_output=False,
        )

        self.assertIsInstance(quantized, QuantFullDecoderSurface)
        self.assertFalse(quantized.output_hypernetworks.mlps[0].quantize_output)
        self.assertIsInstance(quantized.output_hypernetworks.mlps[0].output_post_act_fake_quantize, nn.Identity)

    def test_full_scope_can_quantize_mask_projection_inputs_asymmetrically(self) -> None:
        model = _DummyDecoderSurface()

        quantized = quantize_decoder_surface(
            model,
            make_qconfig(),
            scope="full",
            quantize_mask_projection_hyper_input=False,
            quantize_mask_projection_upscaled_input=True,
        )

        self.assertIsInstance(quantized, QuantFullDecoderSurface)
        self.assertFalse(quantized.mask_projection.matmul.quantize_a_input)
        self.assertTrue(quantized.mask_projection.matmul.quantize_b_input)

    def test_full_scope_can_use_int16_for_hyper_in_quantizers(self) -> None:
        model = _DummyDecoderSurface()

        quantized = quantize_decoder_surface(
            model,
            make_qconfig(),
            scope="full",
            hypernetwork_output_bit=16,
            mask_projection_hyper_input_bit=16,
        )

        self.assertIsInstance(quantized, QuantFullDecoderSurface)
        self.assertEqual(quantized.output_hypernetworks.mlps[0].output_post_act_fake_quantize.bit, 16)
        self.assertEqual(quantized.mask_projection.matmul.a_layer_pre_act_fake_quantize.bit, 16)
        self.assertEqual(quantized.mask_projection.matmul.b_layer_pre_act_fake_quantize.bit, 8)

    def test_full_scope_rejects_unknown_exclusions(self) -> None:
        model = _DummyDecoderSurface()

        with self.assertRaisesRegex(ValueError, "Unsupported full decoder exclusions"):
            quantize_decoder_surface(model, make_qconfig(), scope="full", full_exclude=("bad_stage",))

    def test_full_scope_enables_label_embedding_act_quantizers_for_calibration(self) -> None:
        model = _DummyDecoderSurface()
        quantized = quantize_decoder_surface(model, make_qconfig(), scope="full")

        enable_calibration_woquantization(quantized, quantizer_type="act_fake_quant")

        self.assertEqual(quantized.label_embedding.input_act_fake_quantize.observer_enabled, 1)
        self.assertEqual(quantized.label_embedding.input_act_fake_quantize.fake_quant_enabled, 0)
        self.assertEqual(quantized.label_embedding.output_act_fake_quantize.observer_enabled, 1)
        self.assertEqual(quantized.label_embedding.output_act_fake_quantize.fake_quant_enabled, 0)


if __name__ == "__main__":
    unittest.main()
