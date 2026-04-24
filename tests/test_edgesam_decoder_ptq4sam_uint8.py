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


if __name__ == "__main__":
    unittest.main()
