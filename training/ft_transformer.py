"""
FT-Transformer for Fear-Free Night Navigator
==============================================
A Feature Tokenizer + Transformer architecture for tabular regression,
implementing the approach from:

    Gorishniy, Y., Rubachev, I., Krotov, V., & Babenko, A. (2021).
    *Revisiting Deep Learning Models for Tabular Data.*
    NeurIPS 2021.   arXiv:2106.11959

Key insight from the paper: treating each numerical feature as a
"token" (via learned linear projections) and feeding the token
sequence through a standard Transformer encoder achieves SOTA
results on tabular benchmarks, outperforming gradient-boosted trees
on medium-to-large datasets with complex feature interactions.

Architecture overview:
    ┌──────────────────────────────────────────────────┐
    │  Input:  x ∈ ℝᵈ  (d preprocessed features)      │
    │                                                    │
    │  1. Feature Tokenizer                              │
    │     Each feature xᵢ → eᵢ = Wᵢ·xᵢ + bᵢ  ∈ ℝᵈᵐ   │
    │     (d separate linear projections)                │
    │                                                    │
    │  2. [CLS] token prepended (learnable)              │
    │     Sequence: [CLS, e₁, e₂, …, eₐ]               │
    │                                                    │
    │  3. Transformer Encoder (L layers)                 │
    │     Multi-head self-attention + FFN + LayerNorm    │
    │     + PreNorm residuals + dropout                  │
    │                                                    │
    │  4. Regression Head                                │
    │     CLS output → Linear → ReLU → Linear → σ(·)×100│
    └──────────────────────────────────────────────────┘

Additionally incorporates ideas from:
  - Attention-based feature importance (interpretability for judges)
  - Stochastic Depth (Huang et al., 2016) for regularisation
  - GEGLU activation (Shazeer, 2020) in the FFN blocks

Author : Fear-Free Night Navigator Team
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
#  BUILDING BLOCKS
# =====================================================================

class GEGLU(nn.Module):
    """
    Gated Exponential GLU activation (Shazeer, 2020).

    FFN(x) = (xW₁ + b₁) ⊙ GELU(xW₂ + b₂)

    Outperforms standard ReLU in Transformer FFN blocks on language
    *and* tabular tasks (confirmed in Gorishniy et al., Table 3).
    Doubles the intermediate weight count, but halves the activation
    dimension, yielding comparable FLOPs.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class PreNormResidual(nn.Module):
    """
    Pre-LayerNorm residual block (Xiong et al., 2020).

    y = x + Dropout(sublayer(LayerNorm(x)))

    Pre-norm (normalise *before* the sublayer) stabilises training
    at depth and allows higher learning rates compared to post-norm.
    """

    def __init__(self, d_model: int, sublayer: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x + self.dropout(self.sublayer(self.norm(x), **kwargs))


class MultiHeadSelfAttention(nn.Module):
    """
    Standard scaled dot-product multi-head self-attention.

    Attention(Q,K,V) = softmax(QKᵀ / √dₖ) V

    Also exposes the raw attention weights for feature importance
    analysis — the CLS token's attention over feature tokens shows
    which features the model attends to for each prediction.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # Store attention weights for interpretability (populated in forward)
        self._attn_weights = None

    def forward(self, x: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        B, S, _ = x.shape

        # Project to Q, K, V simultaneously
        qkv = self.W_qkv(x).reshape(B, S, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)         # (3, B, H, S, d_k)
        q, k, v = qkv.unbind(dim=0)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale   # (B, H, S, S)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        if store_attn:
            # Average across heads; keep CLS→feature attention row
            self._attn_weights = attn.mean(dim=1)[:, 0, 1:]  # (B, d_features)

        out = torch.matmul(attn, v)                            # (B, H, S, d_k)
        out = out.transpose(1, 2).contiguous().reshape(B, S, -1)
        return self.W_out(out)


class TransformerBlock(nn.Module):
    """
    One Transformer encoder layer:
        PreNorm → MHSA → Residual → PreNorm → FFN(GEGLU) → Residual

    Uses stochastic depth (Huang et al., 2016) at training time:
    each block is randomly skipped with probability `drop_path`,
    acting as an implicit ensemble of sub-networks and improving
    generalisation on medium-sized tabular datasets.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        dropout: float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.attn_block = PreNormResidual(
            d_model,
            MultiHeadSelfAttention(d_model, n_heads, dropout),
            dropout,
        )

        # GEGLU doubles the input dim, so we project to 2 × d_ffn
        ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.ffn_block = PreNormResidual(d_model, ffn, dropout)

        self.drop_path_prob = drop_path

    def forward(self, x: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        # Stochastic depth: skip this block with probability p during training
        if self.training and self.drop_path_prob > 0:
            if torch.rand(1).item() < self.drop_path_prob:
                return x

        x = self.attn_block(x, store_attn=store_attn)
        x = self.ffn_block(x)
        return x


# =====================================================================
#  FEATURE TOKENIZER
# =====================================================================

class FeatureTokenizer(nn.Module):
    """
    Convert each scalar feature into a d_model-dimensional embedding.

    For each feature i:   token_i = W_i · x_i + b_i,   W_i ∈ ℝᵈᵐ

    This is the key architectural contribution from Gorishniy et al.:
    by giving each feature its own projection layer, the model can
    learn feature-specific representations that are commensurate in
    the shared embedding space, enabling meaningful cross-feature
    attention.

    Also prepends a learnable [CLS] token that aggregates information
    from all features through attention and is used for the final
    prediction (analogous to BERT's [CLS] for classification).
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # One linear projection per feature (implements W_i · x_i + b_i)
        # Using weight shape (n_features, d_model) and bias (n_features, d_model)
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))

        self._init_weights()

    def _init_weights(self):
        # Kaiming uniform initialisation (He et al., 2015) for the per-
        # feature weights ensures the variance of activation magnitudes
        # is preserved through the tokenizer.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Fan-in-based bound for bias (matches nn.Linear default)
        fan_in = self.d_model
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, n_features)

        Returns
        -------
        tokens : (B, n_features + 1, d_model)   — [CLS] prepended
        """
        # x[:, i] * W[i] + b[i]  for each feature i, vectorised
        # x shape: (B, n_features) → (B, n_features, 1)
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        # tokens shape: (B, n_features, d_model)

        # Prepend [CLS]
        cls = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, d_model)
        return torch.cat([cls, tokens], dim=1)           # (B, n_features+1, d_model)


# =====================================================================
#  FT-TRANSFORMER  (Full Model)
# =====================================================================

class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer for tabular regression.

    Hyperparameters follow the recommendations from Gorishniy et al.
    (2021, §4.2) scaled for our 100K-row dataset:

        d_model  = 192    (token embedding dimension)
        n_heads  = 8      (attention heads; d_k = 24)
        n_layers = 4      (Transformer depth)
        d_ffn    = 512    (FFN intermediate dim before GEGLU split)
        dropout  = 0.15   (attention + FFN dropout)

    The regression head outputs a single value in [0, 100] via
    sigmoid × 100, ensuring the prediction is always within the
    valid target range (bounded output activation).
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 192,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ffn: int = 512,
        dropout: float = 0.15,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features

        # --- Feature Tokenizer -------------------------------------------
        self.tokenizer = FeatureTokenizer(n_features, d_model)

        # --- Transformer Encoder Stack ------------------------------------
        # Linearly increasing stochastic depth rate (0 at bottom, drop_path
        # at top) — deeper layers are dropped more often, consistent with
        # the original stochastic depth paper.
        dp_rates = [drop_path * i / max(n_layers - 1, 1) for i in range(n_layers)]

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ffn, dropout, dp_rates[i])
            for i in range(n_layers)
        ])

        # --- Final LayerNorm on CLS output --------------------------------
        self.final_norm = nn.LayerNorm(d_model)

        # --- Regression Head -----------------------------------------------
        # Two-layer head with ReLU, projecting CLS representation → score.
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, n_features)   float tensor of preprocessed features.
        return_attention : bool
            If True, also return CLS attention weights for interpretability.

        Returns
        -------
        score : (B, 1)  predicted safety score in [0, 100].
        attn  : (B, n_features) attention weights (only if requested).
        """
        # 1. Tokenize (each feature → d_model-dim embedding)
        tokens = self.tokenizer(x)       # (B, n_features+1, d_model)

        # 2. Pass through Transformer encoder stack
        store = return_attention
        for block in self.transformer_blocks:
            tokens = block(tokens, store_attn=store)

        # 3. Extract [CLS] representation
        cls_repr = self.final_norm(tokens[:, 0])   # (B, d_model)

        # 4. Regression head → bounded output via sigmoid × 100
        raw = self.head(cls_repr)                   # (B, 1)
        score = torch.sigmoid(raw) * 100.0          # (B, 1) ∈ [0, 100]

        if return_attention:
            last_block = self.transformer_blocks[-1]
            attn = last_block.attn_block.sublayer._attn_weights  # (B, n_features)
            return score, attn

        return score

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =====================================================================
#  FACTORY FUNCTION
# =====================================================================

def build_ft_transformer(n_features: int, config: str = "base") -> FTTransformer:
    """
    Build an FT-Transformer with preset configurations.

    Configs:
        'small' :  d=128, L=3, H=4   (~  0.5M params, fast iteration)
        'base'  :  d=192, L=4, H=8   (~  1.5M params, hackathon default)
        'large' :  d=256, L=6, H=8   (~  3.5M params, if GPU allows)
    """
    presets = {
        "small": dict(d_model=128, n_heads=4,  n_layers=3, d_ffn=256,
                       dropout=0.1,  drop_path=0.05),
        "base":  dict(d_model=192, n_heads=8,  n_layers=4, d_ffn=512,
                       dropout=0.15, drop_path=0.1),
        "large": dict(d_model=256, n_heads=8,  n_layers=6, d_ffn=768,
                       dropout=0.2,  drop_path=0.15),
    }

    if config not in presets:
        raise ValueError(f"Unknown config '{config}'. Choose from {list(presets)}")

    model = FTTransformer(n_features=n_features, **presets[config])
    print(f"FT-Transformer ({config}): {model.count_parameters():,} trainable params")
    return model


# =====================================================================
#  QUICK SMOKE TEST
# =====================================================================
if __name__ == "__main__":
    N_FEAT = 30  # example feature count from preprocessing
    model = build_ft_transformer(N_FEAT, config="base")

    # Dummy forward pass
    dummy = torch.randn(8, N_FEAT)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}, range [{out.min().item():.2f}, {out.max().item():.2f}]")

    # With attention
    out2, attn = model(dummy, return_attention=True)
    print(f"Attention shape: {attn.shape}  (batch × n_features)")
