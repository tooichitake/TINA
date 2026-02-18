"""
TINA: Triple-encoder Interactive Neighbourhood Attention
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        return self._denormalize(x)

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class NeighborhoodAttention1D(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, kernel_size=7, n_heads=8):
        super(NeighborhoodAttention1D, self).__init__()
        self.scale = scale
        self.dropout_p = attention_dropout
        self.kernel_size = kernel_size
        self.half_K = kernel_size // 2

        self.rpb = nn.Parameter(torch.zeros(n_heads, 2 * kernel_size - 1))
        nn.init.trunc_normal_(self.rpb, std=0.02)

        self.register_buffer("_neighbor_idx", None, persistent=False)
        self.register_buffer("_rpb_idx", None, persistent=False)
        self.register_buffer("_valid_mask", None, persistent=False)

    def _build_indices(self, L, device):
        if self._neighbor_idx is not None and self._neighbor_idx.shape[0] == L:
            return
        K = self.kernel_size
        half_K = self.half_K

        pos = torch.arange(L, device=device)
        window_offset = torch.arange(K, device=device) - half_K

        ideal_neighbor = pos.unsqueeze(1) + window_offset
        self._neighbor_idx = ideal_neighbor.clamp(0, L - 1)
        self._rpb_idx = (self._neighbor_idx - pos.unsqueeze(1)) + (K - 1)
        self._valid_mask = (ideal_neighbor >= 0) & (ideal_neighbor < L)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        scale = self.scale or (E ** -0.5)

        self._build_indices(L, queries.device)

        q = queries.permute(0, 2, 1, 3)
        k = keys.permute(0, 2, 1, 3)
        v = values.permute(0, 2, 1, 3)

        k_nb = k[:, :, self._neighbor_idx]
        v_nb = v[:, :, self._neighbor_idx]

        attn = torch.einsum("bhle,bhlke->bhlk", q, k_nb) * scale
        attn = attn + self.rpb[:, self._rpb_idx]
        attn = attn.masked_fill(~self._valid_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn_weights = attn.clone()

        if self.training and self.dropout_p > 0:
            attn = F.dropout(attn, p=self.dropout_p)

        out = torch.einsum("bhlk,bhlke->bhle", attn, v_nb)
        return out.permute(0, 2, 1, 3).contiguous(), attn_weights


class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout_p = attention_dropout

    def forward(self, queries, keys, values):
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scale = self.scale or (queries.size(-1) ** -0.5)
        attn_weights = torch.matmul(queries, keys.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.training and self.dropout_p > 0:
            attn_out = F.dropout(attn_weights, p=self.dropout_p)
        else:
            attn_out = attn_weights

        out = torch.matmul(attn_out, values)

        return out.transpose(1, 2).contiguous(), attn_weights


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(AttentionLayer, self).__init__()
        self.inner_attention = attention
        self.n_heads = n_heads
        d_k = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_k * n_heads)
        self.key_projection = nn.Linear(d_model, d_k * n_heads)
        self.value_projection = nn.Linear(d_model, d_k * n_heads)
        self.out_projection = nn.Linear(d_k * n_heads, d_model)

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn_weights = self.inner_attention(queries, keys, values)
        return self.out_projection(out.view(B, L, -1)), attn_weights


class SwiGLU(nn.Module):
    """SwiGLU FFN - Swish-Gated Linear Unit."""
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class SelfEncoderLayer(nn.Module):
    def __init__(
        self,
        self_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="gelu",
        norm_cls=nn.RMSNorm,
    ):
        super(SelfEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.use_swiglu = activation == "swiglu"
        if self.use_swiglu:
            self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
        else:
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.activation = F.relu if activation == "relu" else F.gelu
        self.norm1 = norm_cls(d_model)
        self.norm2 = norm_cls(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.self_attention(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        if self.use_swiglu:
            x = x + self.ffn(self.norm2(x))
        else:
            y = self.dropout(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x))))))
            x = x + y
        return x, attn_weights


class CrossEncoderLayer(nn.Module):
    def __init__(
        self,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="gelu",
        norm_cls=nn.RMSNorm,
    ):
        super(CrossEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.cross_attention = cross_attention
        self.use_swiglu = activation == "swiglu"
        if self.use_swiglu:
            self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
        else:
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.activation = F.relu if activation == "relu" else F.gelu
        self.norm1 = norm_cls(d_model)
        self.norm2 = norm_cls(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.cross_attention(x_norm, cross, cross)
        x = x + self.dropout(attn_out)
        if self.use_swiglu:
            x = x + self.ffn(self.norm2(x))
        else:
            y = self.dropout(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x))))))
            x = x + y
        return x, attn_weights


class SelfEncoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(SelfEncoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x):
        attns = []
        for layer in self.layers:
            x, attn = layer(x)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class CrossEncoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(CrossEncoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, cross)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    """Embedding for TINA"""
    def __init__(
        self,
        d_model,
        patch_len,
        dropout,
        n_vars,
        use_time_features=False,
        freq='h',
        use_variable_embedding=True,
    ):
        super(EnEmbedding, self).__init__()
        self.patch_len = patch_len
        self.use_time_features = use_time_features
        self.use_variable_embedding = use_variable_embedding

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.variable_embedding = (
            nn.Embedding(n_vars, d_model) if use_variable_embedding else None
        )

        if use_time_features:
            # Use dynamic time feature dimension based on frequency (same as TimeFeatureEmbedding)
            freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
            time_feature_dim = freq_map.get(freq, 4)  # Default to 4 if freq not found
            self.time_embedding = nn.Linear(time_feature_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        B, n_vars, L = x.shape

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = x.reshape(B * n_vars, x.shape[2], x.shape[3])
        patch_num = x.shape[1]

        x_embed = self.value_embedding(x) + self.position_embedding(x)

        if self.use_variable_embedding and self.variable_embedding is not None:
            var_ids = torch.arange(n_vars, device=x.device)
            var_embed = self.variable_embedding(var_ids)
            var_embed = var_embed.unsqueeze(0).expand(B, -1, -1)
            var_embed = var_embed.reshape(B * n_vars, 1, -1)
            x_embed = x_embed + var_embed

        if self.use_time_features and x_mark is not None:
            x_mark_patched = x_mark.unfold(
                dimension=1, size=self.patch_len, step=self.patch_len
            )
            x_mark_patched = x_mark_patched[:, :patch_num, :, self.patch_len // 2]
            x_mark_patched = x_mark_patched.unsqueeze(1).expand(-1, n_vars, -1, -1)
            x_mark_patched = x_mark_patched.reshape(B * n_vars, patch_num, -1)

            time_embed = self.time_embedding(x_mark_patched)
            x_embed = x_embed + time_embed

        x = x_embed.reshape(B, n_vars, patch_num, -1)
        x = x.reshape(B * n_vars, patch_num, -1)
        return self.dropout(x), n_vars, patch_num


class Model(nn.Module):
    """TINA: Triple-encoder Interactive Neighbourhood Attention"""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.patch_num = configs.seq_len // configs.patch_len
        self.output_attention = getattr(configs, "output_attention", False)

        variant_alias = {
            "full": "full",
            "base": "full",
            "na_full_attention": "na_full_attention",
            "na_to_full_attention": "na_full_attention",
            "wo_cross_encoder": "wo_cross_encoder",
            "wo_variable_embedding": "wo_variable_embedding",
            "wo_time_embedding": "wo_time_embedding",
            "layernorm": "layernorm",
            "rmsnorm_to_layernorm": "layernorm",
            "wo_revin": "wo_revin",
        }
        raw_variant = getattr(configs, "ablation_variant", "full")
        self.ablation_variant = variant_alias.get(str(raw_variant).lower(), "full")

        self.use_full_attention_for_self = self.ablation_variant == "na_full_attention"
        self.disable_cross_encoder = self.ablation_variant == "wo_cross_encoder"
        self.disable_variable_embedding = self.ablation_variant == "wo_variable_embedding"
        self.disable_time_embedding = self.ablation_variant == "wo_time_embedding"
        self.use_layernorm = self.ablation_variant == "layernorm"

        self.use_revin = getattr(configs, "use_revin", True)
        if self.ablation_variant == "wo_revin":
            self.use_revin = False
        if self.use_revin:
            self.revin_layer = RevIN(
                configs.enc_in, affine=getattr(configs, "use_revin_affine", True)
            )

        self.na_kernel_size = getattr(configs, "na_kernel_size", 7)
        activation = getattr(configs, "activation", "gelu")
        use_time_features = (
            getattr(configs, "use_time_features", True) and not self.disable_time_embedding
        )
        norm_cls = nn.LayerNorm if self.use_layernorm else nn.RMSNorm

        self.endo_embedding = EnEmbedding(
            configs.d_model,
            self.patch_len,
            configs.dropout,
            n_vars=configs.enc_in,
            use_time_features=use_time_features,
            freq=configs.freq,
            use_variable_embedding=not self.disable_variable_embedding,
        )

        self.exo_embedding = EnEmbedding(
            configs.d_model,
            self.patch_len,
            configs.dropout,
            n_vars=configs.enc_in,
            use_time_features=use_time_features,
            freq=configs.freq,
            use_variable_embedding=not self.disable_variable_embedding,
        )

        self.exo_encoder = SelfEncoder(
            [
                SelfEncoderLayer(
                    AttentionLayer(
                        FullAttention(attention_dropout=configs.dropout)
                        if self.use_full_attention_for_self
                        else NeighborhoodAttention1D(
                            attention_dropout=configs.dropout,
                            kernel_size=self.na_kernel_size,
                            n_heads=configs.n_heads,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=activation,
                    norm_cls=norm_cls,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=norm_cls(configs.d_model),
        )

        self.endo_encoder = SelfEncoder(
            [
                SelfEncoderLayer(
                    AttentionLayer(
                        FullAttention(attention_dropout=configs.dropout)
                        if self.use_full_attention_for_self
                        else NeighborhoodAttention1D(
                            attention_dropout=configs.dropout,
                            kernel_size=self.na_kernel_size,
                            n_heads=configs.n_heads,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=activation,
                    norm_cls=norm_cls,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=norm_cls(configs.d_model),
        )

        self.cross_encoder = CrossEncoder(
            [
                CrossEncoderLayer(
                    AttentionLayer(
                        FullAttention(attention_dropout=configs.dropout),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=activation,
                    norm_cls=norm_cls,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=norm_cls(configs.d_model),
        )

        self.head_nf = configs.d_model * self.patch_num
        self.head = FlattenHead(
            self.head_nf, configs.pred_len, head_dropout=configs.dropout
        )

        self._init_weights()

    def _init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.RMSNorm, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_revin:
            x_enc = self.revin_layer(x_enc, "norm")

        batch_size = x_enc.shape[0]

        endo_embed, n_vars, _ = self.endo_embedding(
            x_enc[:, :, -1:].permute(0, 2, 1), x_mark_enc
        )

        exo_embed, _, _ = self.exo_embedding(
            x_enc[:, :, :-1].permute(0, 2, 1), x_mark_enc
        )

        exo_out, exo_attns = self.exo_encoder(exo_embed)
        endo_out, endo_attns = self.endo_encoder(endo_embed)

        exo_out_full = exo_out.reshape(
            batch_size, -1, exo_out.shape[-1]
        )

        if self.disable_cross_encoder:
            cross_attns = []
        else:
            endo_out, cross_attns = self.cross_encoder(endo_out, exo_out_full)
        endo_out = endo_out.reshape(-1, n_vars, endo_out.shape[-2], endo_out.shape[-1])
        endo_out = endo_out.permute(0, 1, 3, 2)

        dec_out = self.head(endo_out)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_revin:
            dec_out = self.revin_layer(dec_out, "denorm")

        attns = {
            "exo_attns": exo_attns,
            "endo_attns": endo_attns,
            "cross_attns": cross_attns,
        }
        return dec_out, attns

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        M mode: All variables as both endogenous and exogenous.
        Simplified approach - predict all variables in one forward pass.
        """
        if self.use_revin:
            x_enc = self.revin_layer(x_enc, "norm")

        batch_size = x_enc.shape[0]

        # All variables as endogenous (targets to predict)
        endo_embed, n_vars, _ = self.endo_embedding(
            x_enc.permute(0, 2, 1), x_mark_enc
        )

        # All variables also as exogenous (context)
        exo_embed, _, _ = self.exo_embedding(
            x_enc.permute(0, 2, 1), x_mark_enc
        )

        exo_out, exo_attns = self.exo_encoder(exo_embed)
        endo_out, endo_attns = self.endo_encoder(endo_embed)

        # Reshape to enable cross-variable interaction
        exo_out_full = exo_out.reshape(batch_size, -1, exo_out.shape[-1])
        endo_out_full = endo_out.reshape(batch_size, -1, endo_out.shape[-1])

        if self.disable_cross_encoder:
            endo_out = endo_out_full
            cross_attns = []
        else:
            endo_out, cross_attns = self.cross_encoder(endo_out_full, exo_out_full)
        endo_out = endo_out.reshape(batch_size, n_vars, -1, endo_out.shape[-1])
        endo_out = endo_out.permute(0, 1, 3, 2)

        dec_out = self.head(endo_out)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_revin:
            dec_out = self.revin_layer(dec_out, "denorm")

        attns = {
            "exo_attns": exo_attns,
            "endo_attns": endo_attns,
            "cross_attns": cross_attns,
        }
        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            if self.features == "M":
                dec_out, attns = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:
                dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if self.output_attention:
                return dec_out[:, -self.pred_len :, :], attns
            return dec_out[:, -self.pred_len :, :]
        return None
