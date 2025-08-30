import numpy as np
import nnabla as nn
import nnabla.function as F

def embed_tokens(x, sd):
    w = sd[f"model.embed_tokens.weight"]
    return F.embed(x, w)

def repeat_kv(x, nr):
    if nr == 1:
        return x
    B, H, L, D = x.shape
    x = F.reshape(x, (B, H, 1, L, D))
    x = F.repeat(x, repeats=nr, axis=2)
    x = F.reshape(x, (B, H * nr, L, D))
    return x

def attention(
    sd,
    lc,
    hs,
    head_dim,
    cache,
    am,
    scale,
    dropout_p,
    cos,
    sin
):
    input_shape = hs.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)
    q = sd[f"model.layers.{lc}.self_attn.q_proj.weight"]
    k = sd[f"model.layers.{lc}.self_attn.k_proj.weight"]
    v = sd[f"model.layers.{lc}.self_attn.v_proj.weight"]
    q = F.affine(hs, q, sd[f"model.layers.{lc}.self_attn.q_proj.bias"])
    k = F.affine(hs, k, sd[f"model.layers.{lc}.self_attn.k_proj.bias"])
    v = F.affine(hs, v, sd[f"model.layers.{lc}.self_attn.v_proj.bias"])
    q = F.reshape(q, hidden_shape)
    k = F.reshape(k, hidden_shape)
    v = F.reshape(v, hidden_shape)
    k = F.transpose(k, (0, 2, 1, 3))
    v = F.transpose(v, (0, 2, 1, 3))
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k, v = cache.kv_states.add(k, v)
    nr = q.shape[1] // k.shape[1]
    k = repeat_kv(k, nr)
    v = repeat_kv(v, nr)
    if am is not None and len(am.shape) == 4:
        seq_len = k.shape[-2]
        am = F.slice(
            am,
            start=(0, 0, 0, 0),
            stop=(am.shape[0],
            am.shape[1],
            am.shape[2],
            seq_len),
            step=(1, 1, 1, 1)
        )
    origin_dtype = q.dtype
    def maybe_upcast(x):
        if x.dtype in (np.float16, np.bfloat16):
            return F.cast(x, np.float32)
        return x
    q = maybe_upcast(q)  
    k = maybe_upcast(k)  
    v = maybe_upcast(v)  
    if scale is None:
        scale = 1.0 / (k.shape[-1] ** 0.5)
    attn_weights = F.batch_matmul(q, k, transpose_a=False, transpose_b=True) * scale
    def causal_mask(q_len, k_len):
        mask = np.tril(np.ones((q_len, k_len), dtype=np.bool_))
        return nn.Variable.from_numpy_array(mask)
    q_len, k_len = attn_weights.shape[-2], attn_weights.shape[-1]
    mask = causal_mask(q_len, k_len)
    mask = F.reshape(mask, (1,1,q_len,k_len))
    attn_weights = F.where(mask, attn_weights, F.constant(-1e9, attn_weights.shape))
    if am is not None:
        if am.dtype == np.bool_:
            attn_weights = F.where(am, attn_weights, F.constant(-1e9, attn_weights.shape))
        else:
            attn_weights = attn_weights + am
    attn_probs = F.softmax(attn_weights, axis=-1)
    if dropout_p > 0.0:
        attn_probs = F.dropout(attn_probs, p=dropout_p)
    out = F.batch_matmul(attn_probs, v)
    out = F.cast(out, origin_dtype)
    out = F.transpose(out, (0, 2, 1, 3))
    out = F.reshape(out, (*input_shape, -1))
    attn_output = F.affine(out, sd[f"model.layers.{lc}.self_attn.o_proj.weight"])
    return attn_output

def _slice_last(x, start, stop):
    starts = (0,) * (x.ndim - 1) + (start,)
    stops = (0,) * (x.ndim - 1) + (stop,)
    return F.slice(x, start=starts, stop=stops)

def rotate_half(x):
    hd = x.shape[-1]
    half = hd // 2
    x1 = _slice_last(x, 0, half)
    x2 = _slice_last(x, half, hd)
    return F.concatenate(F.neg(x2), x1, axis=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, expand_dim=1):
    cos = F.expand_dims(cos, axis=expand_dim)
    sin = F.expand_dims(sin, axis=expand_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rms_norm(w, hs, eps):
    input_dtype = hs.dtype #入力
    variance = hs.pow(2).mean(-1, keepdim=True)
    hs = hs * F.rsqrt(variance + eps)
    return w * F.cast(hs, input_dtype)

def mlp(sd, lc, x):
    #gate, up, down, act
    gate_w = sd[f"model.layers.{lc}.mlp.gate_proj.weight"]
    up_w = sd[f"model.layers.{lc}.mlp.up_proj.weight"]
    down_w = sd[f"model.layers.{lc}.mlp.down_proj.weight"]
    gate = F.affine(x, gate_w)
    act = F.mul2(gate, F.sigmoid(gate))
    up = F.affine(x, up_w)
    act_up = F.mul2(act, up)
    down = F.affine(act_up, down_w)
    return down

def decoder_layer(sd, max_layer, position_ids, hs, am):
    cos, sin = rotary_emb(hs, position_ids)
    for i in range(max_layer):
        re = hs
        hs = rms_norm(sd[f"model.layers.{i}.input_layernorm.weight"], hs)
        hs = attention(
            sd=sd,
            layer_count=i,
            hs=hs,
            head_dim,
            cache,
            am,
            scale,
            dropout_p,
            cos,
            sin
        )
        hs = re + hs

        re = hs
        hs = rms_norm(sd[f"model.layers.{i}.post_attention_layernorm.weight"], hs)
        hs = mlp(sd, i, hs)
        hs = re + hs
    return hs

def rotary_emb(hs, position_ids, config):
    B, L, H = hs.shape
    if position_ids is None:
        pos_ids = np.arange(L, dtype=np.int32)
        pos_ids = np.tile(pos_ids, (B, 1))
        position_ids = nn.Variable.from_numpy_array(pos_ids)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(head_dim * partial_rotary_factor)

    base = getattr(config, "rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
    inv_freq = inv_freq.astype(np.float32)
    inv_freq_nn = nn.Variable.from_numpy_array(inv_freq)
    attention_factor = 1.0

    pos = F.reshape(position_ids, (position_ids.shape[0], 1, position_ids.shape[1]))
    freqs = F.batch_matmul(pos, F.reshape(inv_freq_nn, (1, inv_freq_nn.shape[0], 1)))
    freqs = F.transpose(freqs, (0, 2, 1))
    emb = F.concatenate(freqs, freqs, axis=-1)

    cos = F.cos(emb) * attention_factor
    sin = F.sin(emb) * attention_factor

    return cos, sin

def lm_head(sd, x):
    w = sd[f"model.embed_tokens.weight"]
    return F.affine(x, w)
