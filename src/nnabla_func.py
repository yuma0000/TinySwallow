nnabla-only implementation of Attention + RoPE + RMSNorm + MLP + Decoder Layer

依存: nnabla (import as nn), nnabla.functions as F

torch は使いません。shape: (B, T, D)

import math import numpy as np import nnabla as nn import nnabla.functions as F

-----------------------------

Utils

-----------------------------

def _get(sd, names): """state_dict から候補名の最初に見つかったものを返す。無ければ KeyError。 names: ["model.layers.0.self_attn.q_proj.weight", "layers.0.self_attn.q_proj.weight"] など """ for n in names: if n in sd: return sd[n] raise KeyError(f"Missing keys: {names}")

def split_heads(x, num_heads): """(B,T,D) -> (B,H,T,Hd), Hd = D/H""" b, t, d = x.shape assert d % num_heads == 0, "hidden dim must be divisible by num_heads" hd = d // num_heads x = F.reshape(x, (b, t, num_heads, hd)) # (B,T,H,Hd) -> (B,H,T,Hd) x = F.transpose(x, (0, 2, 1, 3)) return x

def merge_heads(x): """(B,H,T,Hd) -> (B,T,D)""" b, h, t, hd = x.shape x = F.transpose(x, (0, 2, 1, 3))  # (B,T,H,Hd) x = F.reshape(x, (b, t, h * hd)) return x

def repeat_kv(x, repeat): """GQA 用: (B,HK,T,Hd) を head 次元に沿って繰り返す -> (B,HK*repeat,T,Hd)""" if repeat == 1: return x # tile: [B, H, T, Hd] reps = (1, repeat, 1, 1) return F.tile(x, reps)

-----------------------------

Rotary Position Embedding (RoPE)

-----------------------------

def rotate_half(x): # x: (..., Hd) hd = x.shape[-1] half = hd // 2 x1 = F.slice(x, start=(0,)* (x.ndim - 1) + (0,), stop=(0,)* (x.ndim - 1) + (half,), step=None) x2 = F.slice(x, start=(0,)* (x.ndim - 1) + (half,), stop=(0,)* (x.ndim - 1) + (hd,), step=None) return F.concatenate(F.neg(x2), x1, axis=-1)

def apply_rotary_pos_emb(q, k, cos, sin): """ q,k: (B,H,T,Hd) cos,sin: (1,1,T,Hd) もしくは (B,H,T,Hd) に broadcast 可能 """ cos = F.broadcast_to(cos, q.shape) sin = F.broadcast_to(sin, q.shape) q_embed = q * cos + rotate_half(q) * sin k_embed = k * cos + rotate_half(k) * sin return q_embed, k_embed

def rotary_emb(seq_len, head_dim, base=10000.0, scaling=1.0, device=None): """numpy で cos/sin を作り、nn.Variable にする。 戻り: cos, sin 形状 (1,1,T,Hd) """ hd = head_dim # inv_freq shape: (Hd/2,) inv_freq = 1.0 / (base ** (np.arange(0, hd, 2, dtype=np.float32) / hd)) pos = np.arange(seq_len, dtype=np.float32)  # (T,) freqs = np.einsum('t,d->td', pos, inv_freq)  # (T,Hd/2) emb = np.concatenate([freqs, freqs], axis=-1)  # (T,Hd) cos = np.cos(emb, dtype=np.float32) * scaling sin = np.sin(emb, dtype=np.float32) * scaling cos = cos.reshape(1, 1, seq_len, hd) sin = sin.reshape(1, 1, seq_len, hd) cos_v = nn.Variable.from_numpy_array(cos) sin_v = nn.Variable.from_numpy_array(sin) return cos_v, sin_v

-----------------------------

RMSNorm

-----------------------------

def rms_norm(weight, x, eps=1e-6): # weight: (D,), x: (B,T,D) var = F.mean(F.pow_scalar(x, 2.0), axis=-1, keepdims=True) x_hat = x * F.rsqrt(var + eps) # broadcast weight -> (1,1,D) w = F.reshape(weight, (1, 1) + weight.shape) return F.mul2(x_hat, w)

-----------------------------

MLP (SwiGLU: down( silu(gate(x)) * up(x) ))

-----------------------------

def silu(x): return x * F.sigmoid(x)

def mlp(state_dict, layer_idx, x): # 期待キー名 (LLaMA 系) g_w = _get(state_dict, [f"model.layers.{layer_idx}.mlp.gate_proj.weight", f"layers.{layer_idx}.mlp.gate_proj.weight"]) g_b = _get(state_dict, [f"model.layers.{layer_idx}.mlp.gate_proj.bias", f"layers.{layer_idx}.mlp.gate_proj.bias"]) if any(k in state_dict for k in [f"model.layers.{layer_idx}.mlp.gate_proj.bias", f"layers.{layer_idx}.mlp.gate_proj.bias"]) else None

u_w = _get(state_dict, [f"model.layers.{layer_idx}.mlp.up_proj.weight", f"layers.{layer_idx}.mlp.up_proj.weight"])
u_b = _get(state_dict, [f"model.layers.{layer_idx}.mlp.up_proj.bias", f"layers.{layer_idx}.mlp.up_proj.bias"]) if any(k in state_dict for k in [f"model.layers.{layer_idx}.mlp.up_proj.bias", f"layers.{layer_idx}.mlp.up_proj.bias"]) else None

d_w = _get(state_dict, [f"model.layers.{layer_idx}.mlp.down_proj.weight", f"layers.{layer_idx}.mlp.down_proj.weight"])
d_b = _get(state_dict, [f"model.layers.{layer_idx}.mlp.down_proj.bias", f"layers.{layer_idx}.mlp.down_proj.bias"]) if any(k in state_dict for k in [f"model.layers.{layer_idx}.mlp.down_proj.bias", f"layers.{layer_idx}.mlp.down_proj.bias"]) else None

gate = F.affine(x, g_w, g_b) if g_b is not None else F.affine(x, g_w)
up   = F.affine(x, u_w, u_b) if u_b is not None else F.affine(x, u_w)
act  = silu(gate)
y    = F.mul2(act, up)
y    = F.affine(y, d_w, d_b) if d_b is not None else F.affine(y, d_w)
return y

-----------------------------

Multi-Head Attention (with optional GQA)

-----------------------------

def attention(state_dict, layer_idx, hidden_states, cos, sin, num_heads, num_kv_heads=None, attn_mask=None, dropout_rate=0.0, training=False): """ hidden_states: (B,T,D) cos,sin: (1,1,T,Hd) あるいは (B,1,T,Hd) attn_mask: (B,1,T,T) など (True=keep/1.0 allowed)。数値なら加算。 GQA: num_kv_heads を指定 (デフォルト None -> MHA) """ # Q,K,V projection q_w = _get(state_dict, [f"model.layers.{layer_idx}.self_attn.q_proj.weight", f"layers.{layer_idx}.self_attn.q_proj.weight"]) q_b = _get(state_dict, [f"model.layers.{layer_idx}.self_attn.q_proj.bias", f"layers.{layer_idx}.self_attn.q_proj.bias"]) if any(k in state_dict for k in [f"model.layers.{layer_idx}.self_attn.q_proj.bias", f"layers.{layer_idx}.self_attn.q_proj.bias"]) else None

k_w = _get(state_dict, [f"model.layers.{layer_idx}.self_attn.k_proj.weight", f"layers.{layer_idx}.self_attn.k_proj.weight"])
k_b = _get(state_dict, [f"model.layers.{layer_idx}.self_attn.k_proj.bias", f"layers.{layer_idx}.self_attn.k_proj.bias"]) if any(k in state_dict for k in [f"model.layers.{layer_idx}.self_attn.k_proj.bias", f"layers.{layer_idx}.self_attn.k_proj.bias"]) else None

v_w = _get(state_dict, [f"model.layers.{layer_idx}.self_attn.v_proj.weight", f"layers.{layer_idx}.self_attn.v_proj.weight"])
v_b = _get(state_dict, [f"model.layers.{layer_idx}.self_attn.v_proj.bias", f"layers.{layer_idx}.self_attn.v_proj.bias"]) if any(k in state_dict for k in [f"model.layers.{layer_idx}.self_attn.v_proj.bias", f"layers.{layer_idx}.self_attn.v_proj.bias"]) else None

q = F.affine(hidden_states, q_w, q_b) if q_b is not None else F.affine(hidden_states, q_w)
k = F.affine(hidden_states, k_w, k_b) if k_b is not None else F.affine(hidden_states, k_w)
v = F.affine(hidden_states, v_w, v_b) if v_b is not None else F.affine(hidden_states, v_w)

# reshape to heads
q = split_heads(q, num_heads)    # (B,H,T,Hd)
if num_kv_heads is None:
    num_kv_heads = num_heads
k = split_heads(k, num_kv_heads)
v = split_heads(v, num_kv_heads)

# RoPE
q, k = apply_rotary_pos_emb(q, k, cos, sin)

# GQA: repeat kv heads to match q heads
if num_heads != num_kv_heads:
    assert num_heads % num_kv_heads == 0, "query_heads must be divisible by kv_heads"
    rep = num_heads // num_kv_heads
    k = repeat_kv(k, rep)
    v = repeat_kv(v, rep)

# (B,H,T,Hd) -> (B*H,T,Hd)
b, h, t, hd = q.shape
qf = F.reshape(q, (b * h, t, hd))
kf = F.reshape(k, (b * h, t, hd))
vf = F.reshape(v, (b * h, t, hd))

# scaled dot-product attention
scale = 1.0 / math.sqrt(hd)
attn = F.batch_matmul(qf, kf, transpose_a=False, transpose_b=True)  # (B*H,T,T)

if attn_mask is not None:
    # boolean mask: True=keep, False=mask -> -inf 相当を足す
    if attn_mask.dtype == np.bool_ or attn_mask.dtype == bool:
        mask = F.logical_not(attn_mask)  # True where to mask
        attn = F.where(mask, F.constant(-1e9, attn.shape), attn)
    else:
        # 数値マスクは加算
        attn = attn + attn_mask

attn = attn * scale
attn = F.softmax(attn, axis=-1)
if dropout_rate > 0.0 and training:
    attn = F.dropout(attn, p=dropout_rate)

out = F.batch_matmul(attn, vf)  # (B*H,T,Hd)

# back to (B,T,D)
out = F.reshape(out, (b, h, t, hd))
out = merge_heads(out)

# output projection
o_w = _get(state_dict, [f"model.layers.{layer_idx}.self_attn.o_proj.weight", f"layers.{layer_idx}.self_attn.o_proj.weight"])
o_b = _get(state_dict, [f"model.layers.{layer_idx}.self_attn.o_proj.bias", f"layers.{layer_idx}.self_attn.o_proj.bias"]) if any(k in state_dict for k in [f"model.layers.{layer_idx}.self_attn.o_proj.bias", f"layers.{layer_idx}.self_attn.o_proj.bias"]) else None
out = F.affine(out, o_w, o_b) if o_b is not None else F.affine(out, o_w)
return out

-----------------------------

Decoder Layer

-----------------------------

def decoder_layer(state_dict, layer_idx, hidden_states, cos, sin, num_heads, num_kv_heads=None, dropout_rate=0.0, training=False): # pre-attn RMSNorm in_w = _get(state_dict, [f"model.layers.{layer_idx}.input_layernorm.weight", f"layers.{layer_idx}.input_layernorm.weight"]) x = rms_norm(in_w, hidden_states)

# self-attention
attn_out = attention(state_dict, layer_idx, x, cos, sin, num_heads, num_kv_heads, None, dropout_rate, training)
hidden_states = hidden_states + attn_out

# post-attn RMSNorm
post_w = _get(state_dict, [f"model.layers.{layer_idx}.post_attention_layernorm.weight", f"layers.{layer_idx}.post_attention_layernorm.weight"])
x = rms_norm(post_w, hidden_states)

# MLP
mlp_out = mlp(state_dict, layer_idx, x)
hidden_states = hidden_states + mlp_out
return hidden_states

-----------------------------

LM Head (Im_head)

-----------------------------

def lm_head(state_dict, x): w = _get(state_dict, ["lm_head.weight", "im_head.weight", "output.weight"])  # 互換キー b = state_dict.get("lm_head.bias", state_dict.get("im_head.bias", state_dict.get("output.bias", None))) return F.affine(x, w, b) if b is not None else F.affine(x, w)

-----------------------------

使い方の最小例

-----------------------------

if name == "main": # ダミー shapes B, T, D, H = 2, 16, 128, 8 Hd = D // H

# ダミー state_dict（実際は学習済みパラメータを入れてください）
rng = np.random.RandomState(0)
def randw(i,o):
    return nn.Variable.from_numpy_array(rng.randn(i,o).astype(np.float32))
def randb(o):
    return nn.Variable.from_numpy_array(rng.randn(o).astype(np.float32))

sd = {}
li = 0
# attn
sd[f"model.layers.{li}.self_attn.q_proj.weight"] = randw(D, D)
sd[f"model.layers.{li}.self_attn.k_proj.weight"] = randw(D, D)
sd[f"model.layers.{li}.self_attn.v_proj.weight"] = randw(D, D)
sd[f"model.layers.{li}.self_attn.o_proj.weight"] = randw(D, D)
# norm
sd[f"model.layers.{li}.input_layernorm.weight"] = randb(D)
sd[f"model.layers.{li}.post_attention_layernorm.weight"] = randb(D)
# mlp
inter = D * 4
sd[f"model.layers.{li}.mlp.gate_proj.weight"] = randw(D, inter)
sd[f"model.layers.{li}.mlp.up_proj.weight"]   = randw(D, inter)
sd[f"model.layers.{li}.mlp.down_proj.weight"] = randw(inter, D)
# head
sd["lm_head.weight"] = randw(D, 32000)

x = nn.Variable.from_numpy_array(rng.randn(B, T, D).astype(np.float32))
cos, sin = rotary_emb(T, Hd)
y = decoder_layer(sd, li, x, cos, sin, num_heads=H)
logits = lm_head(sd, y)
# print shapes for sanity
print("y shape:", y.shape)
print("logits shape:", logits.shape)

