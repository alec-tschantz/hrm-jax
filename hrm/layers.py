import math

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple


def rms_norm(x: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    variance = jnp.mean(x**2, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(variance + eps)


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: jnp.ndarray, k: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    q_embed = (q * cos[..., None, :]) + (rotate_half(q) * sin[..., None, :])
    k_embed = (k * cos[..., None, :]) + (rotate_half(k) * sin[..., None, :])
    return q_embed, k_embed


class RotaryEmbedding(eqx.Module):
    # TODO
    cos_cached: jnp.ndarray = eqx.static_field()
    sin_cached: jnp.ndarray =  eqx.static_field()

    def __init__(
        self, dim: int, max_position_embeddings: int, base: float = 10000.0, *, key
    ):
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        t = jnp.arange(max_position_embeddings, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)

        emb = jnp.concatenate([freqs, freqs], axis=-1)
        self.cos_cached = jnp.cos(emb)
        self.sin_cached = jnp.sin(emb)

    def __call__(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.cos_cached, self.sin_cached


class Attention(eqx.Module):
    qkv_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear
    head_dim: int
    num_heads: int
    num_key_value_heads: int
    causal: bool

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        causal: bool = False,
        *,
        key
    ):
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        qkv_key, o_key = jax.random.split(key)

        total_heads = num_heads + 2 * num_key_value_heads
        self.qkv_proj = eqx.nn.Linear(
            hidden_size, total_heads * head_dim, use_bias=False, key=qkv_key
        )
        self.o_proj = eqx.nn.Linear(
            num_heads * head_dim, hidden_size, use_bias=False, key=o_key
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        cos_sin: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = jax.vmap(self.qkv_proj)(
            hidden_states.reshape(-1, hidden_states.shape[-1])
        )
        qkv = qkv.reshape(batch_size, seq_len, -1, self.head_dim)

        query = qkv[:, :, : self.num_heads]
        key = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads :]

        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if self.num_key_value_heads != self.num_heads:
            key = jnp.repeat(key, self.num_heads // self.num_key_value_heads, axis=2)
            value = jnp.repeat(
                value, self.num_heads // self.num_key_value_heads, axis=2
            )

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = jnp.einsum("bshd,bthd->bhst", query, key) * scale

        if self.causal:
            mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            mask = mask[None, None, :, :]
            scores = jnp.where(mask, scores, -1e10)

        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_output = jnp.einsum("bhst,bthd->bshd", attn_weights, value)

        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        output = jax.vmap(self.o_proj)(attn_output.reshape(-1, attn_output.shape[-1]))
        return output.reshape(batch_size, seq_len, -1)


class SwiGLU(eqx.Module):
    gate_up_proj: eqx.nn.Linear
    down_proj: eqx.nn.Linear

    def __init__(self, hidden_size: int, expansion: float, *, key):
        inter_size = int(round(expansion * hidden_size * 2 / 3))
        inter_size = ((inter_size + 255) // 256) * 256

        gate_up_key, down_key = jax.random.split(key)

        self.gate_up_proj = eqx.nn.Linear(
            hidden_size, inter_size * 2, use_bias=False, key=gate_up_key
        )
        self.down_proj = eqx.nn.Linear(
            inter_size, hidden_size, use_bias=False, key=down_key
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate_up = jax.vmap(self.gate_up_proj)(x.reshape(-1, x.shape[-1]))
        gate_up = gate_up.reshape(*x.shape[:-1], -1)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        activated = jax.nn.silu(gate) * up
        output = jax.vmap(self.down_proj)(activated.reshape(-1, activated.shape[-1]))
        return output.reshape(*x.shape)
