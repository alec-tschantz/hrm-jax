import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import equinox as eqx
import jax
import jax.numpy as jnp

from hrm.layers import Attention, RotaryEmbedding, SwiGLU, rms_norm


class InnerCarry(eqx.Module):
    zh: jnp.ndarray
    zl: jnp.ndarray


class Carry(eqx.Module):
    inner_carry: InnerCarry
    steps: jnp.ndarray
    halted: jnp.ndarray
    current_data: Dict[str, jnp.ndarray]


class Block(eqx.Module):
    self_attn: Attention
    mlp: SwiGLU
    norm_eps: float

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        norm_eps: float,
        *,
        key: jax.random.PRNGKey,
    ):
        attn_key, mlp_key = jax.random.split(key)
        self.self_attn = Attention(
            hidden_size=hidden_size,
            head_dim=hidden_size // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False,
            key=attn_key,
        )
        self.mlp = SwiGLU(hidden_size=hidden_size, expansion=expansion, key=mlp_key)
        self.norm_eps = norm_eps

    def __call__(self, hidden_states: jnp.ndarray, cos_sin=None) -> jnp.ndarray:
        hidden_states = hidden_states + self.self_attn(hidden_states, cos_sin)
        hidden_states = rms_norm(hidden_states, self.norm_eps)
        hidden_states = hidden_states + self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states, self.norm_eps)
        return hidden_states


class ReasoningModule(eqx.Module):
    layers: List[Block]

    def __init__(self, layers: List[Block]):
        self.layers = layers

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        injection: jnp.ndarray,
        cos_sin=None,
    ) -> jnp.ndarray:
        hidden_states = hidden_states + injection
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos_sin)
        return hidden_states


class ACTModel(eqx.Module):
    embed_scale: float
    hidden_size: int
    seq_len: int

    embed_tokens: eqx.nn.Embedding
    puzzle_emb: eqx.nn.Embedding
    rotary_emb: RotaryEmbedding

    lm_head: eqx.nn.Linear
    q_head: eqx.nn.Linear

    h_layers: ReasoningModule
    l_layers: ReasoningModule

    H_init: jnp.ndarray
    L_init: jnp.ndarray

    H_cycles: int
    L_cycles: int

    halt_max_steps: int
    halt_exploration_prob: float
    puzzle_emb_len: int

    def __init__(self, cfg: Any, key: jax.random.PRNGKey):
        hidden_size = cfg.hidden_size
        num_heads = cfg.num_heads
        seq_len = cfg.seq_len

        self.puzzle_emb_len = 1
        self.embed_scale = math.sqrt(hidden_size)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.H_cycles = cfg.H_cycles
        self.L_cycles = cfg.L_cycles
        self.halt_max_steps = cfg.halt_max_steps
        self.halt_exploration_prob = cfg.halt_exploration_prob

        keys = jax.random.split(key, 10)
        self.embed_tokens = eqx.nn.Embedding(
            num_embeddings=cfg.vocab_size,
            embedding_size=hidden_size,
            key=keys[0],
        )
        self.puzzle_emb = eqx.nn.Embedding(
            cfg.num_puzzle_identifiers, hidden_size, key=keys[1]
        )
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_size // num_heads,
            max_position_embeddings=seq_len + self.puzzle_emb_len,
            base=cfg.rope_theta,
            key=keys[2],
        )

        self.lm_head = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=cfg.vocab_size,
            use_bias=False,
            key=keys[3],
        )
        self.q_head = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=2,
            use_bias=True,
            key=keys[4],
        )
        self.q_head = eqx.tree_at(
            lambda m: m.bias,
            self.q_head,
            jnp.full((2,), -5.0),
        )

        h_keys = jax.random.split(keys[5], cfg.H_layers)
        l_keys = jax.random.split(keys[6], cfg.L_layers)
        self.h_layers = ReasoningModule(
            [
                Block(hidden_size, num_heads, cfg.expansion, cfg.rms_norm_eps, key=k)
                for k in h_keys
            ]
        )
        self.l_layers = ReasoningModule(
            [
                Block(hidden_size, num_heads, cfg.expansion, cfg.rms_norm_eps, key=k)
                for k in l_keys
            ]
        )

        self.H_init = jax.random.normal(keys[7], (hidden_size,))
        self.L_init = jax.random.normal(keys[8], (hidden_size,))

    def _input_embeddings(
        self, inputs: jnp.ndarray, puzzle_ids: jnp.ndarray
    ) -> jnp.ndarray:
        emb = jax.vmap(jax.vmap(self.embed_tokens))(inputs)
        p_emb = jax.vmap(self.puzzle_emb)(puzzle_ids)
        emb = jnp.concatenate([p_emb[:, None, :], emb], axis=1)
        return self.embed_scale * emb

    def initial_carry(
        self,
        batch: Dict[str, jnp.ndarray],
    ) -> Carry:
        bsz = batch["inputs"].shape[0]
        shape = (bsz, self.seq_len + self.puzzle_emb_len, self.hidden_size)
        return Carry(
            inner_carry=InnerCarry(jnp.empty(shape), jnp.empty(shape)),
            steps=jnp.zeros((bsz,), dtype=jnp.int32),
            halted=jnp.ones((bsz,), dtype=jnp.bool_),
            current_data={k: jnp.empty_like(v) for k, v in batch.items()},
        )

    def reset_carry(
        self,
        flag: jnp.ndarray,
        carry: InnerCarry,
    ) -> InnerCarry:
        return InnerCarry(
            zh=jnp.where(flag[:, None, None], self.H_init, carry.zh),
            zl=jnp.where(flag[:, None, None], self.L_init, carry.zl),
        )

    def forward_inner(
        self,
        carry: InnerCarry,
        data: Dict[str, jnp.ndarray],
    ) -> Tuple[InnerCarry, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        cos_sin = self.rotary_emb()
        inp = self._input_embeddings(data["inputs"], data["puzzle_identifiers"])
        zh, zl = carry.zh, carry.zl
        for h in range(self.H_cycles):
            for l in range(self.L_cycles):
                if not (h == self.H_cycles - 1 and l == self.L_cycles - 1):
                    zl = jax.lax.stop_gradient(self.l_layers(zl, zh + inp, cos_sin))
            if h < self.H_cycles - 1:
                zh = jax.lax.stop_gradient(self.h_layers(zh, zl, cos_sin))

        zl = self.l_layers(zl, zh + inp, cos_sin)
        zh = self.h_layers(zh, zl, cos_sin)
        new_inner = InnerCarry(
            zh=jax.lax.stop_gradient(zh),
            zl=jax.lax.stop_gradient(zl),
        )

        out = jax.vmap(jax.vmap(self.lm_head))(zh[:, self.puzzle_emb_len :])
        q = jax.vmap(self.q_head)(zh[:, 0])
        return new_inner, out, (q[:, 0], q[:, 1])

    def __call__(
        self,
        carry: Carry,
        batch: Dict[str, jnp.ndarray],
        key: Optional[jax.random.PRNGKey] = None,
        is_training: bool = False,
    ) -> Tuple[Carry, Dict[str, jnp.ndarray]]:
        inner = self.reset_carry(carry.halted, carry.inner_carry)
        steps = jnp.where(carry.halted, 0, carry.steps)
        inputs = jnp.where(
            carry.halted[:, None], batch["inputs"], carry.current_data["inputs"]
        )
        labels = jnp.where(
            carry.halted[:, None], batch["labels"], carry.current_data["labels"]
        )
        pids = jnp.where(
            carry.halted,
            batch["puzzle_identifiers"],
            carry.current_data["puzzle_identifiers"],
        )
        data = {"inputs": inputs, "puzzle_identifiers": pids, "labels": labels}

        inner, logits, (q_h, q_c) = self.forward_inner(inner, data)
        outputs = {"logits": logits, "q_halt_logits": q_h, "q_continue_logits": q_c}
        steps = steps + 1
        last = steps >= self.halt_max_steps
        halted = last
        if is_training and self.halt_max_steps > 1:
            halted |= q_h > q_c
            if key is not None:
                e_key, h_key = jax.random.split(key)
                explore = (
                    jax.random.uniform(e_key, q_h.shape) < self.halt_exploration_prob
                )
                min_step = jnp.where(
                    explore,
                    jax.random.randint(h_key, steps.shape, 2, self.halt_max_steps + 1),
                    self.halt_max_steps,
                )
                halted &= steps >= min_step
            nxt_inner, _, (n_q_h, n_q_c) = self.forward_inner(inner, data)
            next_q = jnp.where(last, n_q_h, jnp.maximum(n_q_h, n_q_c))
            outputs["target_q_continue"] = jax.nn.sigmoid(next_q)
        return Carry(inner, steps, halted, data), outputs
