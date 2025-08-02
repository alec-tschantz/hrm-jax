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
    data: Dict[str, jnp.ndarray]


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

    def __call__(self, h: jnp.ndarray, cos_sin=None) -> jnp.ndarray:
        h = h + self.self_attn(h, cos_sin)
        h = rms_norm(h, self.norm_eps)
        h = h + self.mlp(h)
        h = rms_norm(h, self.norm_eps)
        return h


class ReasoningModule(eqx.Module):
    layers: List[Block]

    def __init__(self, layers: List[Block]):
        self.layers = layers

    def __call__(
        self,
        h: jnp.ndarray,
        injection: jnp.ndarray,
        cos_sin=None,
    ) -> jnp.ndarray:
        h = h + injection
        for layer in self.layers:
            h = layer(h, cos_sin)
        return h


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

    reset_key: jax.random.PRNGKey

    h_cycles: int
    l_cycles: int

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
        self.h_cycles = cfg.h_cycles
        self.l_cycles = cfg.l_cycles
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
        self.puzzle_emb = eqx.tree_at(
            lambda m: m.weight,
            self.puzzle_emb,
            jnp.zeros((cfg.num_puzzle_identifiers, hidden_size)),
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
            lambda m: m.weight,
            self.q_head,
            jnp.zeros((2, hidden_size)),
        )
        self.q_head = eqx.tree_at(
            lambda m: m.bias,
            self.q_head,
            jnp.full((2,), -5.0),
        )

        h_keys = jax.random.split(keys[5], cfg.h_layers)
        l_keys = jax.random.split(keys[6], cfg.l_layers)
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

        self.reset_key = keys[7]

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
            data={k: jnp.empty_like(v) for k, v in batch.items()},
        )

    def _reset_inner(
        self,
        flag: jnp.ndarray,
        carry: InnerCarry,
    ) -> InnerCarry:
        h_key, l_key = jax.random.split(self.reset_key)
        h_init = jax.random.normal(h_key, (self.hidden_size,))
        l_init = jax.random.normal(l_key, (self.hidden_size,))

        return InnerCarry(
            zh=jnp.where(flag[:, None, None], h_init, carry.zh),
            zl=jnp.where(flag[:, None, None], l_init, carry.zl),
        )

    def _embed_inputs(
        self, inputs: jnp.ndarray, puzzle_identifiers: jnp.ndarray
    ) -> jnp.ndarray:
        emb = jax.vmap(jax.vmap(self.embed_tokens))(inputs)
        p_emb = jax.vmap(self.puzzle_emb)(puzzle_identifiers)
        emb = jnp.concatenate([p_emb[:, None, :], emb], axis=1)
        return self.embed_scale * emb

    def _forward_inner(
        self,
        carry: InnerCarry,
        data: Dict[str, jnp.ndarray],
    ) -> Tuple[InnerCarry, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        cos_sin = self.rotary_emb()
        inp = self._embed_inputs(data["inputs"], data["puzzle_identifiers"])
        zh, zl = carry.zh, carry.zl

        for h_step in range(self.h_cycles):
            for l_step in range(self.l_cycles):
                if not (
                    (h_step == self.h_cycles - 1) and (l_step == self.l_cycles - 1)
                ):
                    zl = jax.lax.stop_gradient(self.l_layers(zl, zh + inp, cos_sin))

            if not (h_step == self.h_cycles - 1):
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

    def _filter_halting(
        self,
        carry: Carry,
        batch: Dict[str, jnp.ndarray],
    ) -> Tuple[InnerCarry, jnp.ndarray, Dict[str, jnp.ndarray]]:
        inner = self._reset_inner(carry.halted, carry.inner_carry)
        steps = jnp.where(carry.halted, 0, carry.steps)

        data = {
            "inputs": jnp.where(
                carry.halted[:, None], batch["inputs"], carry.data["inputs"]
            ),
            "labels": jnp.where(
                carry.halted[:, None], batch["labels"], carry.data["labels"]
            ),
            "puzzle_identifiers": jnp.where(
                carry.halted,
                batch["puzzle_identifiers"],
                carry.data["puzzle_identifiers"],
            ),
        }

        return inner, steps, data

    def _halt_logic(
        self,
        inner: InnerCarry,
        data: Dict[str, jnp.ndarray],
        steps: jnp.ndarray,
        q_h: jnp.ndarray,
        q_c: jnp.ndarray,
        key: Optional[jax.random.PRNGKey],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        is_last_step = steps >= self.halt_max_steps
        halted = is_last_step | (q_h > q_c)

        if key is not None:
            e_key, h_key = jax.random.split(key)
            explore = jax.random.uniform(e_key, q_h.shape) < self.halt_exploration_prob
            min_halt_steps = jnp.where(
                explore,
                jax.random.randint(h_key, steps.shape, 2, self.halt_max_steps + 1),
                0,
            )
            halted = halted & (steps >= min_halt_steps)

        _, _, (next_q_h, next_q_c) = self._forward_inner(inner, data)
        next_q = jnp.where(is_last_step, next_q_h, jnp.maximum(next_q_h, next_q_c))
        target_q_continue = jax.nn.sigmoid(next_q)

        return halted, target_q_continue

    def __call__(
        self,
        carry: Carry,
        batch: Dict[str, jnp.ndarray],
        key: Optional[jax.random.PRNGKey] = None,
        is_training: bool = False,
    ) -> Tuple[Carry, Dict[str, jnp.ndarray]]:
        inner, steps, data = self._filter_halting(carry, batch)

        inner, logits, (q_h, q_c) = self._forward_inner(inner, data)
        outputs = {"logits": logits, "q_halt_logits": q_h, "q_continue_logits": q_c}
        steps = steps + 1

        halted = steps >= self.halt_max_steps

        if is_training and self.halt_max_steps > 1:
            halted, target_q_continue = self._halt_logic(
                inner, data, steps, q_h, q_c, key
            )
            outputs["target_q_continue"] = target_q_continue

        return Carry(inner, steps, halted, data), outputs
