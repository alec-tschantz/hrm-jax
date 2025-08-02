import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tyro
from tqdm import tqdm
import wandb

from hrm import ACTModel, Carry, Dataset, act_loss_fn


@dataclass
class TrainConfig:
    seed: int = 0
    data_path: str = "data/sudoku-extreme-1k-aug-1000"
    epochs: int = 100
    eval_interval: int = 10
    global_batch_size: int = 384
    lr: float = 1e-4
    lr_min_ratio: float = 0.1
    lr_warmup_steps: int = 2000
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    hidden_size: int = 512
    context_emb_size: int = 512
    h_cycles: int = 2
    l_cycles: int = 2
    h_layers: int = 4
    l_layers: int = 4
    expansion: int = 4
    num_heads: int = 8
    pos_encodings: str = "rope"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1
    checkpoint_path: Optional[str] = None
    seq_len: Optional[int] = None
    vocab_size: Optional[int] = None
    num_puzzle_identifiers: Optional[int] = None


class TrainState(eqx.Module):
    model: eqx.Module
    opt_state: optax.OptState
    carry: Optional[Carry]
    key: jax.random.PRNGKey


def build_schedule(cfg: TrainConfig, total_steps: int) -> optax.Schedule:
    warmup = cfg.lr_warmup_steps
    peak = cfg.lr
    min_ratio = cfg.lr_min_ratio

    def sched(i: int) -> jnp.ndarray:
        i_f = jnp.array(i, dtype=jnp.float32)
        warm = peak * i_f / jnp.maximum(1.0, float(warmup))
        prog = (i_f - warmup) / jnp.maximum(1.0, float(total_steps - warmup))
        cos = peak * (
            min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + jnp.cos(math.pi * prog))
        )
        return jnp.where(i < warmup, warm, cos)

    return sched


def create_train_state(
    cfg: TrainConfig, key: jax.random.PRNGKey, total_steps: int
) -> Tuple[TrainState, optax.GradientTransformation, optax.Schedule]:
    lr_schedule = build_schedule(cfg, total_steps)
    optim = optax.adamw(
        lr_schedule, b1=cfg.beta1, b2=cfg.beta2, weight_decay=cfg.weight_decay
    )
    m_key, key = jax.random.split(key)
    model = ACTModel(cfg, key=m_key)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    return TrainState(model, opt_state, None, key), optim, lr_schedule


@eqx.filter_jit
def train_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    optim: optax.GradientTransformation,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    carry = state.carry or state.model.initial_carry(batch)
    step_key, next_key = jax.random.split(state.key)

    def loss_fn(
        m: eqx.Module, c: Carry, b: Dict[str, jnp.ndarray], k: jax.random.PRNGKey
    ):
        new_c, out = act_loss_fn(m, c, b, k, True)
        return out.loss, (new_c, out.metrics)

    (_, (new_carry, metrics)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        state.model, carry, batch, step_key
    )

    grads = jax.tree.map(lambda g: g / batch["inputs"].shape[0], grads)
    updates, new_opt_state = optim.update(
        grads, state.opt_state, eqx.filter(state.model, eqx.is_array)
    )
    new_model = eqx.apply_updates(state.model, updates)
    new_state = TrainState(new_model, new_opt_state, new_carry, next_key)
    return new_state, metrics


@eqx.filter_jit
def eval_step(model: eqx.Module, carry: Optional[Carry], batch: Dict[str, jnp.ndarray]):
    carry = carry or model.initial_carry(batch)
    while True:
        carry, out = model(carry, batch, is_training=False)
        if out.all_halted:
            break
    return out.metrics


def evaluate(model: eqx.Module, loader, meta):
    agg: Dict[str, Dict[str, float]] = {n: {} for n in meta.sets}
    for set_name, batch, _ in loader:
        m = eval_step(model, None, batch)
        for k, v in m.items():
            agg[set_name][k] = agg[set_name].get(k, 0.0) + float(v)
    for n, m in agg.items():
        cnt = max(m.pop("count", 1), 1)
        agg[n] = {k: v / cnt for k, v in m.items()}
    return agg


def save_ckpt(state: TrainState, path: Optional[str]):
    if path is None:
        return
    os.makedirs(path, exist_ok=True)
    eqx.tree_serialise_leaves(
        os.path.join(path, f"step_{int(state.key[0])}.eqx"), state.model
    )


def main(cfg: TrainConfig):
    key = jax.random.PRNGKey(cfg.seed)

    train_loader = Dataset(
        cfg.data_path,
        "train",
        cfg.seed,
        cfg.global_batch_size,
        False,
        epochs_per_iter=1,
    )
    eval_loader = Dataset(cfg.data_path, "test", cfg.seed, cfg.global_batch_size, True)

    meta = train_loader.metadata
    cfg.seq_len = meta.seq_len
    cfg.vocab_size = meta.vocab_size
    cfg.num_puzzle_identifiers = meta.num_puzzle_identifiers

    total_steps = len(train_loader) * cfg.epochs
    print(f"batches/epoch: {len(train_loader)}  |  total steps: {total_steps}")

    state, optim, lr_schedule = create_train_state(cfg, key, total_steps)
    wandb.init(project="hrm", name="hrm", config=vars(cfg))

    step = 0
    for epoch in range(cfg.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for _, batch, _ in pbar:
            state, metrics = train_step(state, batch, optim)
            step += 1
            wandb.log(
                {f"train/{k}": v for k, v in metrics.items()}
                | {"train/lr": float(lr_schedule(step - 1))},
                step=step,
            )

        if (epoch + 1) % cfg.eval_interval == 0:
            eval_metrics = evaluate(state.model, eval_loader, eval_loader.metadata)
            for set_name, m in eval_metrics.items():
                wandb.log({f"eval/{set_name}/{k}": v for k, v in m.items()}, step=step)
            save_ckpt(state, cfg.checkpoint_path)

    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
