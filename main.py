import os
import time
from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tyro
from tqdm import tqdm
import wandb

from hrm import Dataset, ACTModel, Carry, act_loss_fn


@dataclass
class TrainConfig:
    seed: int = 0
    data_path: str = "data/sudoku-extreme-1k-aug-1000"
    epochs: int = 20000
    eval_interval: int = 500
    global_batch_size: int = 384
    lr: float = 7e-5
    lr_min_ratio: float = 1.0
    lr_warmup_steps: int = 2000
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    puzzle_emb_ndim: int = 512
    H_cycles: int = 2
    L_cycles: int = 2
    H_layers: int = 4
    L_layers: int = 4
    hidden_size: int = 512
    expansion: int = 4
    num_heads: int = 8
    pos_encodings: str = "rope"
    halt_max_steps: int = 32
    halt_exploration_prob: float = 0.1
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    checkpoint_path: Optional[str] = None
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    seq_len: Optional[int] = None
    vocab_size: Optional[int] = None
    num_puzzle_identifiers: Optional[int] = None


class TrainState(eqx.Module):
    model: eqx.Module
    opt_state: optax.OptState
    carry: Optional[Carry]
    step: jnp.ndarray
    key: jax.random.PRNGKey


def create_train_state(config: TrainConfig, key: jax.random.PRNGKey, total_steps: int):
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr,
        warmup_steps=config.lr_warmup_steps,
        decay_steps=total_steps - config.lr_warmup_steps,
        end_value=config.lr * config.lr_min_ratio,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=config.beta1,
            b2=config.beta2,
            weight_decay=config.weight_decay,
        ),
    )
    model_key, key = jax.random.split(key)
    model = ACTModel(config, key=model_key)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    return TrainState(model, opt_state, None, jnp.array(0, jnp.int32), key), optimizer, lr_schedule


@eqx.filter_jit
def train_step(state: TrainState, batch, optimizer):
    carry = state.carry or state.model.initial_carry(batch)
    step_key, next_key = jax.random.split(state.key)

    def loss_fn(model, carry, batch, key):
        new_carry, loss_output = act_loss_fn(model, carry, batch, key)
        return loss_output.loss / batch["inputs"].shape[0], (new_carry, loss_output)

    (loss, (new_carry, loss_output)), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True
    )(state.model, carry, batch, step_key)
    updates, new_opt_state = optimizer.update(
        grads, state.opt_state, eqx.filter(state.model, eqx.is_array)
    )
    new_model = eqx.apply_updates(state.model, updates)
    new_state = TrainState(
        new_model, new_opt_state, new_carry, state.step + 1, next_key
    )
    metrics = {f"train/{k}": v for k, v in loss_output.metrics.items()}
    return new_state, metrics


@eqx.filter_jit
def eval_step(model: eqx.Module, carry: Optional[Carry], batch):
    carry = carry or model.initial_carry(batch)
    while True:
        carry, loss_output = model(carry, batch, is_training=False)
        if loss_output.all_halted:
            break
    return carry, loss_output


def evaluate(model: eqx.Module, eval_loader, eval_metadata):
    set_metrics = {name: {} for name in eval_metadata.sets}
    for set_name, batch, _ in eval_loader:
        carry, loss_output = eval_step(model, None, batch)
        for k, v in loss_output.metrics.items():
            if k not in set_metrics[set_name]:
                set_metrics[set_name][k] = 0.0
            set_metrics[set_name][k] += float(v)
    for set_name, metrics in set_metrics.items():
        count = max(metrics.pop("count", 1), 1)
        set_metrics[set_name] = {k: v / count for k, v in metrics.items()}
    return set_metrics


def save_checkpoint(state: TrainState, path: str, step: int):
    if path is None:
        return
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"step_{step}.eqx")
    eqx.tree_serialise_leaves(file_path, state.model)


def main(config: TrainConfig):
    key = jax.random.PRNGKey(config.seed)
    train_epochs_per_iter = config.eval_interval or config.epochs
    train_loader = Dataset(
        dataset_path=config.data_path,
        split="train",
        seed=config.seed,
        global_batch_size=config.global_batch_size,
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
    )
    eval_loader = Dataset(
        dataset_path=config.data_path,
        split="test",
        seed=config.seed,
        global_batch_size=config.global_batch_size,
        test_set_mode=True,
    )
    metadata = train_loader.metadata
    config.seq_len = metadata.seq_len
    config.vocab_size = metadata.vocab_size
    config.num_puzzle_identifiers = metadata.num_puzzle_identifiers

    total_steps = int(
        config.epochs
        * metadata.total_groups
        * metadata.mean_puzzle_examples
        / config.global_batch_size
    )

    state, optimizer, lr_schedule = create_train_state(config, key, total_steps)

    wandb.init(
        project=config.project_name or f"{os.path.basename(config.data_path)} ACT-JAX",
        name=config.run_name or f"ACT-{int(time.time())}",
        config=vars(config),
    )
    num_params = sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(state.model, eqx.is_array))
    )
    wandb.log({"num_params": num_params}, step=0)

    pbar = tqdm(total=total_steps)
    for epoch in range(0, config.epochs, train_epochs_per_iter):
        for _, batch, _ in train_loader:
            if state.step >= total_steps:
                break

            state, metrics = train_step(state, batch, optimizer)
            lr = lr_schedule(state.step - 1)
            metrics["train/lr"] = lr
            wandb.log(metrics, step=state.step)
            pbar.update(1)

        eval_metrics = evaluate(state.model, eval_loader, eval_loader.metadata)
        for set_name, metrics in eval_metrics.items():
            wandb.log(
                {f"eval/{set_name}/{k}": v for k, v in metrics.items()}, step=state.step
            )
        save_checkpoint(state, config.checkpoint_path, state.step)

    pbar.close()
    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
