import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, Sequence, Optional, Any
from dataclasses import dataclass

IGNORE_LABEL_ID = -100


class LossOutput(eqx.Module):
    loss: jnp.ndarray
    metrics: Dict[str, jnp.ndarray]
    all_halted: jnp.ndarray


def act_loss_fn(
    model: eqx.Module,
    carry: Any,
    batch: Dict[str, jnp.ndarray],
    key: Optional[jax.random.PRNGKey] = None,
    is_training: bool = True,
):
    new_carry, outputs = model(carry, batch, key=key, is_training=is_training)

    labels = new_carry.data["labels"]
    mask = labels != IGNORE_LABEL_ID
    n_tok = mask.sum(-1)
    valid = new_carry.halted & (n_tok > 0)
    vf = valid.astype(jnp.float32)

    preds = jnp.argmax(outputs["logits"], axis=-1)
    tok_correct = (preds == labels) & mask
    seq_correct = tok_correct.sum(-1) == n_tok

    # losses ----------------------------------------------------------------
    denom = jnp.maximum(n_tok, 1)[..., None]  # [B,1]
    lm_loss = (softmax_cross_entropy(outputs["logits"], labels) / denom).mean()

    q_halt_loss = bce_with_logits(
        outputs["q_halt_logits"].astype(jnp.float32),
        seq_correct.astype(jnp.float32),
    ).mean()

    q_cont_loss = jnp.array(0.0, dtype=jnp.float32)
    if "target_q_continue" in outputs:
        q_cont_loss = bce_with_logits(
            outputs["q_continue_logits"].astype(jnp.float32),
            outputs["target_q_continue"].astype(jnp.float32),
        ).mean()

    total_loss = lm_loss + 0.5 * (q_halt_loss + q_cont_loss)

    # metrics ----------------------------------------------------------------
    valid_sum = vf.sum()
    safe_inv = jnp.where(valid_sum > 0, 1.0 / valid_sum, jnp.nan)

    tok_corr_sum = (tok_correct.astype(jnp.float32) * vf[:, None]).sum()
    tok_total_sum = (mask.astype(jnp.float32) * vf[:, None]).sum()
    mean_tok_acc = jnp.where(tok_total_sum > 0, tok_corr_sum / tok_total_sum, jnp.nan)

    mean_seq_acc = (seq_correct.astype(jnp.float32) * vf).sum() * safe_inv

    q_halt_pred = outputs["q_halt_logits"] >= 0
    q_halt_acc = (
        (q_halt_pred == seq_correct).astype(jnp.float32) * vf
    ).sum() * safe_inv

    mean_steps = (new_carry.steps.astype(jnp.float32) * vf).sum() * safe_inv

    metrics = dict(
        count=valid_sum,
        accuracy=mean_tok_acc,
        exact_accuracy=mean_seq_acc,
        q_halt_accuracy=q_halt_acc,
        steps=mean_steps,
        lm_loss=lm_loss,
        q_halt_loss=q_halt_loss,
        q_continue_loss=q_cont_loss,
    )

    return new_carry, LossOutput(
        loss=total_loss,
        metrics=metrics,
        all_halted=new_carry.halted.all(),
    )


def bce_with_logits(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return -(
        targets * jax.nn.log_sigmoid(logits)
        + (1.0 - targets) * jax.nn.log_sigmoid(-logits)
    )


def softmax_cross_entropy(
    logits: jnp.ndarray, labels: jnp.ndarray, ignore_index: int = IGNORE_LABEL_ID
) -> jnp.ndarray:
    valid_mask = labels != ignore_index
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    transformed_labels = jnp.where(valid_mask, labels, 0)
    pred_logp = jnp.take_along_axis(
        log_probs, transformed_labels[..., None], axis=-1
    ).squeeze(-1)
    return -jnp.where(valid_mask, pred_logp, 0)


def stablemax_cross_entropy(
    logits: jnp.ndarray, labels: jnp.ndarray, ignore_index: int = IGNORE_LABEL_ID
) -> jnp.ndarray:
    logprobs = log_stablemax(logits.astype(jnp.float64), axis=-1)
    valid_mask = labels != ignore_index
    transformed_labels = jnp.where(valid_mask, labels, 0)
    pred_logp = jnp.take_along_axis(
        logprobs, transformed_labels[..., None], axis=-1
    ).squeeze(-1)
    return -jnp.where(valid_mask, pred_logp, 0)


def log_stablemax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    s_x = s(x)
    return jnp.log(s_x / jnp.sum(s_x, axis=axis, keepdims=True))


def s(x: jnp.ndarray, epsilon: float = 1e-30) -> jnp.ndarray:
    return jnp.where(x < 0, 1 / (1 - x + epsilon), x + 1)
