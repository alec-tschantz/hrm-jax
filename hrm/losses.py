import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, Tuple, Any, Optional, Sequence
from dataclasses import dataclass

from hrm.model import ACTModel, Carry

IGNORE_LABEL_ID = -100


class LossOutput(eqx.Module):
    loss: jnp.ndarray
    metrics: Dict[str, jnp.ndarray]
    outputs: Dict[str, jnp.ndarray]
    all_halted: jnp.ndarray


class ACTLossHead(eqx.Module):
    model: ACTModel
    loss_type: str

    def __init__(self, model: ACTModel, loss_type: str = "softmax_cross_entropy"):
        self.model = model
        self.loss_type = loss_type

    def initial_carry(self, batch: Dict[str, jnp.ndarray]) -> Carry:
        return self.model.initial_carry(batch)

    def __call__(
        self,
        carry: Carry,
        batch: Dict[str, jnp.ndarray],
        return_keys: Sequence[str] = (),
        key: Optional[jax.random.PRNGKey] = None,
        is_training: bool = False,
    ) -> Tuple[Carry, LossOutput]:
        new_carry, outputs = self.model(carry, batch, key, is_training)
        labels = new_carry.current_data["labels"]
        mask = labels != IGNORE_LABEL_ID
        loss_counts = mask.sum(-1)
        loss_divisor = jnp.maximum(loss_counts, 1)[..., None]
        is_correct = mask & (jnp.argmax(outputs["logits"], axis=-1) == labels)
        seq_is_correct = is_correct.sum(-1) == loss_counts
        valid_metrics = new_carry.halted & (loss_counts > 0)
        metrics = {
            "count": valid_metrics.sum(),
            "accuracy": jnp.where(
                valid_metrics,
                (is_correct.astype(jnp.float32) / loss_divisor).sum(-1),
                0,
            ).sum(),
            "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
            "q_halt_accuracy": (
                valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)
            ).sum(),
            "steps": jnp.where(valid_metrics, new_carry.steps, 0).sum(),
        }
        loss_fn = globals()[self.loss_type]
        lm_loss = (
            loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID)
            / loss_divisor
        ).sum()
        qh = outputs["q_halt_logits"]
        q_halt_loss = binary_cross_entropy_with_logits(
            qh.astype(jnp.float32), seq_is_correct.astype(jnp.float32)
        ).sum()
        q_continue_loss = 0.0
        if "target_q_continue" in outputs:
            qc = outputs["q_continue_logits"]
            tc = outputs["target_q_continue"]
            q_continue_loss = binary_cross_entropy_with_logits(
                qc.astype(jnp.float32), tc.astype(jnp.float32)
            ).sum()
            metrics["q_continue_loss"] = q_continue_loss
        metrics["lm_loss"] = lm_loss
        metrics["q_halt_loss"] = q_halt_loss
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        filtered_outputs = {k: outputs[k] for k in return_keys if k in outputs}
        loss_output = LossOutput(
            loss=total_loss,
            metrics=metrics,
            outputs=filtered_outputs,
            all_halted=new_carry.halted.all(),
        )
        return new_carry, loss_output


def s(x: jnp.ndarray, epsilon: float = 1e-30) -> jnp.ndarray:
    return jnp.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    s_x = s(x)
    return jnp.log(s_x / jnp.sum(s_x, axis=axis, keepdims=True))


def stablemax_cross_entropy(
    logits: jnp.ndarray, labels: jnp.ndarray, ignore_index: int = -100
) -> jnp.ndarray:
    logprobs = log_stablemax(logits, axis=-1)
    valid_mask = labels != ignore_index
    transformed_labels = jnp.where(valid_mask, labels, 0)
    prediction_logprobs = jnp.take_along_axis(
        logprobs, transformed_labels[..., None], axis=-1
    ).squeeze(-1)
    return -jnp.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(
    logits: jnp.ndarray, labels: jnp.ndarray, ignore_index: int = -100
) -> jnp.ndarray:
    valid_mask = labels != ignore_index
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    transformed_labels = jnp.where(valid_mask, labels, 0)
    prediction_logprobs = jnp.take_along_axis(
        log_probs, transformed_labels[..., None], axis=-1
    ).squeeze(-1)
    return -jnp.where(valid_mask, prediction_logprobs, 0)


def binary_cross_entropy_with_logits(
    logits: jnp.ndarray, labels: jnp.ndarray
) -> jnp.ndarray:
    return -(
        labels * jax.nn.log_sigmoid(logits) + (1 - labels) * jax.nn.log_sigmoid(-logits)
    )
