import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Iterator, Tuple
import jax
import jax.numpy as jnp


@dataclass
class DatasetMetadata:
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    ignore_label_id: Optional[int]
    pad_id: int
    blank_identifier_id: int
    total_groups: int
    mean_puzzle_examples: float
    sets: list


class Dataset:
    def __init__(
        self,
        dataset_path: str,
        split: str,
        seed: int,
        global_batch_size: int,
        test_set_mode: bool,
        epochs_per_iter: int = 1,
    ):
        self.dataset_path = dataset_path
        self.split = split
        self.seed = seed
        self.global_batch_size = global_batch_size
        self.test_set_mode = test_set_mode
        self.epochs_per_iter = epochs_per_iter

        self.metadata = self._load_metadata()
        self._data = None
        self._iters = 0

    def _load_metadata(self) -> DatasetMetadata:
        with open(
            os.path.join(self.dataset_path, self.split, "dataset.json"), "r"
        ) as f:
            return DatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None,
        }

        self._data = {}
        for set_name in self.metadata.sets:
            self._data[set_name] = {
                field_name: np.load(
                    os.path.join(
                        self.dataset_path, self.split, f"{set_name}__{field_name}.npy"
                    ),
                    mmap_mode=mmap_mode,
                )
                for field_name, mmap_mode in field_mmap_modes.items()
            }

    def _sample_batch(
        self, rng: np.random.Generator, dataset: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a batch for training mode."""
        group_order = rng.permutation(dataset["group_indices"].size - 1)
        group_order = np.tile(group_order, self.epochs_per_iter)

        batch_indices = []
        batch_puzzle_indices = []
        current_size = 0
        start_index = 0

        while start_index < group_order.size and current_size < self.global_batch_size:
            group_id = group_order[start_index]
            puzzle_id = rng.integers(
                dataset["group_indices"][group_id],
                dataset["group_indices"][group_id + 1],
            )
            start_index += 1

            puzzle_start = dataset["puzzle_indices"][puzzle_id]
            puzzle_size = int(dataset["puzzle_indices"][puzzle_id + 1] - puzzle_start)
            append_size = min(puzzle_size, self.global_batch_size - current_size)

            batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
            batch_indices.append(
                puzzle_start + rng.choice(puzzle_size, append_size, replace=False)
            )

            current_size += append_size

        if current_size < self.global_batch_size:
            return None, None

        return np.concatenate(batch_indices), np.concatenate(batch_puzzle_indices)

    def _collate_batch(self, batch: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
        """Convert batch to JAX arrays and handle padding."""
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = -100

        if batch["puzzle_identifiers"].size < self.global_batch_size:
            pad_size = self.global_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": -100,
                "puzzle_identifiers": self.metadata.blank_identifier_id,
            }
            batch = {
                k: np.pad(
                    v,
                    ((0, pad_size),) + ((0, 0),) * (v.ndim - 1),
                    constant_values=pad_values[k],
                )
                for k, v in batch.items()
            }

        return {k: jnp.array(v) for k, v in batch.items()}

    def __iter__(self) -> Iterator[Tuple[str, Dict[str, jnp.ndarray], int]]:
        self._lazy_load_dataset()

        if self.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()

    def __len__(self) -> int:
        self._lazy_load_dataset()
        total = 0
        for ds in self._data.values():
            n = len(ds["inputs"])
            if self.test_set_mode:
                total += math.ceil(n / self.global_batch_size)
            else:
                total += n // self.global_batch_size
        return total

    def _iter_test(self):
        """Iterate through test data sequentially."""
        for set_name, dataset in self._data.items():
            total_examples = len(dataset["inputs"])
            start_index = 0

            while start_index < total_examples:
                end_index = min(total_examples, start_index + self.global_batch_size)

                puzzle_indices = []
                puzzle_index = (
                    np.searchsorted(
                        dataset["puzzle_indices"], start_index, side="right"
                    )
                    - 1
                )

                for i in range(start_index, end_index):
                    while (
                        puzzle_index + 1 < len(dataset["puzzle_indices"])
                        and i >= dataset["puzzle_indices"][puzzle_index + 1]
                    ):
                        puzzle_index += 1
                    puzzle_indices.append(puzzle_index)

                batch = self._collate_batch(
                    {
                        "inputs": dataset["inputs"][start_index:end_index],
                        "labels": dataset["labels"][start_index:end_index],
                        "puzzle_identifiers": dataset["puzzle_identifiers"][
                            puzzle_indices
                        ],
                    }
                )

                yield set_name, batch, end_index - start_index
                start_index += self.global_batch_size

    def _iter_train(self):
        """Iterate through training data with random sampling."""
        for set_name, dataset in self._data.items():
            self._iters += 1
            rng = np.random.Generator(np.random.Philox(seed=self.seed + self._iters))

            while True:
                batch_indices, batch_puzzle_indices = self._sample_batch(rng, dataset)

                if batch_indices is None:
                    break

                batch = self._collate_batch(
                    {
                        "inputs": dataset["inputs"][batch_indices],
                        "labels": dataset["labels"][batch_indices],
                        "puzzle_identifiers": dataset["puzzle_identifiers"][
                            batch_puzzle_indices
                        ],
                    }
                )

                yield set_name, batch, self.global_batch_size
