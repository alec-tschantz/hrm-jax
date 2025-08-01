# hrm-jax

Jax implementation of [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734). Based on the original [pytorch version](https://github.com/sapientinc/HRM/tree/main).



```sh
pip install -e .

python build_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000

python main.py
```