# CNN & Recurrent Sequence Models — PyTorch and TensorFlow

Image classification (CIFAR-10, residual CNN) and time-series classification (BiLSTM/GRU with
attention pooling), each implemented in **both PyTorch and TensorFlow/Keras**, with
**Optuna hyperparameter tuning** and a set of **regularization** techniques against overfitting.

## Layout
```
src/
  config.py          shared default hyperparameters
  data.py            CIFAR-10 loader + synthetic time-series generator (numpy, framework-agnostic)
  utils.py           seeding, metrics (acc / macro-F1 / confusion matrix), plots, CLI helpers
  search_space.py    Optuna search space (shared by both frameworks)
  tune.py            Optuna tuner: TPE sampler + median pruner
  torch_impl/        models.py, train.py
  tf_impl/           models.py, train.py
tests/test_smoke.py  forward-pass and 2-epoch end-to-end tests
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-torch.txt     # PyTorch pipeline
pip install -r requirements-tf.txt        # TensorFlow pipeline (separate venv recommended)
```

## Run
```bash
# Train + evaluate (writes runs/<framework>_<task>/ with metrics, plots, model)
python -m src.torch_impl.train --task image --epochs 30
python -m src.torch_impl.train --task timeseries --cell gru --hidden 128
python -m src.tf_impl.train    --task image --epochs 30
python -m src.tf_impl.train    --task timeseries

# Hyperparameter search (validation split only), then retrain with the best params
python -m src.tune --framework torch --task image --trials 20 --tune-epochs 8 --subset 10000
python -m src.torch_impl.train --params runs/tune_torch_image/best_params.json --epochs 40

pytest -q
```
CLI priority: explicit flags > `--params` JSON > `src/config.py` defaults. Run `--help` for all options.

## Models
| Task | Model |
|---|---|
| Image | Residual conv blocks (2x Conv-BN, 1x1 skip, max-pool, spatial dropout) -> GAP -> dropout -> linear |
| Time series | LayerNorm -> stacked (Bi)LSTM/GRU -> additive attention pooling -> LayerNorm -> dropout -> linear |

The synthetic time-series task (6 classes) is defined by a base frequency (5 or 9 Hz) and the
temporal position of a short high-frequency burst, plus noise and a distractor channel, so the model
must use temporal context. Swap in your own data by replacing `make_timeseries` in `src/data.py`
(return `(N, T, C)` float32 arrays split into train/val/test).

## Regularization
| Technique | Where |
|---|---|
| Data augmentation | images: random crop/translate + horizontal flip; series: Gaussian jitter + amplitude scaling |
| Dropout | spatial dropout in conv blocks, dropout before classifier, dropout between RNN layers |
| Weight decay | decoupled (AdamW) |
| Label smoothing | cross-entropy loss |
| Batch / Layer norm | CNN uses BN; RNN uses LayerNorm |
| Gradient clipping | global norm (stabilises RNN training) |
| LR schedule | OneCycle (PyTorch) / cosine decay (TF) |
| Early stopping | on validation loss, best weights restored |

## Evaluation protocol
Train / validation / test splits are fixed by seed. Tuning and early stopping use only the validation
split; the test set is evaluated once at the end. Outputs: accuracy, macro-F1, per-class report
(`test_classification_report.json`), confusion matrix, and training curves (`history.png`) to check for
overfitting (train/val gap).

## Notes
- Implementation differences are minor: TF reports `val_loss` with label smoothing included, and
  augmentation runs inside the Keras model rather than in the data pipeline.
- Add your own results table here after running the experiments.
