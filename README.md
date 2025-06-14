# Moras

This repository collects small experiments and utilities. It currently includes
pose estimation metric helpers with accompanying unit tests.

## Installing dependencies

Create a virtual environment and install requirements:

```bash
pip install -r requirements.txt
```

The tests require NumPy. If it is not installed, the test suite will skip the
metric tests.

## Using the metrics

Import the metrics directly from the `metrics` package:

```python
from metrics import f1_depth, mpjpe2d, mpjpe3d, mean_average_precision, pca
```

Each function accepts NumPy arrays. See the docstrings in
`metrics/pose_metrics.py` for details on the expected shapes and return values.

## Running tests

Install the dependencies as above and then run:

```bash
pytest -q
```

If NumPy is installed, all tests in `tests/` should pass.
