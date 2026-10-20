# Machine Learning

Experimental machine-learning programs by Flyxion.

## Order-aware data attribution

`ordered_attribution.py` is an exploratory scaffold for studying the contributions
of training-data chunks during sequential learning.

The synthetic task includes useful, duplicate, irrelevant, and corrupted
chunks. The experiment compares sampled training-order marginal contributions
with leave-one-out attribution and evaluates removal of low-scoring chunks.

The planted categories describe how examples are constructed; their actual
effects depend on the model, training procedure, and evaluation task.

This is work in progress. Preliminary smoke tests are development checks
and do not establish that one attribution method outperforms another.
GPU runtime and memory requirements have not yet been measured locally.

## Usage

Use Python with PyTorch and NumPy installed. Inspect the available options:

```bash
python ordered_attribution.py --help
```

For GPU execution, use a PyTorch installation compatible with your NVIDIA
hardware and driver.

## License

MIT. See LICENSE.
