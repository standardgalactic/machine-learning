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

## First completed GPU experiment

Completed on October 20, 2026, using an NVIDIA GeForce RTX 5060
with 8 GB VRAM.

The experiment used 54 synthetic chunks: 18 core, 12 duplicate,
12 irrelevant, and 12 harmful. It completed 397 training runs,
including 220 permutation runs. The permutation stage took 1,054 seconds.

### Attribution results

| Metric | Leave-one-out | Permutation, matched budget | Permutation, full budget |
| --- | ---: | ---: | ---: |
| Core > irrelevant AUROC | 0.556 | 1.000 | 1.000 |
| Duplicate > irrelevant AUROC | 0.556 | 1.000 | 1.000 |
| Irrelevant > harmful AUROC | 0.688 | 1.000 | 1.000 |
| Useful > junk AUROC | 0.625 | 1.000 | 1.000 |

Reported Spearman rank reliability was 0.255 between leave-one-out
reference orders and 0.969 between permutation blocks. The precise
sampling and budget allocation should be considered when comparing
these reliability measurements.

### Removal experiment

Held-out test loss after removing the lowest-ranked chunks;
lower is better. The baseline with all chunks was 1.8886.

| Method | Remove 12 | Remove 24 | Remove 36 |
| --- | ---: | ---: | ---: |
| Leave-one-out | 2.0190 | 1.9187 | 2.1039 |
| Permutation, matched budget | 1.5713 | 1.0524 | 1.5725 |
| Permutation, full budget | 1.5713 | 1.0524 | 1.5835 |
| Random | 2.0323 | 2.0751 | 2.5636 |
| Category-based oracle | 1.5713 | 1.0524 | 1.3160 |

Removing 24 chunks using either permutation ranking reduced test loss
by approximately 44.3%, matching the category-based oracle at that
removal count.

This single run supports the usefulness of permutation attribution
under the tested configuration. It does not establish general
superiority across datasets, models, or seeds. Independent-seed
replication remains pending.

The machine-readable outputs are retained in
`ordered_attribution_out/results.json` and
`ordered_attribution_out/values.csv`.
