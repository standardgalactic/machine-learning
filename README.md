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

## Curriculum experiment: October 20, 2026

Executed on an NVIDIA RTX 5060 with
`python curriculum_experiment.py --seeds 1 --perms 110`.

Seed 0 used 54 chunks: 18 core, 12 duplicate, 12 irrelevant,
and 12 harmful.

### Training order and selection

| Method | Test loss | Difference from random |
| --- | ---: | ---: |
| Random | 1.9773 | 0.0000 |
| Scalar descending | 3.2279 | +1.2506 |
| Scalar ascending | 2.1691 | +0.1918 |
| Early descending | 3.0766 | +1.0993 |
| Category oracle | 3.0973 | +1.1200 |
| Core only | 1.4840 | -0.4933 |
| Greedy | 3.2733 | +1.2960 |
| Greedy with stopping | 1.1858 | -0.7915 |

Lower loss is better. Greedy stopping retained 17 chunks:
13 core and four duplicates, with no irrelevant or harmful chunks.
Its loss was approximately 40% lower than random ordering.

Sorting all chunks by average attribution did not produce a successful
curriculum in this run. Conditional selection with stopping performed
better, distinguishing subset selection from full-corpus ordering.

### Position-dependent marginal contributions

Mean loss reduction by training-position bin:

| Category | Bin 1 | Bin 2 | Bin 3 | Bin 4 | Bin 5 | Bin 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Core | +0.10284 | +0.09771 | +0.09062 | +0.08265 | +0.08711 | +0.07827 |
| Duplicate | +0.09403 | +0.09721 | +0.09460 | +0.07828 | +0.07379 | +0.06510 |
| Irrelevant | -0.12892 | -0.04172 | -0.02730 | -0.03552 | -0.03356 | -0.03009 |
| Harmful | -0.15205 | -0.12207 | -0.11025 | -0.12353 | -0.12489 | -0.13420 |

### Arrival analysis

| Pair type | Pairs | First | Second | Gap | Second / first |
| --- | ---: | ---: | ---: | ---: | ---: |
| Duplicate | 12 | +0.09271 | +0.07426 | +0.01845 | 0.801 |
| Same-family core | 18 | +0.09707 | +0.08146 | +0.01560 | 0.839 |
| Cross-family core | 135 | +0.09326 | +0.08527 | +0.00798 | 0.914 |

Second arrivals retained less marginal value, particularly for duplicate
pairs. Pairs share chunks and trajectories and are not independent
experimental replications.

### Limitations

These are exploratory results from one seed. Across-seed standard
deviations are undefined. Full-corpus and subset methods use different
training amounts; follow-up experiments should compare random and
useful-only 17-chunk subsets with matched optimization budgets.
Validation/test separation for selection and stopping remains to
be verified.

Per-seed results and summary.json are saved in curriculum_out/.

## Five-seed curriculum experiment

Completed October 20, 2026, using an NVIDIA RTX 5060.
Command: `python curriculum_experiment.py --seeds 5`.
Each seed used 220 permutations and 54 training chunks.

| Method | Mean test loss | Standard deviation | Seeds beating random |
| --- | ---: | ---: | ---: |
| Random | 2.0175 | 0.0908 | — |
| Scalar descending | 3.0259 | 0.1970 | 0/5 |
| Scalar ascending | 1.6996 | 0.3808 | 4/5 |
| Early descending | 3.2231 | 0.2446 | 0/5 |
| Category oracle | 2.9265 | 0.1597 | 0/5 |
| Core only | 1.4399 | 0.0986 | 5/5 |
| Greedy, full corpus | 3.1308 | 0.1411 | 0/5 |
| Greedy with stopping | 1.1371 | 0.0879 | 5/5 |

Greedy stopping achieved the lowest reported test loss in every seed.
It retained 13–19 chunks, exclusively core examples and duplicates.
Its mean loss was 43.6% below random and 21.0% below core-only training.

Descending scalar ordering and full-corpus greedy ordering consistently
performed worse than random. These results distinguish useful subset
selection with stopping from ordering the entire training corpus.

### Arrival analysis across seeds

| Pair type | First-arrival gain | Second-arrival gain | Gap | Retained |
| --- | ---: | ---: | ---: | ---: |
| Duplicate | 0.10019 | 0.08075 | 0.01944 | 0.806 |
| Same-family core | 0.10308 | 0.08379 | 0.01929 | 0.814 |
| Cross-family core | 0.09848 | 0.08839 | 0.01010 | 0.898 |

Duplicate and same-family pairs showed similar attenuation.
Arrival-position effects remain a possible contributor, and pairs
sharing chunks and trajectories are not independent replications.

These findings cover seeds 0–4 within one synthetic setup.
Seed 0 repeats the earlier pilot with more permutation samples.
Validation/test separation for adaptive selection and stopping remains
to be verified. Comparisons with equally sized useful-only and random
subsets under matched training budgets remain outstanding.

Timestamped copies of the outputs are retained under results/.

