#!/usr/bin/env python3
"""
Position-dependent marginals and prefix-aware ordering, built on ordered_attribution.py.

The permutation runs in ordered_attribution.py average each chunk's marginal contribution over
every position at which the chunk can arrive. This script keeps the position instead of averaging
it away, and uses it to ask two questions on the same synthetic world.

First, do the position curves carry information that a single scalar value does not? A redundant
chunk should contribute when it arrives before its twin and almost nothing when it arrives after.
For exact duplicate pairs the script therefore compares the marginal of whichever member arrives
first with the marginal of whichever arrives second, over all sampled orderings. Two control groups
are measured the same way: pairs of core chunks from the same evaluation family, which overlap
partially, and pairs of core chunks from different families, which should not overlap. The
cross-family gap measures the general decay of marginals over training, so a duplicate gap well
above it is evidence of redundancy detection and not just of position.

Second, does choosing the order of training from conditional marginals beat choosing it from scalar
values? A greedy procedure picks, at each step, the remaining chunk with the largest loss reduction
given the model as it currently stands, and the run can stop at the first step where no remaining
chunk helps. Selection uses the attribution split and scoring uses the untouched test split. The
procedure costs about N squared over two short training bursts, so it is a reference for what
prefix-aware selection can reach, not an amortized method.

Orderings scored by test loss after sequential training (lower is better): random, scalar_desc
(permutation values, highest first), scalar_asc (the reverse, as a sanity check), early_desc
(highest first-bin marginal first, which ignores the prefix), greedy, greedy_stop (greedy prefix up
to the first non-positive gain), category_oracle (core, duplicate, irrelevant, harmful), and
core_only (the core chunks alone in random order).

Seeds replicate the whole world, warm start, sampled orderings and orderings scored. Defaults for the
world and model match ordered_attribution.py, so seed 0 reproduces its world.

Run: python curriculum_experiment.py --seeds 5
Each seed takes roughly the permutation time of ordered_attribution.py plus a few minutes for the
greedy stage. Use --perms to trade accuracy for time.
"""

import argparse, copy, json, os, time, warnings

import numpy as np
import torch

from ordered_attribution import CAT_NAMES, Lab, center_by_position, evaluate, lm_loss, perm_marginals


# ----------------------------------------------------------------- position curves

def position_curves(M, POS, bins):
    """Mean marginal of each chunk within each position bin. Shape (N, bins); nan where a bin was never sampled."""
    P, N = M.shape
    which = np.minimum((POS * bins) // N, bins - 1)
    curves = np.full((N, bins), np.nan)
    for b in range(bins):
        mask = which == b
        cnt = mask.sum(0)
        tot = (M * mask).sum(0)
        ok = cnt > 0
        curves[ok, b] = tot[ok] / cnt[ok]
    return curves


def category_curves(cats, curves):
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for c, name in enumerate(CAT_NAMES):
            sel = curves[cats == c]
            out[name] = np.nanmean(sel, axis=0).tolist() if len(sel) else []
    return out


def duplicate_pairs(lab):
    """Pair each duplicate chunk with the core chunk it copies, found by exact content equality."""
    core = [i for i in range(lab.N) if lab.cats[i] == 0]
    pairs = []
    for d in (i for i in range(lab.N) if lab.cats[i] == 1):
        for c in core:
            if torch.equal(lab.chunks[d], lab.chunks[c]):
                pairs.append((c, d))
                break
    return pairs


def arrival_stats(M, POS, pairs):
    """For each pair, the mean marginal of the member that arrived first and of the member that arrived second."""
    if not pairs:
        return dict(n_pairs=0, first=float("nan"), second=float("nan"), gap=float("nan"), retained=float("nan"))
    f, s = [], []
    for a, b in pairs:
        a_first = POS[:, a] < POS[:, b]
        f.append(np.where(a_first, M[:, a], M[:, b]).mean())
        s.append(np.where(a_first, M[:, b], M[:, a]).mean())
    f, s = float(np.mean(f)), float(np.mean(s))
    return dict(n_pairs=len(pairs), first=f, second=s, gap=f - s, retained=(s / f if f > 0 else float("nan")))


def control_pairs(lab):
    core = [i for i in range(lab.N) if lab.cats[i] == 0]
    same = [(a, b) for a in core for b in core if a < b and lab.fams[a] == lab.fams[b]]
    cross = [(a, b) for a in core for b in core if a < b and lab.fams[a] != lab.fams[b]]
    return same, cross


# ----------------------------------------------------------------- greedy conditional ordering

def _snapshot(m, opt):
    return {k: v.clone() for k, v in m.state_dict().items()}, copy.deepcopy(opt.state_dict())


def _restore(m, opt, snap):
    m.load_state_dict(snap[0])
    opt.load_state_dict(copy.deepcopy(snap[1]))


def greedy_order(lab, log_every=10):
    """Forward selection by true conditional gain on the attribution split.

    At every step each remaining chunk is trained on for the usual number of Adam steps starting from
    the current model and optimizer state, and the chunk giving the lowest attribution loss is kept.
    Returns the full order and the gain realized at every step."""
    m, V, K = lab.model, lab.V, lab.a.steps
    m.load_state_dict(lab.base)
    opt = torch.optim.Adam(m.parameters(), lr=lab.a.lr, fused=lab.dev.type == "cuda")

    def train(i):
        x = lab.chunks[i]
        for _ in range(K):
            opt.zero_grad(set_to_none=True)
            lm_loss(m, x, V).backward()
            opt.step()

    remaining = list(range(lab.N))
    order, gains = [], []
    cur = evaluate(m, lab.attr, V)
    t0 = time.time()
    while remaining:
        snap = _snapshot(m, opt)
        trial = {}
        for i in remaining:
            train(i)
            trial[i] = evaluate(m, lab.attr, V)
            _restore(m, opt, snap)
        best = min(trial, key=trial.get)
        train(best)
        gains.append(cur - trial[best])
        cur = trial[best]
        order.append(best)
        remaining.remove(best)
        if len(order) % log_every == 0:
            print(f"  greedy step {len(order)}/{lab.N} ({time.time() - t0:.0f}s)", flush=True)
    return order, gains


# ----------------------------------------------------------------- one seed

def run_seed(a, seed, dev):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    lab = Lab(a, dev, rng)
    N, cats = lab.N, lab.cats
    counts = {n: int((cats == c).sum()) for c, n in enumerate(CAT_NAMES)}
    print(f"\n=== seed {seed}: N={N} {counts}", flush=True)

    t0 = time.time()
    print(f"permutation runs ({a.perms})", flush=True)
    M, POS = perm_marginals(lab, a.perms, rng)
    values = center_by_position(M, POS).mean(0)
    curves = position_curves(M, POS, a.bins)  # raw marginals: centering would remove the position effect
    perm_time = time.time() - t0

    pairs = duplicate_pairs(lab)
    same, cross = control_pairs(lab)
    arrival = {
        "duplicate": arrival_stats(M, POS, pairs),
        "same_family_core": arrival_stats(M, POS, same),
        "cross_family_core": arrival_stats(M, POS, cross),
    }

    def score(order):
        return float(lab.run([int(i) for i in order])[2])

    def random_mean(pool):
        return float(np.mean([score(rng.permutation(pool)) for _ in range(a.random_draws)]))

    orderings = {"random": random_mean(np.arange(N))}
    orderings["scalar_desc"] = score(np.argsort(-values))
    orderings["scalar_asc"] = score(np.argsort(values))
    orderings["early_desc"] = score(np.argsort(-np.nan_to_num(curves[:, 0], nan=-np.inf)))
    orderings["category_oracle"] = score(np.argsort(cats, kind="stable"))
    core_ids = np.where(cats == 0)[0]
    orderings["core_only"] = random_mean(core_ids)

    greedy = None
    if not a.skip_greedy:
        t1 = time.time()
        print("greedy conditional ordering", flush=True)
        order, gains = greedy_order(lab)
        stop = next((k for k, g in enumerate(gains) if g <= a.stop_eps), N)
        kept = order[:stop]
        orderings["greedy"] = score(order)
        orderings["greedy_stop"] = score(kept)
        greedy = dict(
            order=[int(i) for i in order],
            order_categories=[CAT_NAMES[int(cats[i])] for i in order],
            gains=[float(g) for g in gains],
            n_kept=int(stop),
            kept_composition={n: int(sum(cats[i] == c for i in kept)) for c, n in enumerate(CAT_NAMES)},
            seconds=time.time() - t1,
        )

    res = dict(
        seed=seed, counts=counts, perm_seconds=perm_time, orderings=orderings, arrival=arrival,
        curves_by_category=category_curves(cats, curves), greedy=greedy,
        values_by_category={n: float(values[cats == c].mean()) for c, n in enumerate(CAT_NAMES)},
    )

    d = os.path.join(a.out, f"seed_{seed}")
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "marginals.npz"), M=M, POS=POS, cats=cats, fams=lab.fams, values=values, curves=curves)
    with open(os.path.join(d, "results.json"), "w") as f:
        json.dump(dict(args=vars(a), **res), f, indent=2)
    report_seed(res, a.bins)
    return res


# ----------------------------------------------------------------- reporting

def report_seed(r, bins):
    print("\nmean marginal by position bin (loss reduction; first bin is the start of training)")
    print(f"{'category':12s}" + "".join(f"{'bin ' + str(b + 1):>11s}" for b in range(bins)))
    for n, cv in r["curves_by_category"].items():
        if cv:
            print(f"{n:12s}" + "".join(f"{v:+11.5f}" for v in cv))
    print("\narrival analysis: marginal of the member that arrives first versus second")
    print(f"{'pair type':20s}{'pairs':>7s}{'first':>12s}{'second':>12s}{'gap':>12s}{'retained':>10s}")
    for k, s in r["arrival"].items():
        print(f"{k:20s}{s['n_pairs']:7d}{s['first']:+12.5f}{s['second']:+12.5f}{s['gap']:+12.5f}{s['retained']:10.3f}")
    print("\ntest loss after sequential training in the given order (lower is better)")
    for k, v in r["orderings"].items():
        print(f"  {k:16s}{v:8.4f}")
    g = r["greedy"]
    if g:
        print(f"\ngreedy stop kept {g['n_kept']} chunks: {g['kept_composition']}")


def aggregate(results):
    names = list(results[0]["orderings"].keys())
    rand = np.array([r["orderings"]["random"] for r in results])
    agg = {"orderings": {}, "arrival": {}}
    for n in names:
        v = np.array([r["orderings"][n] for r in results])
        dlt = v - rand
        agg["orderings"][n] = dict(
            mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
            vs_random=float(dlt.mean()), seeds_below_random=int((dlt < 0).sum()), n_seeds=len(v))
    for k in results[0]["arrival"]:
        agg["arrival"][k] = {
            q: float(np.nanmean([r["arrival"][k][q] for r in results])) for q in ("first", "second", "gap", "retained")}
    return agg


def report_aggregate(agg):
    n = next(iter(agg["orderings"].values()))["n_seeds"]
    print(f"\n=== across {n} seed(s)")
    print(f"{'ordering':18s}{'mean':>9s}{'sd':>9s}{'vs random':>11s}{'seeds below random':>20s}")
    for k, d in agg["orderings"].items():
        print(f"{k:18s}{d['mean']:9.4f}{d['sd']:9.4f}{d['vs_random']:+11.4f}{d['seeds_below_random']:>12d}/{d['n_seeds']}")
    print(f"\n{'pair type':20s}{'first':>12s}{'second':>12s}{'gap':>12s}{'retained':>10s}")
    for k, d in agg["arrival"].items():
        print(f"{k:20s}{d['first']:+12.5f}{d['second']:+12.5f}{d['gap']:+12.5f}{d['retained']:10.3f}")


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Position curves and greedy conditional ordering on the ordered_attribution world.")
    ap.add_argument("--seeds", type=int, default=1, help="number of independent worlds")
    ap.add_argument("--seed0", type=int, default=0, help="first seed; seed 0 reproduces the ordered_attribution.py world")
    ap.add_argument("--perms", type=int, default=220)
    ap.add_argument("--bins", type=int, default=6, help="position bins for the curves")
    ap.add_argument("--random_draws", type=int, default=5)
    ap.add_argument("--stop_eps", type=float, default=0.0, help="greedy_stop halts at the first gain at or below this")
    ap.add_argument("--skip_greedy", action="store_true")
    ap.add_argument("--out", type=str, default="curriculum_out")
    # world and model, defaults identical to ordered_attribution.py
    ap.add_argument("--vocab", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=9)
    ap.add_argument("--seqs_per_chunk", type=int, default=8)
    ap.add_argument("--eval_fams", type=int, default=6)
    ap.add_argument("--distract_fams", type=int, default=3)
    ap.add_argument("--warm_fams", type=int, default=3)
    ap.add_argument("--core_per_fam", type=int, default=3)
    ap.add_argument("--dups", type=int, default=12)
    ap.add_argument("--irrel", type=int, default=12)
    ap.add_argument("--harm", type=int, default=12)
    ap.add_argument("--eval_per_fam", type=int, default=64)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=10, help="Adam steps per chunk")
    ap.add_argument("--corrupt_frac", type=float, default=1.0)
    ap.add_argument("--warm_steps", type=int, default=300)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")

    results = []
    for s in range(a.seed0, a.seed0 + a.seeds):
        results.append(run_seed(a, s, dev))
        agg = aggregate(results)
        with open(os.path.join(a.out, "summary.json"), "w") as f:
            json.dump(dict(args=vars(a), seeds=[r["seed"] for r in results], aggregate=agg), f, indent=2)
    report_aggregate(aggregate(results))
    print(f"\nsaved per-seed results and summary.json to {a.out}/")


if __name__ == "__main__":
    main()
