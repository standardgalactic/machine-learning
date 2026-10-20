#!/usr/bin/env python3
"""
Order-aware data attribution by sampled permutations, versus leave-one-out,
on a synthetic world where the value of every training chunk is known by construction.

World
-----
Each "family" f owns a hidden lookup table T_f (a single-cycle permutation of V content tokens).
A sequence is [marker_f, x0, T_f(x0), T_f(T_f(x0)), ...]. A model can only predict the
continuation of family f after it has seen enough transitions of T_f. A training chunk is a
handful of such sequences. Chunk categories planted in the corpus:

  core        distinct chunks from the evaluation families (each covers part of the table)
  duplicate   exact copies of core chunks (redundant: the original already supplies everything)
  irrelevant  chunks from distractor families that are never evaluated (harmless, no value)
  harmful     chunks that use an evaluation family's marker with a corrupted table

Value function (order dependent)
--------------------------------
Fix a warm-started initial model theta_0. For an ordering pi of a chunk set, train sequentially,
K Adam steps per chunk, and let L_k(pi) be the held-out loss on the evaluation families after the
k-th chunk. The marginal of the chunk in position k is  g_k(pi) = L_{k-1}(pi) - L_k(pi).

Permutation attribution:  phi_i = E_pi[ g_{pos_pi(i)}(pi) ],  pi uniform over orderings of all N chunks.
One training run per sampled ordering yields all N marginals, because every prefix is evaluated
along the way. Sum_i phi_i equals the expected total loss reduction (efficiency), so the credit for
a redundant pair is split rather than lost. Optional position centering subtracts the mean marginal at
each position (variance reduction; it shifts all phi_i by one constant, so rankings are unchanged in
expectation).

Leave-one-out baseline:  loo_i = L_N(pi_ref without i) - L_N(pi_ref), one fixed reference ordering.

Run:  python ordered_attribution.py            (roughly 20 to 30 minutes; the model is tiny, so speed is bound by kernel launches rather than VRAM)
"""
import argparse, csv, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

CAT_NAMES = ["core", "duplicate", "irrelevant", "harmful"]


# ----------------------------------------------------------------- synthetic world
def cyclic_table(V, rng):
    p = rng.permutation(V)
    T = np.empty(V, dtype=np.int64)
    T[p] = np.roll(p, -1)
    return T


def corrupt(T, rng, frac):
    idx = rng.choice(len(T), max(2, int(round(frac * len(T)))), replace=False)
    T2 = T.copy()
    T2[idx] = np.roll(T[idx], 1)  # targets are distinct, so every selected entry changes
    return T2


def walks(marker, T, n, L, V, rng):
    s = np.empty((n, L), dtype=np.int64)
    s[:, 0] = V + marker
    s[:, 1] = rng.integers(0, V, n)
    for t in range(2, L):
        s[:, t] = T[s[:, t - 1]]
    return s


def build_world(a, rng):
    V, L, spc = a.vocab, a.seqlen, a.seqs_per_chunk
    nf = a.eval_fams + a.distract_fams + a.warm_fams
    tables = [cyclic_table(V, rng) for _ in range(nf)]
    ev = list(range(a.eval_fams))
    di = list(range(a.eval_fams, a.eval_fams + a.distract_fams))
    wa = list(range(a.eval_fams + a.distract_fams, nf))
    chunks, cats, fams = [], [], []
    core_ids = []
    for f in ev:
        for _ in range(a.core_per_fam):
            core_ids.append(len(chunks))
            chunks.append(walks(f, tables[f], spc, L, V, rng)); cats.append(0); fams.append(f)
    src = rng.choice(core_ids, a.dups, replace=a.dups > len(core_ids)) if a.dups else []
    for j in src:
        chunks.append(chunks[int(j)].copy()); cats.append(1); fams.append(fams[int(j)])
    for _ in range(a.irrel):
        f = int(rng.choice(di))
        chunks.append(walks(f, tables[f], spc, L, V, rng)); cats.append(2); fams.append(f)
    for _ in range(a.harm):
        f = int(rng.choice(ev))
        chunks.append(walks(f, corrupt(tables[f], rng, a.corrupt_frac), spc, L, V, rng)); cats.append(3); fams.append(f)

    def evalset():
        return np.concatenate([walks(f, tables[f], a.eval_per_fam, L, V, rng) for f in ev])

    warm = np.concatenate([walks(f, tables[f], 256, L, V, rng) for f in wa])
    return dict(chunks=chunks, cats=cats, fams=fams, attr=evalset(), test=evalset(), warm=warm)


# ----------------------------------------------------------------- model and training
class TinyLM(nn.Module):
    def __init__(s, vocab_total, L, d, layers, heads):
        super().__init__()
        s.tok = nn.Embedding(vocab_total, d)
        s.pos = nn.Embedding(L, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        s.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        s.ln = nn.LayerNorm(d)
        s.head = nn.Linear(d, vocab_total, bias=False)
        s.register_buffer("mask", torch.triu(torch.ones(L, L, dtype=torch.bool), 1))

    def forward(s, x):
        h = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device))
        return s.head(s.ln(s.enc(h, mask=s.mask)))


def lm_loss(model, x, V):
    # position 1 (the random start token) is unpredictable, so score positions 2..L-1 over content tokens
    lg = model(x)[:, 1:-1, :V]
    return F.cross_entropy(lg.reshape(-1, V), x[:, 2:].reshape(-1))


@torch.no_grad()
def evaluate(model, x, V):
    return lm_loss(model, x, V).item()


class Lab:
    def __init__(self, a, dev, rng):
        self.a, self.dev, self.V = a, dev, a.vocab
        w = build_world(a, rng)
        T = lambda z: torch.as_tensor(z, device=dev)
        self.chunks = [T(c) for c in w["chunks"]]
        self.cats, self.fams = np.array(w["cats"]), np.array(w["fams"])
        self.attr, self.test = T(w["attr"]), T(w["test"])
        self.N = len(self.chunks)
        nf = a.eval_fams + a.distract_fams + a.warm_fams
        self.model = TinyLM(a.vocab + nf, a.seqlen, a.dim, a.layers, a.heads).to(dev)
        # Warm-up learns the sequence format only: every batch uses fresh random tables, so no marker-to-table
        # association can be memorized and the model starts near the uniform loss on every real family.
        opt = torch.optim.Adam(self.model.parameters(), lr=a.lr)
        B, V, L = 128, a.vocab, a.seqlen
        wmark = V + a.eval_fams + a.distract_fams + torch.randint(0, a.warm_fams, (a.warm_steps, B), device=dev)
        for step in range(a.warm_steps):
            perm = torch.argsort(torch.rand(B, V, device=dev), dim=1)
            tab = torch.empty(B, V, dtype=torch.long, device=dev)
            tab.scatter_(1, perm, torch.roll(perm, -1, dims=1))
            x = torch.empty(B, L, dtype=torch.long, device=dev)
            x[:, 0] = wmark[step]
            x[:, 1] = torch.randint(0, V, (B,), device=dev)
            for t in range(2, L):
                x[:, t] = tab.gather(1, x[:, t - 1:t]).squeeze(1)
            opt.zero_grad(set_to_none=True)
            lm_loss(self.model, x, V).backward()
            opt.step()
        self.base = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.n_runs = 0

    def run(self, order, trace=False):
        """Train sequentially on the chunks in `order`. Returns (trace or None, final attr loss, final test loss)."""
        self.n_runs += 1
        m = self.model
        m.load_state_dict(self.base)
        opt = torch.optim.Adam(m.parameters(), lr=self.a.lr, fused=self.dev.type == "cuda")
        tr = [evaluate(m, self.attr, self.V)] if trace else None
        for i in order:
            x = self.chunks[int(i)]
            for _ in range(self.a.steps):
                opt.zero_grad(set_to_none=True)
                lm_loss(m, x, self.V).backward()
                opt.step()
            if trace:
                tr.append(evaluate(m, self.attr, self.V))
        return tr, evaluate(m, self.attr, self.V), evaluate(m, self.test, self.V)


# ----------------------------------------------------------------- attribution methods
def perm_marginals(lab, P, rng, log_every=20):
    N = lab.N
    M, POS = np.zeros((P, N)), np.zeros((P, N), dtype=int)
    t0 = time.time()
    for p in range(P):
        order = rng.permutation(N)
        tr = lab.run(order, trace=True)[0]
        M[p, order] = -np.diff(np.array(tr))
        POS[p, order] = np.arange(N)
        if (p + 1) % log_every == 0:
            print(f"  permutation {p + 1}/{P}  ({time.time() - t0:.0f}s)", flush=True)
    return M, POS


def center_by_position(M, POS):
    base = np.array([M[POS == k].mean() for k in range(M.shape[1])])
    return M - base[POS] + base.mean()  # constant added back so the values still sum to the expected total gain


def loo_values(lab, order):
    full = lab.run(order)[1]
    vals = np.zeros(lab.N)
    for i in range(lab.N):
        vals[i] = lab.run([j for j in order if j != i])[1] - full
    return vals


# ----------------------------------------------------------------- metrics
def _rank(x):
    try:
        from scipy.stats import rankdata
        return rankdata(x)
    except Exception:
        o = np.argsort(x, kind="stable")
        r = np.empty(len(x)); r[o] = np.arange(len(x))
        return r


def spearman(x, y):
    return float(np.corrcoef(_rank(np.asarray(x)), _rank(np.asarray(y)))[0, 1])


def auroc(hi, lo):
    h, l = np.asarray(hi)[:, None], np.asarray(lo)[None, :]
    return float(((h > l).sum() + 0.5 * (h == l).sum()) / (h.size * l.size))


def category_report(cats, scores):
    out = {}
    for c, name in enumerate(CAT_NAMES):
        s = scores[cats == c]
        out[name] = (float(s.mean()), float(s.std())) if len(s) else (float("nan"), float("nan"))
    return out


def pair_aurocs(cats, s):
    g = lambda c: s[cats == c]
    junk = np.concatenate([g(2), g(3)])
    useful = np.concatenate([g(0), g(1)])
    return {
        "core > irrelevant": auroc(g(0), g(2)),
        "duplicate > irrelevant": auroc(g(1), g(2)),
        "irrelevant > harmful": auroc(g(2), g(3)),
        "useful > junk": auroc(useful, junk),
    }


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
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
    ap.add_argument("--corrupt_frac", type=float, default=1.0, help="fraction of table entries rewritten in harmful chunks")
    ap.add_argument("--warm_steps", type=int, default=300)
    ap.add_argument("--perms", type=int, default=220)
    ap.add_argument("--loo_orders", type=int, default=2)
    ap.add_argument("--drop_ks", type=str, default="12,24,36")
    ap.add_argument("--eval_orders", type=int, default=3)
    ap.add_argument("--random_draws", type=int, default=3)
    ap.add_argument("--no_center", action="store_true")
    ap.add_argument("--out", type=str, default="ordered_attribution_out")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    print(f"device: {dev}")
    lab = Lab(a, dev, rng)
    N, cats = lab.N, lab.cats
    counts = {n: int((cats == c).sum()) for c, n in enumerate(CAT_NAMES)}
    print(f"chunks: N={N} {counts}")
    init_loss = evaluate(lab.model, lab.attr, lab.V)
    full = lab.run(rng.permutation(N))
    print(f"held-out loss: warm start {init_loss:.3f}, after all chunks (one random order) {full[2]:.3f}, "
          f"uniform guess {np.log(lab.V):.3f}")

    # ---- leave-one-out under several reference orderings
    loo = []
    for r in range(a.loo_orders):
        print(f"leave-one-out, reference order {r + 1}/{a.loo_orders} ({N + 1} runs)")
        loo.append(loo_values(lab, rng.permutation(N)))

    # ---- sampled-permutation attribution
    print(f"permutation attribution ({a.perms} runs)")
    M, POS = perm_marginals(lab, a.perms, rng)
    Mv = M if a.no_center else center_by_position(M, POS)
    perm_full = Mv.mean(0)
    perm_se = Mv.std(0, ddof=1) / np.sqrt(a.perms)
    matched = N + 1
    perm_matched = Mv[:matched].mean(0)

    # ---- reliability at matched budget (N+1 runs per estimate)
    rel = {}
    if a.loo_orders >= 2:
        rel["loo order A vs order B"] = spearman(loo[0], loo[1])
    if a.perms >= 2 * matched:
        rel["perm block 1 vs block 2"] = spearman(Mv[:matched].mean(0), Mv[matched:2 * matched].mean(0))

    methods = {"loo": loo[0], "perm_matched": perm_matched, "perm_full": perm_full}
    cat_rep = {k: category_report(cats, v) for k, v in methods.items()}
    aur = {k: pair_aurocs(cats, v) for k, v in methods.items()}

    # ---- drop experiment: remove the k lowest-valued chunks, retrain, score on the untouched test split
    ks = [int(k) for k in a.drop_ks.split(",") if k.strip() and int(k) < N]
    ref_orders = [rng.permutation(N) for _ in range(a.eval_orders)]

    def retrain(keep_idx):
        keep = set(int(i) for i in keep_idx)
        return float(np.mean([lab.run([j for j in ro if j in keep])[2] for ro in ref_orders]))

    prio = np.array([3, 2, 1, 0])[cats] + rng.random(N) * 0.1  # oracle: harmful first, then irrelevant, then duplicate
    drop = {"keep all": {0: retrain(range(N))}}
    for name, sc in list(methods.items()) + [("oracle", prio)]:
        drop[name] = {}
        for k in ks:
            drop[name][k] = retrain(np.argsort(-sc)[: N - k])
    drop["random"] = {}
    for k in ks:
        drop["random"][k] = float(np.mean([retrain(rng.permutation(N)[: N - k]) for _ in range(a.random_draws)]))

    # ---- report
    print("\nmean value by planted category (loss reduction on held-out evaluation families; sd in brackets)")
    print(f"{'method':14s}" + "".join(f"{n:>22s}" for n in CAT_NAMES))
    for k, rep in cat_rep.items():
        print(f"{k:14s}" + "".join(f"{rep[n][0]:+13.5f} ({rep[n][1]:.5f})" for n in CAT_NAMES))
    print("\nAUROC (1.0 means the planted ordering is recovered perfectly, 0.5 is chance)")
    names = list(next(iter(aur.values())).keys())
    print(f"{'method':14s}" + "".join(f"{n:>26s}" for n in names))
    for k, d in aur.items():
        print(f"{k:14s}" + "".join(f"{d[n]:26.3f}" for n in names))
    print("\nrank reliability at matched budget (Spearman)")
    for k, v in rel.items():
        print(f"  {k}: {v:.3f}")
    print(f"\ntest loss after dropping the k lowest-valued chunks (lower is better), baseline {drop['keep all'][0]:.4f}")
    print(f"{'method':14s}" + "".join(f"{'k=' + str(k):>10s}" for k in ks))
    for name in ["loo", "perm_matched", "perm_full", "random", "oracle"]:
        print(f"{name:14s}" + "".join(f"{drop[name][k]:10.4f}" for k in ks))
    print(f"\ntotal training runs: {lab.n_runs}")

    # ---- save
    with open(os.path.join(a.out, "values.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk", "category", "family", "loo_A"] + (["loo_B"] if len(loo) > 1 else [])
                   + ["perm_matched", "perm_full", "perm_se"])
        for i in range(N):
            w.writerow([i, CAT_NAMES[cats[i]], int(lab.fams[i]), loo[0][i]] + ([loo[1][i]] if len(loo) > 1 else [])
                       + [perm_matched[i], perm_full[i], perm_se[i]])
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(dict(args=vars(a), counts=counts, category_means=cat_rep, auroc=aur, reliability=rel,
                       drop={k: {str(kk): vv for kk, vv in v.items()} for k, v in drop.items()}), f, indent=2)
    print(f"saved values.csv and results.json to {a.out}/")


if __name__ == "__main__":
    main()
