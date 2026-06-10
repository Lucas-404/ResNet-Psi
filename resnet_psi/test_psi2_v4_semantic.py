"""
Teste v4 — Projeção semântica (MiniLM + JL + Top-K) vs baseline v3 (hash).

Hipótese: embedding semântico faz palavras próximas caírem em regiões próximas
do grid, dando ao protótipo euclidiano uma base semântica (não só estatística
de bigramas).

Setup:
  - MiniLM (paraphrase-multilingual-MiniLM-L12-v2, 384-d)
  - Matriz JL 2304×384, seed=42, variância 1/d
  - Top-K esparso (K=50, K=100) — esparsidade permite onda viajar
  - Normalização [0, 2] — compatível com fonte positiva da física

Baseline: v3 + energy + lam=0.05 em 4 classes = 40.8%
Comparação: v4 com K=50 e K=100, mesma atenção.
"""
import importlib.util
import os
import sys
import time
import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("psi2", HERE + "/resnet-psi2.py")
psi2 = importlib.util.module_from_spec(spec)
sys.modules["psi2"] = psi2
spec.loader.exec_module(psi2)

CATS_4 = ['sci.space', 'rec.sport.hockey', 'talk.politics.mideast', 'comp.graphics']
CATS_10 = [
    'sci.space', 'rec.sport.hockey', 'talk.politics.mideast', 'comp.graphics',
    'sci.med', 'rec.autos', 'talk.religion.misc', 'comp.sys.ibm.pc.hardware',
    'misc.forsale', 'alt.atheism',
]


def load_20news(cats, n_train=60, n_test=30):
    tr = fetch_20newsgroups(subset='train', categories=cats,
                            remove=('headers', 'footers', 'quotes'))
    te = fetch_20newsgroups(subset='test', categories=cats,
                            remove=('headers', 'footers', 'quotes'))
    nc = len(tr.target_names)

    def balance(ds, n):
        X, y = [], []
        for c in range(nc):
            idx = [i for i, lbl in enumerate(ds.target) if lbl == c][:n]
            X += [ds.data[i] for i in idx]
            y += [c] * len(idx)
        return X, np.array(y)

    return *balance(tr, n_train), *balance(te, n_test), nc


def run(proj, mode, lam, Xtr, ytr, Xte, yte, nc, bs=16, topk=None):
    if topk is not None:
        psi2.V4_TOPK = topk
    rn = psi2.ResNetPsiText(n_classes=nc, lam=lam, mode=mode, projection=proj)
    t0 = time.time()
    rn.fit(Xtr, ytr, bs=bs)
    preds = rn.predict(Xte, bs=bs)
    elapsed = time.time() - t0
    acc = accuracy_score(yte, preds)

    cmaps = rn.extract(Xte[:bs], bs=bs)
    cov = (cmaps > 0.01).float().mean().item()
    cmax = cmaps.max().item()

    # Distância par-a-par entre protótipos
    protos = torch.stack([p.flatten() for p in rn.prototypes.values()])
    d = torch.cdist(protos.unsqueeze(0), protos.unsqueeze(0)).squeeze(0)
    d = d + torch.eye(len(protos), device=d.device) * 1e9
    return acc, cov, cmax, elapsed, d[d < 1e8].min().item(), d[d < 1e8].mean().item()


def exp_4classes():
    print("\n" + "=" * 78)
    print(" V4 SEMÂNTICO — 4 classes distintas (baseline v3 = 40.8%)")
    print("=" * 78)
    Xtr, ytr, Xte, yte, nc = load_20news(CATS_4, 60, 30)
    print(f"Train {len(Xtr)} | Test {len(Xte)} | classes {nc} | random {1/nc:.1%}\n")

    configs = [
        ('v3', 'energy', 0.05, None, 'v3-energy (baseline)'),
        ('v4', 'none',   0.0,  50,   'v4 K=50 (semântico puro, sem atenção)'),
        ('v4', 'energy', 0.05, 50,   'v4 K=50 + energy'),
        ('v4', 'none',   0.0,  100,  'v4 K=100 (semântico puro, sem atenção)'),
        ('v4', 'energy', 0.05, 100,  'v4 K=100 + energy'),
    ]
    results = []
    for proj, mode, lam, topk, label in configs:
        print(f"--- {label} ---")
        acc, cov, cmax, t, dmin, dmean = run(proj, mode, lam, Xtr, ytr, Xte, yte, nc,
                                              topk=topk)
        print(f"    acc={acc:.3f}  cov={cov:.1%}  cmax={cmax:.2f}  "
              f"dmin={dmin:.3f}  dmean={dmean:.3f}  t={t:.1f}s\n")
        results.append((label, acc, cov, dmin, dmean, t))

    print("-" * 78)
    print(f"{'config':<45} {'acc':>7} {'cov':>7} {'dmin':>7} {'dmean':>7}")
    for label, acc, cov, dmin, dmean, t in results:
        print(f"{label:<45} {acc:>7.3f} {cov:>6.1%} {dmin:>7.2f} {dmean:>7.2f}")
    return results


def exp_10classes(best_config):
    proj, mode, lam, topk, label = best_config
    print("\n" + "=" * 78)
    print(f" V4 ESCALA — 10 classes ({label})")
    print("=" * 78)
    Xtr, ytr, Xte, yte, nc = load_20news(CATS_10, 60, 30)
    print(f"Train {len(Xtr)} | Test {len(Xte)} | classes {nc} | random {1/nc:.1%}\n")
    acc, cov, cmax, t, dmin, dmean = run(proj, mode, lam, Xtr, ytr, Xte, yte, nc,
                                          topk=topk)
    uplift = acc - 1/nc
    print(f"    acc={acc:.3f}  uplift={uplift:+.3f}  cov={cov:.1%}  "
          f"dmin={dmin:.3f}  dmean={dmean:.3f}  t={t:.1f}s")
    return acc, uplift


def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    print(f"Device: {psi2.DEVICE}")
    print(f"FS={psi2.FIELD_SIZE}  STIM_TOTAL={psi2.STIM_TOTAL}")

    r4 = exp_4classes()

    # Pega melhor v4 pra testar em 10 classes
    v4_results = [r for r in r4 if 'v4' in r[0]]
    best = max(v4_results, key=lambda r: r[1])
    print(f"\n[melhor v4 em 4 classes] {best[0]}  acc={best[1]:.3f}")

    # Reconstrói config do melhor
    label = best[0]
    if 'K=50' in label:
        topk = 50
    elif 'K=100' in label:
        topk = 100
    else:
        topk = 50
    mode = 'energy' if 'energy' in label else 'none'
    lam = 0.05 if 'energy' in label else 0.0
    best_config = ('v4', mode, lam, topk, label)

    if best[1] > 0.45:  # só escala se valer a pena
        exp_10classes(best_config)
    else:
        print(f"\n[pulando 10 classes — v4 não superou 45% em 4 classes]")


if __name__ == '__main__':
    main()
