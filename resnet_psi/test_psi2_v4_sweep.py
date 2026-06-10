"""
Sweep K + escala de classes para v4 semântico.

Fase 1 — K sweep em 4 classes: K ∈ {10, 20, 30, 50, 75, 100, 200}
Fase 2 — Escala com melhor K: 4, 10, 20 classes

Sem atenção (v4 K=50 mostrou que Energy empata — poupa tempo).
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
CATS_10 = CATS_4 + [
    'sci.med', 'rec.autos', 'talk.religion.misc', 'comp.sys.ibm.pc.hardware',
    'misc.forsale', 'alt.atheism',
]
CATS_20 = None  # None = todas


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

    Xtr, ytr = balance(tr, n_train)
    Xte, yte = balance(te, n_test)
    return Xtr, ytr, Xte, yte, nc


def run(proj, mode, lam, Xtr, ytr, Xte, yte, nc, bs=16, topk=None):
    if topk is not None:
        psi2.V4_TOPK = topk
    rn = psi2.ResNetPsiText(n_classes=nc, lam=lam, mode=mode, projection=proj)
    t0 = time.time()
    rn.fit(Xtr, ytr, bs=bs)
    preds = rn.predict(Xte, bs=bs)
    elapsed = time.time() - t0
    acc = accuracy_score(yte, preds)

    protos = torch.stack([p.flatten() for p in rn.prototypes.values()])
    d = torch.cdist(protos.unsqueeze(0), protos.unsqueeze(0)).squeeze(0)
    d = d + torch.eye(len(protos), device=d.device) * 1e9
    dmin = d[d < 1e8].min().item()
    dmean = d[d < 1e8].mean().item()
    return acc, elapsed, dmin, dmean


def sweep_k():
    print("\n" + "=" * 74)
    print(" FASE 1 — K sweep em 4 classes (v4, sem atenção)")
    print("=" * 74)
    Xtr, ytr, Xte, yte, nc = load_20news(CATS_4, 60, 30)
    print(f"Train {len(Xtr)} | Test {len(Xte)}\n")

    results = []
    for k in [10, 20, 30, 50, 75, 100, 200]:
        print(f"--- K={k} ---")
        acc, t, dmin, dmean = run('v4', 'none', 0.0, Xtr, ytr, Xte, yte, nc, topk=k)
        print(f"    acc={acc:.3f}  dmin={dmin:.2f}  dmean={dmean:.2f}  t={t:.1f}s")
        results.append((k, acc, dmin, dmean, t))

    print("\n" + "-" * 74)
    print(f"{'K':>5} {'acc':>7} {'dmin':>8} {'dmean':>8} {'ratio':>7} {'tempo':>8}")
    for k, acc, dmin, dmean, t in results:
        ratio = dmin / dmean  # quanto mais próximo de 1, mais uniforme
        print(f"{k:>5} {acc:>7.3f} {dmin:>8.2f} {dmean:>8.2f} {ratio:>7.3f} {t:>7.1f}s")
    return results


def scale_classes(best_k):
    print("\n" + "=" * 74)
    print(f" FASE 2 — Escala (K={best_k}, v4 sem atenção)")
    print("=" * 74)

    setups = [('4', CATS_4), ('10', CATS_10), ('20', CATS_20)]
    results = []
    for name, cats in setups:
        Xtr, ytr, Xte, yte, nc = load_20news(cats, 60, 30)
        print(f"\n--- {name} classes  (random {1/nc:.1%}) ---")
        acc, t, dmin, dmean = run('v4', 'none', 0.0, Xtr, ytr, Xte, yte, nc, topk=best_k)
        uplift = acc - 1/nc
        print(f"    acc={acc:.3f}  uplift={uplift:+.3f}  "
              f"dmin={dmin:.2f}  dmean={dmean:.2f}  t={t:.1f}s")
        results.append((name, nc, acc, uplift, dmin, dmean, t))

    print("\n" + "-" * 74)
    print(f"{'nC':<4} {'acc':>7} {'rand':>7} {'uplift':>8} "
          f"{'dmin':>8} {'dmean':>8} {'tempo':>8}")
    for name, nc, acc, up, dmin, dmean, t in results:
        print(f"{name:<4} {acc:>7.3f} {1/nc:>7.3f} {up:>+8.3f} "
              f"{dmin:>8.2f} {dmean:>8.2f} {t:>7.1f}s")
    return results


def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    print(f"Device: {psi2.DEVICE}  FS={psi2.FIELD_SIZE}")

    r1 = sweep_k()
    best_k = max(r1, key=lambda r: r[1])[0]
    print(f"\n[melhor K] {best_k}")

    r2 = scale_classes(best_k)

    print("\n" + "=" * 74)
    print(" RESUMO")
    print("=" * 74)
    print(f"  Melhor K : {best_k}")
    print(f"  Escala   : " + " | ".join(f"{nc}c:{acc:.3f}"
                                          for _, nc, acc, *_ in r2))


if __name__ == '__main__':
    main()
