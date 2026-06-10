"""
Comparação: projeção v1 (byte linear) vs v3 (bigram hash) × 4 modos de atenção.

Projeções:
  v1 — byte em posição linear, fundo 0.0, trunca em 2304 bytes
  v3 — hash (b1*257+b2) % 2304, acumula 0.5, clamp 2.0 (densidade de bigramas)

Modos de atenção:
  none | sobel | energy | ngram
"""
import importlib.util
import os
import sys
import time
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("psi2", HERE + "/resnet-psi2.py")
psi2 = importlib.util.module_from_spec(spec)
sys.modules["psi2"] = psi2
spec.loader.exec_module(psi2)

CATS = ['sci.space', 'rec.sport.hockey', 'talk.politics.mideast', 'comp.graphics']
MODES = ['none', 'sobel', 'energy', 'ngram']
PROJECTIONS = ['v1', 'v3']
LAMS = {'none': 0.0, 'sobel': 0.03, 'energy': 0.05, 'ngram': 0.02}


def load_balanced(n_train_per_class=60, n_test_per_class=30):
    tr = fetch_20newsgroups(subset='train', categories=CATS,
                            remove=('headers', 'footers', 'quotes'))
    te = fetch_20newsgroups(subset='test', categories=CATS,
                            remove=('headers', 'footers', 'quotes'))

    def balance(ds, n):
        X, y = [], []
        for c in range(len(CATS)):
            idx = [i for i, lbl in enumerate(ds.target) if lbl == c][:n]
            X += [ds.data[i] for i in idx]
            y += [c] * len(idx)
        return X, np.array(y)

    Xtr, ytr = balance(tr, n_train_per_class)
    Xte, yte = balance(te, n_test_per_class)
    return Xtr, ytr, Xte, yte


def run(proj, mode, lam, Xtr, ytr, Xte, yte, bs=16):
    rn = psi2.ResNetPsiText(n_classes=len(CATS), lam=lam, mode=mode, projection=proj)
    t0 = time.time()
    rn.fit(Xtr, ytr, bs=bs)
    preds = rn.predict(Xte, bs=bs)
    elapsed = time.time() - t0
    acc = accuracy_score(yte, preds)

    cmaps = rn.extract(Xte[:bs], bs=bs)
    cov = (cmaps > 0.01).float().mean().item()
    cmax = cmaps.max().item()
    return acc, cov, cmax, elapsed


def main():
    print(f"Device: {psi2.DEVICE}")
    print(f"Field:  {psi2.FIELD_SIZE}x{psi2.FIELD_SIZE}  STIM_TOTAL={psi2.STIM_TOTAL}\n")

    Xtr, ytr, Xte, yte = load_balanced(n_train_per_class=60, n_test_per_class=30)
    print(f"Train: {len(Xtr)}  Test: {len(Xte)}  Classes: {len(CATS)}")
    print(f"Baseline aleatório: {1/len(CATS):.1%}\n")

    results = []
    for proj in PROJECTIONS:
        for mode in MODES:
            lam = LAMS[mode]
            print(f"--- proj={proj} mode={mode} lam={lam} ---")
            acc, cov, cmax, t = run(proj, mode, lam, Xtr, ytr, Xte, yte)
            print(f"    acc={acc:.3f}  cov={cov:.1%}  cmax={cmax:.2f}  t={t:.1f}s\n")
            results.append((proj, mode, lam, acc, cov, cmax, t))

    print("=" * 74)
    print("RESUMO")
    print("=" * 74)
    print(f"{'proj':<5} {'modo':<8} {'lam':>6} {'acc':>7} {'cov':>7} {'cmax':>7} {'tempo':>7}")
    for proj, mode, lam, acc, cov, cmax, t in results:
        print(f"{proj:<5} {mode:<8} {lam:>6.3f} {acc:>7.3f} {cov:>7.1%} {cmax:>7.2f} {t:>6.1f}s")

    print("\nMelhor por projeção:")
    for proj in PROJECTIONS:
        best = max([r for r in results if r[0] == proj], key=lambda r: r[3])
        print(f"  {proj}: mode={best[1]}  acc={best[3]:.3f}")


if __name__ == '__main__':
    main()
