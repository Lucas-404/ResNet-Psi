"""
Três experimentos sobre ResNet-Psi texto (v3 + energy sem gate, baseline 40.8% em 4 classes).

Exp 1 — Pooling espacial no crystal_map antes da classificação
    modos: none, avg_k2, avg_k3, max_k2, max_k3
    pergunta: suavizar picos tolera variações morfológicas entre textos da mesma classe?

Exp 2 — Expansão do grid (FIELD_SIZE)
    fs: 48, 64
    pergunta: mais slots de hash (menos colisão) vs dissipação da onda?

Exp 3 — Escalar número de classes (4 → 10 → 20)
    pergunta: degradação linear ou colapso por saturação do espaço 2D?
    também mede distância par-a-par entre protótipos pra diagnosticar.

Cada exp usa a melhor config dos anteriores quando possível.

Config base: v3 + energy, lam=0.05, seed 42.
"""
import importlib.util
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("psi2", HERE + "/resnet-psi2.py")
psi2 = importlib.util.module_from_spec(spec)
sys.modules["psi2"] = psi2
spec.loader.exec_module(psi2)


# ──────────────────────────────────────────────────────────────────────
# Pooling customizado — aplicado no crystal_map antes da classificação
# ──────────────────────────────────────────────────────────────────────
def apply_pool(cmaps, kind='none', k=2):
    """cmaps: (N, FS, FS). Retorna (N, FS', FS')."""
    if kind == 'none':
        return cmaps
    x = cmaps.unsqueeze(1)  # (N, 1, FS, FS)
    if kind == 'avg':
        x = F.avg_pool2d(x, kernel_size=k, stride=k)
    elif kind == 'max':
        x = F.max_pool2d(x, kernel_size=k, stride=k)
    else:
        raise ValueError(kind)
    return x.squeeze(1)


class ResNetPsiPool(psi2.ResNetPsiText):
    """ResNetPsiText com pooling aplicado no crystal_map antes de agregar/classificar."""

    def __init__(self, n_classes, lam=None, mode='energy', projection='v3',
                 pool_kind='none', pool_k=2):
        super().__init__(n_classes, lam=lam, mode=mode, projection=projection)
        self.pool_kind = pool_kind
        self.pool_k = pool_k

    def fit(self, train_texts, train_labels, bs=64):
        cmaps = self.extract(train_texts, bs, verbose=True)
        cmaps = apply_pool(cmaps, self.pool_kind, self.pool_k)
        labels_np = np.array(train_labels)
        for cls in range(self.n_classes):
            mask = (labels_np == cls)
            if mask.any():
                self.prototypes[cls] = cmaps[mask].mean(dim=0)
        return self

    def predict(self, test_texts, bs=64):
        cmaps = self.extract(test_texts, bs)
        cmaps = apply_pool(cmaps, self.pool_kind, self.pool_k)
        return psi2.classify_euclidean(cmaps, self.prototypes)


# ──────────────────────────────────────────────────────────────────────
# Dataset loading (4/10/20 classes)
# ──────────────────────────────────────────────────────────────────────
CATS_4 = ['sci.space', 'rec.sport.hockey', 'talk.politics.mideast', 'comp.graphics']
CATS_10 = [
    'sci.space', 'rec.sport.hockey', 'talk.politics.mideast', 'comp.graphics',
    'sci.med', 'rec.autos', 'talk.religion.misc', 'comp.sys.ibm.pc.hardware',
    'misc.forsale', 'alt.atheism',
]
CATS_20 = None  # None = todas


def load_20news(cats, n_train=60, n_test=30):
    tr = fetch_20newsgroups(subset='train', categories=cats,
                            remove=('headers', 'footers', 'quotes'))
    te = fetch_20newsgroups(subset='test', categories=cats,
                            remove=('headers', 'footers', 'quotes'))
    n_classes = len(tr.target_names)

    def balance(ds, n):
        X, y = [], []
        for c in range(n_classes):
            idx = [i for i, lbl in enumerate(ds.target) if lbl == c][:n]
            X += [ds.data[i] for i in idx]
            y += [c] * len(idx)
        return X, np.array(y)

    Xtr, ytr = balance(tr, n_train)
    Xte, yte = balance(te, n_test)
    return Xtr, ytr, Xte, yte, n_classes


# ──────────────────────────────────────────────────────────────────────
# Diagnóstico: distância par-a-par entre protótipos
# ──────────────────────────────────────────────────────────────────────
def proto_stats(prototypes):
    """Retorna dist min/mean entre protótipos e norma média."""
    protos = torch.stack([p.flatten() for p in prototypes.values()])
    n = protos.shape[0]
    d = torch.cdist(protos.unsqueeze(0), protos.unsqueeze(0)).squeeze(0)
    d = d + torch.eye(n, device=d.device) * 1e9  # mascara diagonal
    norms = protos.norm(dim=1).mean().item()
    return {'dmin': d.min().item(), 'dmean_off': d[d < 1e8].mean().item(), 'norm': norms}


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────
def run_one(Xtr, ytr, Xte, yte, n_classes, *,
            pool_kind='none', pool_k=2, lam=0.05, mode='energy', projection='v3', bs=16):
    rn = ResNetPsiPool(n_classes=n_classes, lam=lam, mode=mode, projection=projection,
                        pool_kind=pool_kind, pool_k=pool_k)
    t0 = time.time()
    rn.fit(Xtr, ytr, bs=bs)
    preds = rn.predict(Xte, bs=bs)
    elapsed = time.time() - t0
    acc = accuracy_score(yte, preds)
    stats = proto_stats(rn.prototypes)
    return acc, elapsed, stats


def exp1_pooling():
    """Baseline 4 classes + variantes de pooling."""
    print("\n" + "=" * 74)
    print(" EXPERIMENTO 1 — Pooling espacial no crystal_map (4 classes)")
    print("=" * 74)

    Xtr, ytr, Xte, yte, nc = load_20news(CATS_4, 60, 30)
    print(f"Train {len(Xtr)} | Test {len(Xte)} | classes {nc} | FS={psi2.FIELD_SIZE}")

    configs = [
        ('none', None),
        ('avg', 2),
        ('avg', 3),
        ('max', 2),
        ('max', 3),
    ]
    results = []
    for kind, k in configs:
        tag = f"{kind}" if kind == 'none' else f"{kind}_k{k}"
        print(f"\n--- pool={tag} ---")
        acc, t, st = run_one(Xtr, ytr, Xte, yte, nc,
                             pool_kind=kind, pool_k=k or 2)
        print(f"    acc={acc:.3f}  dmin={st['dmin']:.3f}  dmean={st['dmean_off']:.3f}  "
              f"norm={st['norm']:.3f}  t={t:.1f}s")
        results.append((tag, acc, t, st))

    print("\n" + "-" * 74)
    print(f"{'pool':<10} {'acc':>7} {'dmin':>8} {'dmean':>8} {'norm':>8} {'tempo':>8}")
    for tag, acc, t, st in results:
        print(f"{tag:<10} {acc:>7.3f} {st['dmin']:>8.3f} {st['dmean_off']:>8.3f} "
              f"{st['norm']:>8.3f} {t:>7.1f}s")
    return results


def _patched_compute(fs_target):
    """Cria versão de compute_crystal_maps_text com FS dinâmico."""
    def fn(texts, bs=256, lam=None, mode='sobel', projection='v3', verbose=False):
        X = psi2.PROJECTIONS[projection](texts, field_size=fs_target)
        N = len(X)
        out = []
        lam_t = torch.tensor(psi2.CRYSTAL_LAM if lam is None else lam,
                             device=psi2.DEVICE, dtype=torch.float32)
        stats = {'coup_max_global': 0.0, 'coup_max_mean': 0.0, 'n_batches': 0}
        for i in range(0, N, bs):
            pert = X[i:i+bs]
            B = len(pert)
            f, v = pert.clone(), torch.zeros_like(pert)
            mem = psi2.CrystalCompetitivo(B, fs_target)
            src_on = pert * 0.5
            src_off = torch.zeros_like(src_on)
            coup_hist = []
            with torch.no_grad():
                for s in range(psi2.STIM_TOTAL):
                    src = src_on if s < psi2.STIM_ON else src_off
                    f, v, coup_max = psi2.psi_step(f, v, src, mem.crystal_map, lam_t, mode=mode)
                    coup_hist.append(coup_max.item())
                    mem.update_envelope(f)
                    if mem.window_idx > 0:
                        mem.try_crystallize(f)
                    f = mem.remit(f)
            stats['coup_max_global'] = max(stats['coup_max_global'], max(coup_hist))
            stats['coup_max_mean'] += sum(coup_hist) / len(coup_hist)
            stats['n_batches'] += 1
            out.append(mem.crystal_map)
        if verbose and stats['n_batches'] > 0:
            print(f"  [FS={fs_target} proj={projection} mode={mode} lam={lam_t.item():.3f}] "
                  f"coup.max_global={stats['coup_max_global']:.4f}  "
                  f"coup.max_mean={stats['coup_max_mean']/stats['n_batches']:.4f}  "
                  f"src.max={X.max().item():.2f}  src.mean={X.mean().item():.3f}")
        return torch.cat(out, dim=0)
    return fn


def exp2_field_size(best_pool):
    """FS=48 vs FS=64, com a melhor config de pooling."""
    print("\n" + "=" * 74)
    print(f" EXPERIMENTO 2 — Grid size (pool={best_pool[0]}, k={best_pool[1]})")
    print("=" * 74)

    Xtr, ytr, Xte, yte, nc = load_20news(CATS_4, 60, 30)
    orig_compute = psi2.compute_crystal_maps_text
    results = []
    for fs in [48, 64]:
        print(f"\n--- FS={fs} ---")
        psi2.compute_crystal_maps_text = _patched_compute(fs)
        acc, t, st = run_one(Xtr, ytr, Xte, yte, nc,
                             pool_kind=best_pool[0], pool_k=best_pool[1])
        print(f"    acc={acc:.3f}  dmin={st['dmin']:.3f}  norm={st['norm']:.3f}  t={t:.1f}s")
        results.append((fs, acc, t, st))
    psi2.compute_crystal_maps_text = orig_compute

    print("\n" + "-" * 74)
    print(f"{'FS':<5} {'acc':>7} {'dmin':>8} {'norm':>8} {'tempo':>8}")
    for fs, acc, t, st in results:
        print(f"{fs:<5} {acc:>7.3f} {st['dmin']:>8.3f} {st['norm']:>8.3f} {t:>7.1f}s")
    return results


def exp3_classes(best_pool):
    """Escalar classes 4 → 10 → 20 com a melhor config."""
    print("\n" + "=" * 74)
    print(f" EXPERIMENTO 3 — Nº de classes (pool={best_pool[0]}, k={best_pool[1]})")
    print("=" * 74)

    setups = [
        ('4', CATS_4),
        ('10', CATS_10),
        ('20', CATS_20),
    ]
    results = []
    for name, cats in setups:
        print(f"\n--- {name} classes ---")
        Xtr, ytr, Xte, yte, nc = load_20news(cats, 60, 30)
        print(f"    train {len(Xtr)} | test {len(Xte)} | random={1/nc:.1%}")
        acc, t, st = run_one(Xtr, ytr, Xte, yte, nc,
                             pool_kind=best_pool[0], pool_k=best_pool[1])
        uplift = acc - 1/nc
        print(f"    acc={acc:.3f}  uplift vs random={uplift:+.3f}  "
              f"dmin={st['dmin']:.3f}  dmean={st['dmean_off']:.3f}  t={t:.1f}s")
        results.append((name, nc, acc, uplift, t, st))

    print("\n" + "-" * 74)
    print(f"{'nC':<4} {'acc':>7} {'rand':>7} {'uplift':>8} {'dmin':>8} {'dmean':>8} {'tempo':>8}")
    for name, nc, acc, up, t, st in results:
        print(f"{name:<4} {acc:>7.3f} {1/nc:>7.3f} {up:>+8.3f} "
              f"{st['dmin']:>8.3f} {st['dmean_off']:>8.3f} {t:>7.1f}s")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', type=str, default='',
                        help='vírgula-separado: 1,2,3 experimentos a pular')
    args = parser.parse_args()
    skip = set(args.skip.split(',')) if args.skip else set()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    print(f"Device: {psi2.DEVICE}")
    print(f"FS base={psi2.FIELD_SIZE}  STIM_TOTAL={psi2.STIM_TOTAL}")
    print(f"Config: v3 + energy + lam=0.05 | baseline anterior (none): 40.8%")

    if '1' not in skip:
        r1 = exp1_pooling()
        best = max(r1, key=lambda r: r[1])
        tag = best[0]
        if tag == 'none':
            best_pool = ('none', 2)
        else:
            kind, _, k = tag.partition('_k')
            best_pool = (kind, int(k))
        print(f"\n[best pool exp1] {best_pool} — acc={best[1]:.3f}")
    else:
        best_pool = ('none', 2)
        print(f"\n[skipping exp1, usando pool={best_pool}]")

    r2 = exp2_field_size(best_pool) if '2' not in skip else []
    r3 = exp3_classes(best_pool) if '3' not in skip else []

    print("\n" + "=" * 74)
    print(" RESUMO GERAL")
    print("=" * 74)
    if r2:
        best_fs = max(r2, key=lambda r: r[1])
        print(f"  Exp 2 melhor: FS={best_fs[0]} acc={best_fs[1]:.3f}")
    if r3:
        print(f"  Exp 3 (escala): " + " → ".join(
            f"{nc}c:{acc:.3f}" for name, nc, acc, *_ in r3))


if __name__ == '__main__':
    main()
