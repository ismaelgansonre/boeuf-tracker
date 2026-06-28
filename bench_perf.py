"""
bench_perf.py
-------------
Benchmark des optimisations du pipeline cattle-tracker.

Mesure:
1. database.match() loop vs vectorisé (vary N animaux)
2. database.match() avec reid_engine (compare pondéré DINO/HSV/LBP)
3. analyze_behavior() avant/après Numba JIT

Le benchmark DINOv2 batch vs single est optionnel (--with-dino) car il
charge le modèle (~3s + 500MB RAM).
"""
import argparse
import time
import sys
import numpy as np


def bench_match(n_animals: int, n_iters: int = 2000):
    """Compare l'ancien match (boucle Python + recompute normes) vs le nouveau."""
    from database import EmbeddingDatabase

    rng = np.random.default_rng(42)
    db = EmbeddingDatabase(path=":memory:")
    db.animals = {
        f"boeuf_{i:03d}": {"embedding": rng.standard_normal(464).astype(np.float32),
                           "count": 1, "first_seen": ""}
        for i in range(n_animals)
    }
    db._dirty = True
    q = rng.standard_normal(464).astype(np.float32)
    db._ensure_cache()

    # Vectorisé (nouveau code, fallback auto boucle si N≤10)
    t = time.perf_counter()
    for _ in range(n_iters):
        name, sim = db.match(q, threshold=0.55)
    t_vec = time.perf_counter() - t

    # Boucle Python STRICTE = ancien code d'origine
    # (recompute np.linalg.norm(data["embedding"]) à chaque itération)
    t = time.perf_counter()
    for _ in range(n_iters):
        best_name, best_sim = None, -1.0
        for name, data in db.animals.items():
            emb = data["embedding"]
            sim = float(np.dot(emb, q) / (np.linalg.norm(emb) * np.linalg.norm(q) + 1e-8))
            if sim > best_sim:
                best_sim = sim
                best_name = name
    t_loop = time.perf_counter() - t

    return t_loop, t_vec


def bench_match_reid(n_animals: int, n_iters: int = 2000):
    """Compare match avec reid_engine (compare pondéré par composante)."""
    from database import EmbeddingDatabase

    class FakeReID:
        DINO_DIM = 384
        HSV_DIM = 48
        LBP_DIM = 32
        w_dino = 0.5
        w_hsv = 0.3
        w_lbp = 0.2
        @staticmethod
        def compare(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    rng = np.random.default_rng(42)
    db = EmbeddingDatabase(path=":memory:", reid_engine=FakeReID())
    db.animals = {
        f"boeuf_{i:03d}": {"embedding": rng.standard_normal(464).astype(np.float32),
                           "count": 1, "first_seen": ""}
        for i in range(n_animals)
    }
    db._dirty = True
    q = rng.standard_normal(464).astype(np.float32)
    db._ensure_cache()

    # Vectorisé
    t = time.perf_counter()
    for _ in range(n_iters):
        name, sim = db.match(q, threshold=0.55)
    t_vec = time.perf_counter() - t

    # Boucle Python avec compare() (vrai code original, recompute normes)
    reid = FakeReID()
    t = time.perf_counter()
    for _ in range(n_iters):
        best_name, best_sim = None, -1.0
        for name, data in db.animals.items():
            sim = reid.compare(q, data["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_name = name
    t_loop = time.perf_counter() - t

    return t_loop, t_vec


def bench_analyze_behavior(n_tracks: int = 5, n_iters: int = 500):
    """Mesure analyze_behavior (Numba JIT warm-up inclus)."""
    from state import STATE
    from processor import analyze_behavior

    rng = np.random.default_rng(7)
    boxes = rng.integers(100, 800, size=(n_tracks, 4)).astype(float)
    boxes[:, 2] += 200
    boxes[:, 3] += 200
    track_ids = np.arange(1, n_tracks + 1)

    # Pré-remplir STATE["track_history"] avec ~10 points par track
    STATE["track_history"].clear()
    t_base = time.time()
    for tid in track_ids:
        STATE["track_history"][int(tid)] = [
            (float(rng.uniform(200, 600)), float(rng.uniform(200, 600)),
             t_base - i * 0.5)
            for i in range(10)
        ]

    # Warmup JIT (1er appel compile)
    analyze_behavior(boxes, track_ids, time.time(), (1080, 1920))

    # Mesure
    t = time.perf_counter()
    for _ in range(n_iters):
        analyze_behavior(boxes, track_ids, time.time(), (1080, 1920))
    return time.perf_counter() - t


def bench_dino_batch(device: str = "cpu", n_crops: int = 4, n_iters: int = 20):
    """Compare N forwards DINOv2 individuels vs 1 forward batché.
    Optionnel, nécessite transformers + torch (~3s load)."""
    try:
        import torch
        from reid import CattleReID
    except Exception as e:
        return None, None, str(e)

    reid = CattleReID(model_name="facebook/dinov2-small",
                      device=device, use_compile=(device.startswith("cuda")))
    rng = np.random.default_rng(0)
    crops = [
        rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
        for _ in range(n_crops)
    ]
    # warmup (compile JIT si activé)
    reid.get_embedding_batch(crops)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # Single (boucle)
    t = time.perf_counter()
    for _ in range(n_iters):
        for c in crops:
            reid.get_embedding(c)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_single = time.perf_counter() - t

    # Batch
    t = time.perf_counter()
    for _ in range(n_iters):
        reid.get_embedding_batch(crops)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_batch = time.perf_counter() - t

    return t_single, t_batch, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-dino", action="store_true",
                    help="Inclut le bench DINOv2 (lent à charger)")
    ap.add_argument("--n-animals", type=int, nargs="+", default=[5, 20, 50, 200])
    args = ap.parse_args()

    print("=" * 70)
    print("BENCHMARK cattle-tracker — optimisations perf")
    print("=" * 70)

    # 1) database.match() cosine simple
    print("\n[1] database.match() — cosine simple (vectorisé vs boucle)")
    print(f"{'N':>6} | {'boucle ms/call':>14} | {'vecto ms/call':>14} | {'gain':>6}")
    print("-" * 60)
    for n in args.n_animals:
        t_loop, t_vec = bench_match(n)
        gain = t_loop / t_vec if t_vec > 0 else 0
        print(f"{n:>6} | {t_loop*1000/2000:>14.4f} | {t_vec*1000/2000:>14.4f} | "
              f"{gain:>5.1f}x")

    # 2) database.match() avec reid_engine
    print("\n[2] database.match() — reid_engine (compare pondéré)")
    print(f"{'N':>6} | {'boucle ms/call':>14} | {'vecto ms/call':>14} | {'gain':>6}")
    print("-" * 60)
    for n in args.n_animals:
        t_loop, t_vec = bench_match_reid(n)
        gain = t_loop / t_vec if t_vec > 0 else 0
        print(f"{n:>6} | {t_loop*1000/2000:>14.4f} | {t_vec*1000/2000:>14.4f} | "
              f"{gain:>5.1f}x")

    # 3) analyze_behavior (Numba JIT)
    print("\n[3] analyze_behavior (Numba JIT)")
    t = bench_analyze_behavior(n_tracks=5, n_iters=500)
    print(f"5 tracks × 500 iters: {t*1000:.1f} ms total = {t*1000/500:.3f} ms/call")

    # 4) DINOv2 batch vs single (optionnel) — CPU et GPU si dispo
    if args.with_dino:
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except Exception:
            has_cuda = False

        for dev in (["cpu", "cuda"] if has_cuda else ["cpu"]):
            print(f"\n[4-{dev}] DINOv2 — batch vs single forward "
                  f"(chargement modèle ~3s)")
            t_s, t_b, err = bench_dino_batch(device=dev, n_crops=4, n_iters=10)
            if err:
                print(f"  SKIP: {err}")
                continue
            gain = t_s / t_b if t_b > 0 else 0
            per_crop_s = t_s * 1000 / (10 * 4)
            per_crop_b = t_b * 1000 / (10 * 4)
            print(f"  single : {t_s*1000/10:6.1f} ms / 4 crops = "
                  f"{per_crop_s:5.1f} ms/crop")
            print(f"  batch  : {t_b*1000/10:6.1f} ms / 4 crops = "
                  f"{per_crop_b:5.1f} ms/crop")
            print(f"  gain   : {gain:.2f}x")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
