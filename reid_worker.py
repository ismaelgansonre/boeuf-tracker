"""
reid_worker.py
--------------
Thread worker qui découple le calcul d'embeddings DINOv2 de la boucle vidéo.

PROBLÈME
--------
Dans processor.py, `reid.get_embedding_batch()` est appelé de façon synchrone
dans la boucle principale. Toutes les `embed_every` frames (~10), DINOv2 bloque
la boucle pendant 15-40 ms → chute de FPS perceptible.

SOLUTION
--------
Un thread consommateur calcule les embeddings en arrière-plan. La boucle vidéo :
  1. soumet les crops (submit_batch) — non-bloquant
  2. récupère les embeddings prêts (collect_ready) — non-bloquant

Les crops non encore traités sont simplement ignorés pour cette frame : ils
seront re-soumis à la frame suivante (le bovin n'a pas disparu). Le re-embedding
EMA n'a pas besoin d'être instantané — il stabilise l'embedding sur le temps.

Puisque DINOv2 tourne sur MPS et YOLO sur MLX/Metal, les deux backends GPU
travaillent en parallèle sans se marcher sur les pieds.

FALLBACK
--------
Si le worker lève une exception, `collect_ready` dégrade gracieusement : la
boucle retombe sur un batch synchrone via `get_embedding_batch()`. Le pipeline
ne casse jamais.
"""
import queue
import threading
import numpy as np

from console import warn


class ReIDWorker:
    """Thread worker asynchrone pour le calcul d'embeddings Re-ID.

    Usage côté boucle vidéo :
        worker = ReIDWorker(reid_engine)
        worker.submit_batch({det_idx: crop, ...})   # non-bloquant
        emb_by_idx = worker.collect_ready()          # non-bloquant, peut être vide
    """

    def __init__(self, reid_engine, max_queue: int = 2, max_batch: int = 12):
        self.reid = reid_engine
        self.max_batch = max_batch
        # File COURTE et volontairement remplacable (cf. submit_batch).
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._lock = threading.Lock()
        # {det_idx: embedding} — résultats disponibles, accès protégé
        self._results: dict[int, np.ndarray] = {}
        self._stop = threading.Event()
        self._failed = False  # si True, on bascule en mode synchrone
        self._thread = threading.Thread(target=self._loop, daemon=True, name="reid-worker")
        self._thread.start()

    def _loop(self):
        """Consommateur : prend les batches de la queue, calcule, stocke."""
        try:
            while not self._stop.is_set():
                try:
                    token, batch = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if batch is None:
                    break  # signal d'arrêt
                try:
                    # batch: list[(det_idx, crop)]
                    crops = [c for _, c in batch]
                    embeddings = self.reid.get_embedding_batch(crops)
                    with self._lock:
                        for (det_idx, _), emb in zip(batch, embeddings):
                            if emb is not None and emb.shape[0] == self.reid.TOTAL_DIM:
                                self._results[det_idx] = emb
                except Exception as e:
                    warn(f"[ReIDWorker] erreur batch: {e}")
        except Exception as e:
            warn(f"[ReIDWorker] thread crashé, fallback synchrone: {e}")
            self._failed = True

    def submit_batch(
        self,
        batch: dict[int, np.ndarray],
        priority_ids: set[int] | None = None,
    ) -> bool:
        """Soumet un batch de crops. Non-bloquant.

        batch: {track_id: crop_bgr}
        priority_ids: pistes sans nom, à traiter en premier.

        LE DERNIER SOUMIS GAGNE
        -----------------------
        Avec une file longue, un troupeau nombreux la remplissait de batches
        successifs : le worker traitait des imagettes vieilles de plusieurs
        secondes pendant que les nouvelles étaient rejetées. Résultat visible à
        l'écran — des bovins qui restaient étiquetés "?" indéfiniment alors
        qu'ils étaient bien détectés.

        La file est donc courte, et un batch en attente est REMPLACÉ par le
        plus récent : un crop périmé n'a aucune valeur, le bovin est toujours
        là à la frame suivante.

        BORNE SUR LA TAILLE DU BATCH
        ----------------------------
        Le forward est découpé à `max_batch` imagettes, en servant d'abord les
        pistes anonymes. Un batch de 25 bloque le worker ~0.5 s ; deux batches
        de 12 rendent le premier résultat deux fois plus vite, et ce sont les
        nouveaux arrivants qui en profitent.

        Retourne True si accepté, False si le worker est hors service.
        """
        if self._failed:
            return False
        if not batch:
            return True

        items = list(batch.items())
        if priority_ids:
            items.sort(key=lambda kv: kv[0] not in priority_ids)
        items = items[:self.max_batch]

        while True:
            try:
                self._queue.put_nowait((None, items))
                return True
            except queue.Full:
                try:
                    # Évince le batch le plus ancien au profit de celui-ci.
                    self._queue.get_nowait()
                except queue.Empty:
                    pass  # vidée entre-temps par le worker : on retente

    def collect_ready(self) -> dict[int, np.ndarray]:
        """Récupère et vide les résultats disponibles. Non-bloquant.
        Retourne {det_idx: embedding} — peut être vide si rien n'est prêt.
        """
        with self._lock:
            ready = self._results
            self._results = {}
        return ready

    def has_failed(self) -> bool:
        """True si le worker a crashé → l'appelant doit faire du synchrone."""
        return self._failed

    def stop(self):
        """Arrêt propre du thread (à l'exit)."""
        self._stop.set()
        try:
            self._queue.put_nowait((None, None))
        except queue.Full:
            pass
