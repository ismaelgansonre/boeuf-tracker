"""
# Backward compatibility - imports from new structure
from utils.names import NameGenerator, GlobalCounter, get_counter, get_name_generator, next_bovin_key, make_name_generator

__all__ = ["NameGenerator", "GlobalCounter", "get_counter", "get_name_generator", "next_bovin_key", "make_name_generator", "NAME_POOL"]

Stratégie de stabilité cross-session :
  1. Le nom d'un bovin est determiné par sa CLE dans cattle_db.pkl (Boeuf_001,
     Boeuf_002, ...). Cette cle est elle-meme stable grace à l'embedding
     Re-ID.
  2. Le mappage "Boeuf_NNN → Nom" est calcule PAR ORDRE DE CREATION : le 1er
     bovin enregistre recoit le 1er nom du pool, le 2e = 2e nom, etc.
  3. Au prochain chargement, on relit la DB, on ordonne les bovins par
     first_seen, et on re-mappe les memes noms. Le bovin "Boeuf_003"
     d'aujourd'hui aura TOUJOURS le meme nom la prochaine fois.

NOMS JAMAIS REUTILISES :
  On maintient un COMPTEUR GLOBAL persistant (names_counter.json) qui ne
  fait qu'augmenter, meme si la DB est reset. Ainsi, un bovin vu apres un
  reset de DB ne recevra JAMAIS un nom deja utilise. Le 1er bovin de la
  1ere video = Marguerite ; le 1er bovin de la 2e video (apres reset) =
  le nom SUIVANT dans le pool, jamais Marguerite.

Pool de noms : melange europeen + africain (~100), equilibre 50/50.
Pas d'emojis (compat terminal/unix).
"""
import json
import os
from datetime import datetime
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────
# Pool de noms — melange europeen (FR/EN/IT/DE) + africain (WF/WA/SN/ML).
# L'ordre est important : le premier bovin enregistre recoit le premier nom.
# ──────────────────────────────────────────────────────────────────────────
NAME_POOL = [
    # Europeens (FR/IT/EN/DE/ES ~ 50 noms)
    "Marguerite", "Ulysse", "Belline", "Hercules", "Colette",
    "Asterix", "Camille", "Aurelius", "Sybille", "Beethoven",
    "Charline", "Bastille", "Daphne", "Romarin", "Eugenie",
    "Hyppolyte", "Cendrillon", "Vasco", "Mathilde", "Achille",
    "Clemence", "Silvain", "Anemone", "Hortense", "Gustave",
    "Azalee", "Beranger", "Cerise", "Eleonore", "Fernand",
    "Gregoire", "Helene", "Irene", "Josephine", "Killian",
    "Leopold", "Marceline", "Norbert", "Olympe", "Pascaline",
    "Quentin", "Rosemonde", "Sebastien", "Therese", "Ulrich",
    "Valentine", "Wenceslas", "Xavier", "Yseult", "Zephir",
    # Africains (~50 noms, cultures variees : Wolof, Bambara, Mandinka, Swahili, etc.)
    "N'Dalla", "Sundiata", "Amina", "Kimpa", "Samba",
    "Ayo", "Chiamaka", "Folami", "Kelechi", "Nuru",
    "Sade", "Tunde", "Adaeze", "Bayo", "Damilola",
    "Enitan", "Fisayo", "Ifeoma", "Joke", "Kosi",
    "Lumumba", "Mandela", "Nzinga", "Onyeka", "Panya",
    "Sankara", "Togo", "Usman", "Yaw", "Zola",
    "Aissatou", "Boubacar", "Cheick", "Djeneba", "Fatoumata",
    "Goundo", "Hawa", "Issa", "Kadiatou", "Lalla",
    "Mariama", "Nafissatou", "Ousmane", "Rokia", "Salimata",
    "Tiemoko", "Wassa", "Yacouba", "Zenab", "Bintou",
    # Quelques noms supplementaires pour eviter la repetition si >100 bovins
    "Carbone", "Eclipse", "Foudre", "Givre", "Hurricane",
    "Ivresse", "Jade", "Krystal", "Lune", "Mirage",
    "Neptune", "Onyx", "Perle", "Quartz", "Rubis",
    "Saphir", "Topaze", "Umbra", "Vega", "Whisky",
]


# ──────────────────────────────────────────────────────────────────────────
# Compteur global persistant. Garantit que chaque bovin JAMAIS vu recoit un
# nom unique, meme apres un reset de la DB. Le compteur ne fait qu'augmenter.
# ──────────────────────────────────────────────────────────────────────────
COUNTER_PATH = "names_counter.json"


class GlobalCounter:
    """Compteur persistant pour la creation de nouvelles cles (Boeuf_NNN).

    Survit aux resets de DB : si on purge cattle_db.pkl, le compteur garde
    sa valeur. Le prochain bovin sera Boeuf_042 (par ex), jamais Boeuf_001.
    """

    def __init__(self, path: str = COUNTER_PATH):
        self.path = path
        self.value = self._load()

    def _load(self) -> int:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    return int(json.load(f).get("counter", 0))
            except Exception:
                return 0
        return 0

    def _save(self) -> None:
        try:
            with open(self.path, "w") as f:
                json.dump({"counter": self.value}, f)
        except Exception:
            pass

    def next(self) -> int:
        """Incremente et retourne le prochain numero de bovin."""
        self.value += 1
        self._save()
        return self.value

    def peek(self) -> int:
        """Retourne le numero du dernier bovin cree (sans incrementer)."""
        return self.value

    def reset(self) -> None:
        """Remet le compteur a zero (reset explicite uniquement)."""
        self.value = 0
        self._save()


# Singleton global (charge une fois au demarrage)
_global_counter: Optional[GlobalCounter] = None


def get_counter() -> GlobalCounter:
    """Retourne le compteur global (singleton)."""
    global _global_counter
    if _global_counter is None:
        _global_counter = GlobalCounter()
    return _global_counter


def next_bovin_key() -> str:
    """Genere la prochaine cle unique pour un nouveau bovin.

    Garantit l'unicite meme apres reset de DB :
      Boeuf_001, Boeuf_002, ... Boeuf_042, [reset DB], Boeuf_043, ...
    """
    n = get_counter().next()
    return f"Boeuf_{n:03d}"


class NameGenerator:
    """Assigne des noms stables aux bovins à partir de la DB existante.

    Usage :
        gen = NameGenerator(db.animals)
        name = gen.get("Boeuf_001")  # → "Marguerite"
    """

    def __init__(self, animals: dict):
        """animals : dict {key: {first_seen, ...}} du EmbeddingDatabase

        Le mapping se fait par le NUMERO de la cle (Boeuf_004 -> NAME_POOL[3]),
        PAS par l'index de tri. Ainsi, apres un switch de video qui vide la DB,
        les nouveaux Boeuf_004+ recoivent des noms DIFFERENTS de Boeuf_001-003
        qui etaient dans la video precedente.
        """
        self._mapping: dict[str, str] = {}
        for key in animals:
            # Extrait le numero: Boeuf_004 -> 4 -> index 3 (0-based)
            idx = -1
            if key.startswith("Boeuf_"):
                try:
                    idx = int(key.split("_")[1]) - 1
                except (ValueError, IndexError):
                    idx = -1
            if idx < 0:
                # Cle non standard: fallback hash stable
                idx = abs(hash(key)) % len(NAME_POOL)
            if idx < len(NAME_POOL):
                self._mapping[key] = NAME_POOL[idx]
            else:
                # Au-dela du pool : combinaison suffixee (rare en pratique)
                base = NAME_POOL[idx % len(NAME_POOL)]
                cycle = idx // len(NAME_POOL)
                self._mapping[key] = f"{base} {cycle + 2}"

    def get(self, key: str) -> str:
        """Retourne le nom propre pour une clé (Boeuf_001 → 'Marguerite')."""
        return self._mapping.get(key, key)

    def all(self) -> dict[str, str]:
        """Retourne tout le mapping."""
        return dict(self._mapping)


def make_name_generator(db) -> NameGenerator:
    """Helper : instancie un NameGenerator à partir d'un EmbeddingDatabase."""
    return NameGenerator(db.animals)
