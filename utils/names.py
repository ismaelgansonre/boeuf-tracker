"""
utils/names.py
--------------
Name generator for identified cattle.

Stability strategy:
1. A cattle's name is determined by its key in cattle_db.pkl (Boeuf_001, Boeuf_002, ...)
2. The "Boeuf_NNN → Name" mapping is computed by creation order
3. On reload, we re-read the DB, sort by first_seen, and re-map the same names
"""
import json
import os
from datetime import datetime
from typing import Optional


# ─── Name Pool ────────────────────────────────────────────────────────────────
NAME_POOL = [
    # Europeans (FR/IT/EN/DE/ES ~ 50 names)
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
    # Africans (~50 names: Wolof, Bambara, Mandinka, Swahili, etc.)
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
    # Extras for >100 cattle
    "Carbone", "Eclipse", "Foudre", "Givre", "Hurricane",
    "Ivresse", "Jade", "Krystal", "Lune", "Mirage",
    "Neptune", "Onyx", "Perle", "Quartz", "Rubis",
    "Saphir", "Topaze", "Umbra", "Vega", "Whisky",
]


class GlobalCounter:
    """Persistent counter for creating new keys (Boeuf_NNN).

    Survives DB resets: if we purge cattle_db.pkl, the counter keeps its value.
    """

    def __init__(self, path: str = "names_counter.json"):
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

    def next(self) -> str:
        """Increment and return next key."""
        self.value += 1
        self._save()
        return f"Boeuf_{self.value:03d}"

    def reset(self) -> None:
        """Reset counter to 0."""
        self.value = 0
        self._save()

    def __repr__(self) -> str:
        return f"GlobalCounter(value={self.value})"


class NameGenerator:
    """Generates stable names for cattle."""

    def __init__(self, counter: GlobalCounter | None = None):
        self.counter = counter or GlobalCounter()
        self._used_names: set[str] = set()
        self._key_to_name: dict[str, str] = {}

    def get_name(self, key: str) -> str:
        """Get or create name for a key."""
        if key in self._key_to_name:
            return self._key_to_name[key]
        # Find next available name
        for name in NAME_POOL:
            if name not in self._used_names:
                self._used_names.add(name)
                self._key_to_name[key] = name
                return name
        # Fallback: generate a unique name
        idx = len(self._used_names)
        name = f"Neutre_{idx+1}"
        self._used_names.add(name)
        self._key_to_name[key] = name
        return name

    def next_key(self) -> str:
        """Generate next unique key."""
        return self.counter.next()

    def build_from_db(self, animals: dict) -> None:
        """Build name mapping from existing database."""
        sorted_animals = sorted(
            animals.items(),
            key=lambda x: x[1].get("first_seen", ""),
        )
        for key, data in sorted_animals:
            name = data.get("proper_name")
            if name:
                self._used_names.add(name)
                self._key_to_name[key] = name
            else:
                self.get_name(key)  # Assign new name


# ─── Convenience functions ─────────────────────────────────────────────────────
_counter_instance: Optional[GlobalCounter] = None
_generator_instance: Optional[NameGenerator] = None


def get_counter() -> GlobalCounter:
    global _counter_instance
    if _counter_instance is None:
        _counter_instance = GlobalCounter()
    return _counter_instance


def get_name_generator() -> NameGenerator:
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = NameGenerator(get_counter())
    return _generator_instance


def next_bovin_key() -> str:
    return get_counter().next()


def make_name_generator() -> NameGenerator:
    return NameGenerator(get_counter())
