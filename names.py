"""
names.py
---------
Generateur de noms propres pour les bovins identifies.

Stratégie de stabilité cross-session :
  1. Le nom d'un bovin est determiné par sa CLE dans cattle_db.pkl (Boeuf_001,
     Boeuf_002, ...). Cette cle est elle-meme stable grace à l'embedding
     Re-ID.
  2. Le mappage "Boeuf_NNN → Nom" est calcule PAR ORDRE DE CREATION : le 1er
     bovin enregistre = 1er nom du pool, le 2e = 2e nom, etc.
  3. Au prochain chargement, on relit la DB, on ordonne les bovins par
     first_seen, et on re-mappe les memes noms. Le bovin "Boeuf_003"
     d'aujourd'hui aura TOUJOURS le meme nom la prochaine fois.

Pool de noms : melange europeen + africain (~100), equilibre 50/50.
Pas d'emojis (compat terminal/unix).
"""
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


class NameGenerator:
    """Assigne des noms stables aux bovins à partir de la DB existante.

    Usage :
        gen = NameGenerator(db.animals)
        name = gen.get("Boeuf_001")  # → "Marguerite"
    """

    def __init__(self, animals: dict):
        """animals : dict {key: {first_seen, ...}} du EmbeddingDatabase"""
        # Tri par first_seen (les plus anciens d'abord) pour ordre stable.
        sorted_animals = sorted(
            animals.items(),
            key=lambda x: x[1].get("first_seen") or "",
        )
        self._mapping: dict[str, str] = {}
        for idx, (key, _) in enumerate(sorted_animals):
            if idx < len(NAME_POOL):
                self._mapping[key] = NAME_POOL[idx]
            else:
                # Au-delà du pool : combinaison suffixée (rare en pratique)
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
