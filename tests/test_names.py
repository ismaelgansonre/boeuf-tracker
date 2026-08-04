"""Tests names.py — le pool de noms et sa stabilité dans le temps.

Le nom affiché d'un bovin est `NAME_POOL[numero_de_cle - 1]`. Le pool fait
donc partie du format de données : toucher à son ORDRE renomme rétroactivement
des animaux déjà enregistrés en base. Ces tests verrouillent ce contrat.
"""
from collections import Counter

from names import NAME_POOL, NameGenerator


# Préfixe historique : ces positions sont gravées dans les bases existantes.
_FROZEN_PREFIX = {
    0: "Marguerite",
    1: "Ulysse",
    49: "Zephir",
    50: "N'Dalla",
    99: "Bintou",
    119: "Whisky",
}


def test_pool_prefix_is_frozen():
    """Régression : seuls des AJOUTS en fin de liste sont autorisés."""
    for index, expected in _FROZEN_PREFIX.items():
        assert NAME_POOL[index] == expected, (
            f"NAME_POOL[{index}] a change ({NAME_POOL[index]!r} au lieu de "
            f"{expected!r}) — les bovins deja en base seraient renommes."
        )


def test_pool_has_no_duplicates():
    """Deux bovins distincts ne doivent jamais porter le même nom."""
    doublons = [n for n, c in Counter(NAME_POOL).items() if c > 1]
    assert doublons == []


def test_pool_is_ascii_only():
    """Compat terminal/logs : pas d'accent ni d'emoji dans les étiquettes."""
    assert [n for n in NAME_POOL if not n.isascii()] == []


def test_pool_is_large_enough_for_a_full_herd():
    """Au-delà du pool, les noms sont suffixés ("Daphne 2") et deviennent
    ambigus à l'écran. Un troupeau de stabulation dépasse 120 têtes."""
    assert len(NAME_POOL) >= 300


def test_generator_maps_key_number_to_pool_index():
    gen = NameGenerator({"Boeuf_001": {}, "Boeuf_002": {}})
    assert gen.get("Boeuf_001") == NAME_POOL[0]
    assert gen.get("Boeuf_002") == NAME_POOL[1]


def test_generator_is_stable_across_instances():
    """Deux sessions successives doivent produire le même mapping."""
    animals = {"Boeuf_007": {}, "Boeuf_042": {}}
    assert NameGenerator(animals).all() == NameGenerator(animals).all()


def test_generator_suffixes_cycle_beyond_pool():
    idx = len(NAME_POOL) + 1        # 2e cycle
    gen = NameGenerator({f"Boeuf_{idx:03d}": {}})
    assert gen.get(f"Boeuf_{idx:03d}") == f"{NAME_POOL[idx - 1 - len(NAME_POOL)]} 2"


def test_generator_falls_back_on_unknown_key():
    gen = NameGenerator({})
    assert gen.get("inconnu") == "inconnu"
