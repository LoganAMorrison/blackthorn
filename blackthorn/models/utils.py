from typing import List, Tuple

from ..constants import Gen


LEPTON_STR_GEN: List[Tuple[str, Gen]] = [
    ("e", Gen.Fst),
    ("mu", Gen.Snd),
    ("tau", Gen.Trd),
]
UP_QUARK_STR_GEN: List[Tuple[str, Gen]] = [
    ("u", Gen.Fst),
    ("c", Gen.Snd),
    ("t", Gen.Trd),
]
DOWN_QUARK_STR_GEN: List[Tuple[str, Gen]] = [
    ("d", Gen.Fst),
    ("s", Gen.Snd),
    ("b", Gen.Trd),
]


def final_state_generations_n_to_three_leptons(gen_n: Gen, unique: bool = False):
    gen1 = gen_n
    gen2, gen3 = {Gen.Fst, Gen.Snd, Gen.Trd}.difference({gen_n})
    gens = [
        (gen1, gen1, gen1),
        (gen1, gen2, gen2),
        (gen1, gen3, gen3),
    ]

    if not unique:
        gens.append((gen2, gen1, gen2))
        gens.append((gen2, gen2, gen1))
        gens.append((gen3, gen1, gen3))
        gens.append((gen3, gen3, gen1))

    return gens


def final_state_strings_n_to_three_leptons(gen_n: Gen, unique: bool = False):
    strs = ["e", "mu", "tau"]

    def gen_tup_to_str_tup(tup):
        return tuple(map(lambda gen: strs[gen], tup))

    gen_tups = final_state_generations_n_to_three_leptons(gen_n, unique)
    return list(map(gen_tup_to_str_tup, gen_tups))


def energies_two_body_final_state(cme, m1, m2) -> Tuple[float, float]:
    e1 = (cme**2 + m1**2 - m2**2) / (2 * cme)
    e2 = (cme**2 - m1**2 + m2**2) / (2 * cme)
    return e1, e2
