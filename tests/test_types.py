from __future__ import annotations

import pytest

from flybrain.loop.types import Npc


def _npc(options: tuple[str, ...]) -> Npc:
    return Npc(
        id=1,
        index=0,
        name="Chicken",
        combat_level=1,
        x=0,
        z=0,
        size=1,
        distance=1,
        hp=3,
        max_hp=3,
        in_combat=False,
        target_index=-1,
        reachable=True,
        options=options,
    )


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (("Attack",), True),
        (("attack",), True),
        (("Talk-to", "ATTACK"), True),
        (("Talk-to", "Pickpocket"), False),
        ((), False),
    ],
)
def test_attackable_reads_the_attack_option_case_insensitively(options, expected):
    assert _npc(options).attackable is expected
