"""Gesture readings and the tap-to-change-effect state machine.

Measurements are divided by ``Hand.scale`` so they mean the same thing whether
the hand is near the camera or far from it.
"""

from __future__ import annotations

import math

import numpy as np

from .config import GestureConfig
from .tracking import (
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
    Hand,
)

# (tip, pip) pairs for the four fingers. The thumb bends sideways rather than
# curling, so it gets its own test.
_FINGERS = (
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
)


def pinch(hand: Hand) -> float:
    """Thumb-tip to index-tip distance, normalised by hand size."""
    gap = np.linalg.norm(hand.points[THUMB_TIP] - hand.points[INDEX_TIP])
    return float(gap) / hand.scale


def axis_angle(hand: Hand) -> float:
    """Angle of the thumb-to-index axis - which way this hand is turned."""
    delta = hand.points[INDEX_TIP] - hand.points[THUMB_TIP]
    return math.atan2(float(delta[1]), float(delta[0]))


def finger_states(hand: Hand) -> tuple[bool, bool, bool, bool, bool]:
    """Which fingers are straight, as (thumb, index, middle, ring, pinky)."""
    wrist = hand.points[WRIST]
    states = []
    for tip, pip in _FINGERS:
        # A straight finger puts its tip further from the wrist than its middle
        # joint; a curled one tucks the tip back in.
        states.append(
            np.linalg.norm(hand.points[tip] - wrist)
            > np.linalg.norm(hand.points[pip] - wrist)
        )

    # The thumb is judged on how far it sits from the palm's edge instead.
    palm_width = float(np.linalg.norm(hand.points[INDEX_MCP] - hand.points[PINKY_MCP]))
    thumb_gap = float(np.linalg.norm(hand.points[THUMB_TIP] - hand.points[PINKY_MCP]))
    thumb = palm_width > 1e-6 and thumb_gap / palm_width > 1.15
    return (thumb, *states)


def extended_fingers(hand: Hand) -> int:
    """Count roughly-straight fingers, thumb included."""
    return sum(finger_states(hand))


def is_two_fingers(hand: Hand) -> bool:
    """Index and middle out, ring and pinky curled in - the orb gesture.

    The thumb is deliberately ignored: people hold this sign with it tucked in
    or cocked out and mean the same thing by it.
    """
    _, index, middle, ring, pinky = finger_states(hand)
    return index and middle and not (ring or pinky)


def orb_hand(hands: list[Hand]) -> Hand | None:
    for hand in hands:
        if is_two_fingers(hand):
            return hand
    return None


def is_open_palm(hand: Hand) -> bool:
    """An open hand with the fingers spread, as opposed to a fist or a point."""
    return extended_fingers(hand) >= 4


def palms_open(hands: list[Hand]) -> bool:
    """Both hands held open - the gesture that grows the rose in the box."""
    return len(hands) >= 2 and all(is_open_palm(hand) for hand in hands[:2])


def solo_palm(hands: list[Hand]) -> bool:
    """A single open palm, with no second hand up.

    Held on its own this shows the rose resting on your hand with no box around
    it - the box needs two hands to have anything to hang between.
    """
    return len(hands) == 1 and is_open_palm(hands[0])


def wants_rose(hands: list[Hand]) -> bool:
    return palms_open(hands) or solo_palm(hands)


def relative_twist(hands: list[Hand]) -> float:
    """How far the two hands are rotated relative to each other, in radians.

    Read straight off the two thumb-to-index axes rather than tracked over time.
    The shape twists because the fingertips moved, so there is no twist state to
    keep - this is only for the readout.
    """
    if len(hands) < 2:
        return 0.0
    delta = axis_angle(hands[1]) - axis_angle(hands[0])
    return math.atan2(math.sin(delta), math.cos(delta))


def is_tapping(hand: Hand, cfg: GestureConfig) -> bool:
    """True while this hand holds its thumb and index finger together."""
    return pinch(hand) < cfg.tap_threshold


class EffectSwitcher:
    """Tapping thumb to index on both hands at once steps to the next effect.

    The tap has to be released before another one counts, so holding both
    pinches shut changes the effect once rather than racing through the list.
    The cooldown then covers the moment right after a tap, where fingers are
    still close enough to read as a second one.
    """

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self._armed = True
        self._last_switch = -1e9

    def update(self, hands: list[Hand], now: float) -> int:
        """Return 1 when the effect should advance, otherwise 0."""
        if not self.cfg.enabled:
            return 0

        tapping = sum(1 for hand in hands if is_tapping(hand, self.cfg))
        both = len(hands) >= 2 and tapping >= 2

        if not both:
            self._armed = True
            return 0

        if not self._armed or now - self._last_switch < self.cfg.cooldown:
            return 0

        self._armed = False
        self._last_switch = now
        return 1

    def reset(self) -> None:
        self._armed = True
        self._last_switch = -1e9
