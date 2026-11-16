"""IPDnet (fixed 3-microphone array, online DP-IPD).

Ported from ``Audio-WestlakeU/FN-SSL@76fcb281`` ``IPDnet/FixedAarryIPDnet.py``
(MIT); see ``THIRD_PARTY.md``.
"""

from .model import IPDNetBaseline

__all__ = ["IPDNetBaseline"]
