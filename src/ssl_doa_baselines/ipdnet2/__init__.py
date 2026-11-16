"""IPDnet2-Small (fixed 3-microphone array, online DP-IPD).

Ported from ``Audio-WestlakeU/FN-SSL@76fcb281`` ``IPDnet2/`` (MIT); see
``THIRD_PARTY.md``.  Importing this package does not build the Mamba backbone;
only selecting ``ipdnet2_small`` does.
"""

from .model import IPDnet2SmallBaseline

__all__ = ["IPDnet2SmallBaseline"]
