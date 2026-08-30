"""Hyperparameters for the pick CLI."""

# Pickrate z-score thresholds (relative to per-event mean)
PICKRATE_HIGH_Z: float = 1.0   # "often picked" — top ~10% of pickrates
PICKRATE_LOW_Z: float = -0.5   # "rarely picked" — below-average pickrate

# Advantage threshold: after z-scoring over available brawlers, z > 0 means
# above-average pick for the current draft state; z < 0 means below-average.
Q_THRESHOLD: float = 0.0

# Annotation strings (appended after the score cell; 3 terminal columns wide)
ANN_BRAIN: str = " 🧠"  # Q > 0 and low pickrate (hidden gem)
ANN_SECRET: str = " 🤫"  # overview: good map score, low pickrate (secret pick)
ANN_X: str = " ✗ "  # Q ≤ 0 or bad map score, high pickrate (overrated)
ANNOTATION_SLOT: int = 3  # terminal columns reserved for annotation suffix
