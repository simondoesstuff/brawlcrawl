"""Hyperparameters for the pick CLI."""

# Pickrate z-score thresholds (relative to per-event mean)
PICKRATE_HIGH_Z: float = 1.0   # "often picked" — top ~10% of pickrates
PICKRATE_LOW_Z: float = 0.0    # "rarely picked" — below-average pickrate

# 6th-pick display
WIN_PROB_THRESHOLD: float = 0.50  # divider between favoured / unfavoured picks

# Annotation strings (appended after the score cell; 3 terminal columns wide)
ANN_BRAIN: str = " 🧠"  # model ≥50% win, low pickrate (hidden gem)
ANN_X: str = " ✗ "      # model <50% win, high pickrate (overrated)
ANNOTATION_SLOT: int = 3  # terminal columns reserved for annotation suffix
