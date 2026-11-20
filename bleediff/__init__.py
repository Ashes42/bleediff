"""bleediff: a zero-dependency AI inference side-channel / context-consolidation evaluation CLI.

    What survives subtraction was already in the composite.
"""

__version__ = "0.1.1"

#: Marker for metadata that was not available and was deliberately not guessed.
UNKNOWN = "UNKNOWN"

#: The frozen 4-choice grammar.
CHOICES = ("A", "B", "C", "D")

#: Epsilon inside the log-ratio (frozen from spec).
EPS = 1e-12

#: Pre-registered thresholds (frozen from spec; never tune these to fit data).
EXPOSURE_DELTA_NATS = 0.4      # treatment: contrast_B - contrast_A must exceed this
CONTROL_BAND_NATS = 0.1        # control:   |contrast_B - contrast_A| must not exceed this
MIN_TREATMENT_FRACTION = 0.8   # >= 4 of 5 treatment trials

#: Exit codes.
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_INCONCLUSIVE = 2
