"""Exceptions. Every subclass of Inconclusive maps to verdict INCONCLUSIVE / exit code 2."""


class Inconclusive(Exception):
    """The experiment could not be run soundly. Fail closed: never guess."""

    reason = "INCONCLUSIVE"

    def __init__(self, message: str, *, reason: str | None = None):
        super().__init__(message)
        if reason:
            self.reason = reason


class ServerUnreachable(Inconclusive):
    reason = "SERVER_UNREACHABLE"


class TokenizationError(Inconclusive):
    reason = "TOKENIZATION_INVALID"


class MissingProbabilities(Inconclusive):
    reason = "PROBABILITY_FIELDS_MISSING"


class ProbabilityCollapse(Inconclusive):
    """Distribution looks post-sampling / greedy-collapsed (e.g. {A: 1.0, B: 0.0})."""

    reason = "PROBABILITY_COLLAPSE"


class FixtureError(Inconclusive):
    reason = "FIXTURES_INVALID"


class BaselineError(Inconclusive):
    reason = "BASELINE_INVALID"


class TraceError(Inconclusive):
    reason = "TRACE_INVALID"
