"""JevBERT: Jev-compatible typed decision API server (P0.5 PoC)."""

__all__ = ["CONTRACT_PROFILE", "CONFIDENCE_DEFINITION", "__version__"]

__version__ = "0.1.0"

#: Contract profile name of this implementation (spec 2.2). Not a TypeSafe version number.
CONTRACT_PROFILE = "jevbert-core-2026-09-21"

#: Confidence definition ID (spec 8.2).
CONFIDENCE_DEFINITION = "normalized-entropy-v1"
