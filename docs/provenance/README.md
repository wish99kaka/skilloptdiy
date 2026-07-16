# Provenance Evidence

This directory contains byte-for-byte mirrors of immutable evidence referenced
by packaged registries. Runtime output under `runs/` remains ignored; registry
checks use these tracked mirrors so a clean clone can verify every commitment.

Never edit a mirrored artifact in place. A replacement requires a new registry
entry and hash rather than rewriting historical evidence.
