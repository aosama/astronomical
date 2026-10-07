# Swift console snapshot

This directory is a physical snapshot of the Observatory console, taken
from apps/supervisor/console at migration time. The Rust supervisor
embedded the same snapshot at build time through include_str!, and the
render stack's source of truth remains Thin Talk's canvas shell at
apps/thin-talk/Sources/ThinTalkCanvas/Resources/web. When that shell
changes, refresh this snapshot so the Swift daemon serves the same
assets the Rust daemon embedded.
