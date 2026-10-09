import Foundation;

import Testing;

import ModelServing;

/// One serialized container for every hermetic MLX journey suite in this
/// target. The container's `.serialized` trait serializes the whole
/// subtree, so GPU-evaluating suites never overlap each other's Metal
/// streams — their bit-exact numeric assertions stay deterministic — while
/// every CPU-only hermetic journey outside this container keeps running
/// fully parallel.
@Suite(.serialized, .tags(.hermeticMlxJourney))
final class HermeticMlxJourneyContainer {}
