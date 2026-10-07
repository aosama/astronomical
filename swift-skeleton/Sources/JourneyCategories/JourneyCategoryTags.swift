import Testing;

/**
 * The three journey categories every Swift test suite declares explicitly.
 *
 * The category is the suite's contract with the runner, replacing the
 * Rust tree's ignored-test + wrapper-script split with SwiftPM-native
 * machinery (tags, enablement conditions, SwiftPM's default serial run):
 * no shell script owns test selection.
 *
 * - `hermeticJourney`: synthesized fixtures, CPU only, no MLX evaluation.
 *   Parallel-safe; always runs under plain `swift test`.
 * - `hermeticMlxJourney`: synthesized tiny fixtures that evaluate MLX
 *   operations (construction, weight binding, forward passes on
 *   kilobyte-scale models). Suite-serialized; always runs under plain
 *   `swift test`. These never load installed model weights.
 * - `realModelJourney`: loads an installed artifact's real weights through
 *   `RealModelJourneyGate`'s environment resolution, and also covers the
 *   opt-in heavy MLX journeys that need no installed weights but write
 *   and read roughly 160 MiB of synthesized tensors on this machine
 *   (gated by their own environment variables for the same reason).
 *   Disabled unless the gate resolves, so plain `swift test` can never
 *   pick one up; suites
 *   carrying this tag must also be `.serialized` and must run one suite
 *   at a time (SwiftPM's default `--no-parallel` keeps this structural —
 *   never pass `--parallel` to a real-model run).
 */
public extension Tag {

    @Tag static var hermeticJourney: Tag;

    @Tag static var hermeticMlxJourney: Tag;

    @Tag static var realModelJourney: Tag;
}
