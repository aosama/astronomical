import Testing;

/// One serialized container for every MLX/GPU journey in this target. Its
/// `.serialized` trait covers sibling hermetic and real-model suites, so
/// model weights and Metal work never overlap. Child suites own their
/// individual journey-category tags.
@Suite(.serialized)
final class MlxGpuJourneyContainer {}
