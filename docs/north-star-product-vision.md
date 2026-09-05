# Astronomical North Star Product Vision

## North Star

A performance optimized local VLM/LLM runner completely optimized for Apple silicon. Built based on the C++ APIs provided from Apple. Astronomical assumes that user is running it for personal use on their Macbook.

## Product principles

- Simple to configure on local machine. No hidden environment variables.
- Privacy comes from local execution, not from extra internal layers. This is not a server software that serves many users. This software runs on local laptop and hence any personal data logged or saved remains on the consumer laptop and never sent on the internet. No need for additional security gymnastics.
- Supports running multiple different model architectures, with custom performance optimizations for specific architecture.

## Platform contract

Correctness is the baseline; speed is opt-in per machine.

- Baseline correctness: every supported request completes with the precision the model artifact declares, on any Apple Silicon M-series Mac running macOS 14.0 or later. Memory pressure, SSD streaming, and unsupported hardware features must never silently reduce precision.
- Probe-gated boosters: custom Metal kernels are optional fast paths, proven per worker by compile plus a bounded execution probe with expected-value validation. A GPU that cannot run a kernel still serves the same request through the equivalent public MLX API with the same declared precision. Verdicts are capability-based; chip marketing names never gate features.
- Support spectrum: the operating-system floor follows the lowest macOS the bundled MLX stack supports (14.0 today). As an open-source project without a physical device matrix, Astronomical declares best-effort support across this spectrum, discloses which macOS × chip combinations were actually tested, and treats community defect reports on untested combinations as first-class evidence. Capability probes and fallback paths, not chip or operating-system names, protect correctness on combinations the project has never touched.

## Capability sequence

1. Can run Text Only models
2. Can run Vision enabled models.
3. Provides a performance optimized experience that is not hardwired to a certain memory size.
4. Automatically tunes and adapts itself to provided maximum possible performance.
5. Provides SSD cache that does not loose fidelity or downgrades accuracy or precision of what its storing as a cache.
6. In all parts of this codebase there must not be loss of precision or degradation of model precision. In other words no hidden quantization.
7. Correctness holds on every supported Apple Silicon Mac and macOS version; measured speedups may differ per GPU generation, but the served precision never does.
