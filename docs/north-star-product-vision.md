# Astronomical North Star Product Vision

## North Star

A performance optimized local VLM/LLM runner completely optimized for Apple silicon. Built based on the C++ APIs provided from Apple. Astronomical assumes that user is running it for personal use on their Macbook.

## Product principles

- Exceptionally performance optimized.
- Simple to configure on local machine. No hidden environment variables.
- Privacy comes from local execution, not from extra internal layers. This is not a server software that serves many users. This software runs on local laptop and hence any personal data logged or saved remains on the consumer laptop and never sent on the internet. No need for additional security gymnastics.
- Supports running multiple different model architectures, with custom performance optimizations for specific architecture.
- Customized and performance optimized implementations for quantizations like OptiQ, OQe, OQ and others.

## Capability sequence

1. Can run Text Only models
2. Can run Vision enabled models.
3. Provides a performance optimized experience that is not hardwired to a certain memory size.
4. Automatically tunes and adapts itself to provided maximum possible performance.
5. Provides SSD cache that does not loose fidelity or downgrades accuracy or precision of what its storing as a cache.
6. In all parts of this codebase there must not be loss of precision or degradation of model precision. In other words no hidden quantization.

## Supported and Optimized for Apple Silicon M5 and Later versions

- This codebase is targeting M5 apple silicon.
- This codebase targets Macos 26 and higher.
