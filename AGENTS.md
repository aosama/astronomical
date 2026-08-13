# Instructions for the Astronomical Project

- This is our constitution at repo-root/docs/north-star-product-vision.md everything is derived from there.

- You keep repo-root/docs/performance-optimizations-lessons.md updated with lessons learnt about performance relevant to LLMs, VLMs and MLX APIs.

- All commands and tests must emit a live progress indicator instead of leaving the user with silent output.

- NEVER pipe long-running commands through | tail -30, | head, or any filter that buffers output and hides live progress from the user. The user must be able to observe progress as it happens.

- All and any tests must have a built in timeout with a maximum of 120 seconds.

- Astronomical is expected to adapt to any laptop, any RAM size, any GPU wired memory limit. Do not hardwire or optimize the codebase just for this laptop that we are developing in.

- Keep full workspace verification and formatting verification only before committing and pushing when the user asks you to commit. This codebase is slow to format check and do a full workspace/test runs.

- Before every requested commit or push run scripts/verify-before-commit.sh; never substitute cargo test --workspace --all-targets because it runs broad integration binaries serially.

- Run cargo fmt or similar commands only before committing, i.e. when the user asks you to commit then you do the cargo fmt

## Public Repository And GitHub Decorum

- Treat every GitHub issue, pull request, comment, release note, repository setting, and committed file as potentially permanent public material.

- Write GitHub content as concise, project-owned prose in English unless the user explicitly requests another language.

- GitHub issues must describe Astronomical goals, evidence, scope, constraints, and acceptance criteria. Do not publish raw agent opinions, dictation fragments, private working notes, or broad external-project audits.

- Cite public documentation, model cards, standards, and upstream APIs only when they establish compatibility, provenance, or licensing.

- Never publish or commit personal names, user names, email addresses, phone numbers, private repository links, local endpoints, local model inventories, credentials, tokens, or machine-specific logs unless the user explicitly approves the exact disclosure.

- Before creating or editing public GitHub content, review it for local paths, personal details, credentials, stale internal terminology, and wording that suggests unapproved source reuse.

- No written language should convey a negative connutation about a person, entity or a programming library.

## Local Environment Boundaries

- Never hardwire a developer home directory, workstation path, local model path, local endpoint, or machine-specific hardware assumption into production code, tests, fixtures, qualification artifacts, documentation, GitHub content, or agent instructions.

- Resolve user-controlled locations through configuration, environment variables, platform-standard application directories, command-line arguments, or explicit user file selection.

- Tests must use temporary directories, repository fixtures, or clearly fictional placeholder paths that cannot identify a developer workstation.

- When upstream source is needed for research, discover its local location from the active environment. Refer to projects by name in repository content and never persist a personal checkout path.

## There are No Downstream Consumers or Dependencies

- There are no downstream consumers or other dependant applications -- hence no need for deprication or compatibilty shims or any other techniques. Work in a fail forward fashion.

## Coding Principles

- Code files should remain around the 500 lines marker not longer.
- Any end-user-facing file-size or memory value must use decimal SI gigabytes: 1 GB = 1,000,000,000 bytes. Do not show binary GiB values under a GB label.

## No Backward Compatibility for REST API surface

- There is no requirement for Backward compatibility for the REST API surface. There are no downstream consumers of these surfaces so compatibility is not an issue.
- There is no requirement for backward compatibility shims.

## Requirements for Performance Profiling and Attribution

- This codebase needs to be performance optimized, to achieve that, all our code, regardless which part, needs to have performance logging and attribution that can be switched on and off through config parameter. The performance logging needs to capture start time and end time of each operation. this will allow us to ATTRIBUTE performance issues to specific code parts. Without attribution we will be guessing why we are observing a slow down, which is a bad state to be in, hence it is imperative that when you are editing code or refactoring or creating new code that they follow a unified performance logging and attribution pattern.

- The highest priority for performance logging and attribution is the critical path involved in Model Loading from disk, Prompt Processing, Tokenization, Disk Cache, Expert Paging and finally token generation. In short any and all operations invovled in actually serving a model request to the end user.

## Test Fixtures and Their Reuse for Performance And Correctness Tests

- The fixture of Romeo and Juliet MUST be used and the source test input for LLMs, there should not be radnom text or tokens used for testing.
