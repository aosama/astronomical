# Instructions for the Astronomical Project

- This is our constitution at repo-root/docs/north-star-product-vision.md everything is derived from there.
- You keep repo-root/docs/performance-optimizations-lessons.md updated with lessons learnt about performance relevant to LLMs, VLMs and MLX APIs.
- The user is ALWAYS looking for you to **proactively** suggest how to reduce over engineering and reduce code complexity.
- Proactively suggest to the user ways to reduce the number of abstractions and layers in this codebase.
- All commands and tests must emit a live progress indicator instead of leaving the user with silent output.
- NEVER pipe long-running commands through | tail -30, | head, or any filter that buffers output and hides live progress from the user. The user must be able to observe progress as it happens.
- All and any tests must have a built in timeout with a maximum of 120 seconds.
- Rust takes a longer time to compile as comapred to other languages, ALWAYS proactively look for opportunities to reduce the overall build/compile time without sacreficing performance.
- Astronomical is expected to adapt to any laptop, any RAM size, any GPU wired memory limit. Do not hardwire or optimize the codebase just for this laptop that we are developing in.
- Keep full workspace verification and formatting verification only before committing and pushing when the user asks you to commit. This codebase is slow to format check and do a full workspace/test runs.
- Before every requested commit or push run scripts/verify-before-commit.sh; never substitute cargo test --workspace --all-targets because it runs broad integration binaries serially.
- Run cargo fmt or similar commands only before committing, i.e. when the user asks you to commit then you do the cargo fmt
- Never insert or write the backtick character in any Markdown file. Use plain text, quotation marks, indentation, or another fit-for-purpose representation instead.

## Public Repository And GitHub Decorum

- Treat every GitHub issue, pull request, comment, release note, repository setting, and committed file as potentially permanent public material.
- Write GitHub content as concise, project-owned prose in English unless the user explicitly requests another language.
- GitHub issues must describe Astronomical goals, evidence, scope, constraints, and acceptance criteria. Do not publish raw agent opinions, dictation fragments, private working notes, or broad external-project audits.
- Cite public documentation, model cards, standards, and upstream APIs only when they establish compatibility, provenance, or licensing. Do not describe Astronomical as copied, ported, or derived from another implementation unless the user has explicitly approved actual source reuse and the required attribution is recorded.
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
- Opt for and Prefer and Utilize simple designs with fewer abstractions that elaborate layers of facades and abstractions.
- Prioritize simple architecture versus elaborate abstractions.
- Prioritize reduction of layers and abstractions while you are working.
- Any end-user-facing file-size or memory value must use decimal SI gigabytes: 1 GB = 1,000,000,000 bytes. Do not show binary GiB values under a GB label.

## Keep Dependencies Updated to Latest Version

- Always make sure we are using the latest dependencies for this project.

## Naming Convensions in Variables

- prefill_chunck_tokens is the correct variable name to use for that intended purpose duing prompt processing. Never write or declare variable names like tokens or chunck_tokens when the intended purpose is prefill_chunck_tokens. Congruency and verbosity in variable names is a mandate.

- The term cache must never be used in abstraction in variable names, this codebase has many types of caches such as persistent_cache and various other caches in memory and for different purposes. Varaible names must be fully qualify what type of cache the code is talking about.

- The term key must never be used in abstraction, is it door_key or ssh_key or is it the key to your heart :)? you must always remediate the codebase to use fully qualified and verbose variable names. The same applies to value .. is it monetary_value is it gold_value :) I hope you get where I am going with this, disambiguate to the maximum in varaible names. When you see crap like this remediate it immediately.

- Long variable names and class names are accepted and the user prefers long communicative names to brief ambigous names.

## Useful Codebases

- MLX is the core Apple Silicon array framework and the source of truth for its behavior.
- MLX-C provides the foundational C bindings used by this project.
- When optimizing MLX behavior, inspect the available upstream MLX and MLX-C source before proposing new native operations. Independently implement only the behavior Astronomical needs and preserve any required attribution for actual source reuse.

## Building The MACOS Application

- Rust optimized builds must be used across all the stack when building the final macos application.

## No Backward Compatibility for Cache or REST API surfaces

- There is no requirement for Backward compatibility for the cache or REST API surfaces. There are no downstream consumers of these surfaces so compatibility is not an issue.
- There is no requirement for backward compatibility shims.

## No Over Engineering or Over Architecting

- This is not a software to launch a rocket to outer space, this is NOT a medical grade software, this is a software that a programmer or a student will run locally to get basic work done.
- This is NOT a profressional grade server application, PLEASE DO NOT OVER ENGINEER SOLUTIONS.
- This is NOT a mission critical software, this will run on a laptop for personal use or study work.
- This is NOT an enterprise grade software, this is a personal use software.

## Requirements for Performance Profiling and Attribution

- This codebase needs to be performance optimized, to achieve that, all our code, regardless which part, needs to have performance logging and attribution that can be switched on and off through config parameter. The performance logging needs to capture start time and end time of each operation. this will allow us to ATTRIBUTE performance issues to specific code parts. Without attribution we will be guessing why we are observing a slow down, which is a bad state to be in, hence it is imperative that when you are editing code or refactoring or creating new code that they follow a unified performance logging and attribution pattern.
- The highest priority for performance logging and attribution is the critical path involved in Model Loading from disk, Prompt Processing, Tokenization, Disk Cache, Expert Paging and finally token generation. In short any and all operations invovled in actually serving a model request to the end user.
- The minimum representative performane test needs to intake 1K tokens and output 512 tokens with a plus or minus 15% allowance. Anything below that is not a representative measurement at all.

## Maximize the Use of the GPU versus CPU

- In this code base, the user of the GPU should be adopted to the maximum extent possible, it has been observed that the more computations that can be offloaded to the GPU, the better the performance will be.
- Do not assume that a certain GPU operation will be slower or will not yield more performance than the CPU.
