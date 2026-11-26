# Instructions for the Astronomical Project

## General Instructions

- Methods and functions must not exceed 80 lines of code. The pre-commit hook enforces it through `cargo xtask function-length-budget` (clippy's `too_many_lines` at 80 code lines, set in `clippy.toml`): split a long function into named steps, and never put `#[allow]` or `#[expect]` of `clippy::too_many_lines` on any function, production or test, which the same check refuses.

## Never Hardwire Local Developer Environment Into the Codebase

- Never hardwire a developer home directory, workstation path, local model path, local endpoint, or machine-specific hardware assumption into production code, tests, fixtures, acceptance artifacts, documentation, GitHub content, or agent instructions.

## There are No RESTAPI/IPC/Unix Socket Downstream Consumers

- There are no downstream consumers or other dependent applications — hence no need for deprecation or compatibility shims or any other techniques. There is no requirement for backward compatibility for the REST API surface. 

## Prohibited Terminology: "qualification" and "hermetic"

- Never use "qualification" or any variant ("qualify", "qualified", etc.) in code, comments, tests, scripts, documentation, commit messages, GitHub content, or replies.
- "hermetic" is equally banned and being decommissioned (issue #1208). Remove existing occurrences and replace with the most specific term below.

## Testing Instructions For Astronomical

- To give Development a model, symlink a copy already on this Mac into `~/.astronomical-dev/models/<publisher>/<model>`, looking in Stable's `~/.astronomical/models` first. When no copy exists, download it into `~/.astronomical/models/<publisher>/<model>` and symlink that. Never download into the Hugging Face cache or through the Development app.

- When your work is done, move the contents of Development's `astro_ssd_prompt_cache` folder, `~/.astronomical-dev/cache`, to the Trash.

- Tests that use real full sized models and loads them in memory for further use by the GPU must use the Rust `serial_test` `#[serial]` annotation; 

- Under `crates/` and `apps/`, a test that uses the GPU takes `serial_test`'s `#[file_serial]` instead of `#[serial]`: nextest runs every test in a process of its own, and only `#[file_serial]` locks across processes (issue #2539).

- The fixture of Shakespear is highly recommended to be used as the input for LLMs; there should not be random text or tokens used for testing. Exceptions can be made where Shakespear is not suitable for the test -- for example a coding fixture.

## Performance Instructions

- A minimal warmup run should be performed before the actual test; t
- The astro SSD prompt cache must be disabled during performance throughput tests and measurements.
- The skill declaration (`skill.declare_to_models`) must be disabled during performance throughput tests and measurements, because it adds tokens to every prompt.

## Release Scripts Folder

- All **Release-related scripts** must live under "<repo-root>/scripts/release", they will help you through the release process. This doesnt mean you have to create shell scripts at all, if you can do what you need in xtask then go ahead.
- Follow the `sparkle_release` skill (`.claude/skills/sparkle_release/SKILL.md`) for every Stable release on the direct download and Sparkle feed, and read all of it before the first step.
- A Mac App Store release is separate and happens only when the user asks for it: follow the `macos_appstore_release` skill (`.claude/skills/macos_appstore_release/SKILL.md`).

## Repository Tooling Lives In xtask

- `<repo-root>/xtask` exists so that repository tooling is typed, unit-tested Rust invoked as `cargo xtask <command>` instead of untested shell, with each subcommand keeping the output lines and exit codes of the shell tool it replaced so hooks and CI steps survive unchanged. Whenever you add or change such tooling, put it in xtask as a new or extended subcommand with its tests and document the invocation in the `xtask` alias comment of `.cargo/config.toml`;
- This repo prefers <repo-root>/xtask convention versus shell scripts, shell scripts are highly discouraged and the repo is moving off of shell scripts and reducing their usage.

## One Gigabyte Everywhere

- A gigabyte in Astronomical is 1,073,741,824 bytes (2^30), labelled "GB", the one Apple uses when it describes a Mac's memory. Every screen, message, log summary, test printout and reply to the user divides bytes by 1,073,741,824, never by 1,000,000,000, so a figure always matches what the menu bar and Observatory show.

## The Memory Areas Have Formal Names

- You MUST read `<repo-root>/docs/glossary.md` before you read, write, review or discuss anything about memory, MLX, caching or streaming. It defines every memory area and every cache by one formal name. 
- Never write the bare word "cache": every mention says which cache it means, by its formal name. Use those names in code, comments, tests, logs, documentation, GitHub content and replies. Screens keep their plain words; the glossary lists which is which.
- When you find a cache or memory area under another name, or a bare "cache", rename it to the formal name as part of your work.

## Hints About This Repo

- Pins and patches are under "<repo-root>/third-party".
- Supported models are under "<repo-root>/registry".
- Performance measurements tests are under "<repo-root>/verification".
- Release speed is recorded in one ledger file a release under "<repo-root>/registry/release_speed", written only by `cargo xtask measure-speed --reference-set --write-ledger`.
- There is no requirements for phone browsing for the web user interface, this is a macos desktop only codebase.
