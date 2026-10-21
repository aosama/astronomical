#!/usr/bin/env sh

# Proves every cargo command cited inside tracked markdown (fenced code blocks
# and inline backtick spans only — prose like "cargo runs" is not a citation)
# resolves to either a stock cargo subcommand or an alias defined in
# .cargo/config.toml, so documentation cannot drift onto retired command
# names the way the release procedure once did. A detector self-check with a
# synthetic fixture fails the contract if the extractor or the validator ever
# stops catching stale citations. A watchdog bounds the whole contract below
# the 120-second test ceiling.

set -eu

TEST_WATCHDOG_SECONDS=110
WATCHDOG_PID=""
SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${WATCHDOG_PID:-}" ]; then
        kill -9 "$WATCHDOG_PID" 2>/dev/null || true
        wait "$WATCHDOG_PID" 2>/dev/null || true
    fi
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe citation-test sandbox" ;;
            *) rm -rf "$SANDBOX_DIRECTORY" ;;
        esac
    fi
}
trap cleanup 0

require_command() {
    command_name="$1"
    command -v "$command_name" >/dev/null 2>&1 || {
        print_error "required command is unavailable: ${command_name}"
        exit 2
    }
}

repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd -P)"

require_command git
require_command awk

( sleep "$TEST_WATCHDOG_SECONDS" && kill -9 "$$" ) 2>/dev/null &
WATCHDOG_PID=$!

# The alias surface of this repository: every key under [alias]. Aliases are
# string-valued (verification/tooling_contracts/test-cargo-alias-nested-config-contract.sh
# explains why), so each key line reads `name = "..."`.
ALIAS_WORDS="$(awk '
    /^\[alias\]$/ { inside_alias_section = 1; next }
    /^\[/ { inside_alias_section = 0 }
    inside_alias_section && /^[a-z][a-z0-9-]* = "/ { print $1 }
' "${repository_root}/.cargo/config.toml" | tr '\n' ' ')"

is_stock_cargo_command() {
    stock_command_word="$1"
    case "$stock_command_word" in
        build|check|clean|doc|fetch|fmt|help|init|install|login|logout|metadata|new|owner|package|publish|report|run|rustc|rustdoc|search|test|tree|update|version|yank|verify-project) return 0 ;;
        # Plugin subcommands provided by this repository's pinned toolchain
        # (cargo-about via install-verification-tools, clippy via rustup).
        about|clippy) return 0 ;;
        *) return 1 ;;
    esac
}

is_citable_command_word() {
    candidate_command_word="$1"
    if is_stock_cargo_command "$candidate_command_word"; then
        return 0
    fi
    case " ${ALIAS_WORDS} " in
        *" ${candidate_command_word} "*) return 0 ;;
        *) return 1 ;;
    esac
}

# Emits the command word of every cargo citation found in fenced code blocks
# and inline backtick spans; bare prose mentions are deliberately ignored.
extract_cited_command_words() {
    # The parameter name is deliberately distinct from the sweep loop's
    # markdown_file variable: sh functions have no locals, so a shared name
    # would clobber the caller's value.
    source_markdown_path="$1"
    awk '
        function emit_cited_command_words(text) {
            remaining = text
            while (match(remaining, /cargo [a-z][a-z0-9-]*[a-z0-9]/)) {
                print substr(remaining, RSTART + 6, RLENGTH - 6)
                remaining = substr(remaining, RSTART + RLENGTH)
            }
        }
        /^```/ { inside_fence = !inside_fence; next }
        inside_fence { emit_cited_command_words($0); next }
        {
            line = $0
            while (match(line, /`[^`]+`/)) {
                # RSTART is global in awk, so the boundaries must be captured
                # before emit_cited_command_words runs its own matches.
                span_start = RSTART
                span_length = RLENGTH
                emit_cited_command_words(substr(line, span_start + 1, span_length - 2))
                line = substr(line, span_start + span_length)
            }
        }
    ' "$source_markdown_path"
}

SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-cargo-citations.XXXXXX")"
FIXTURE_MARKDOWN="${SANDBOX_DIRECTORY}/fixture.md"
EXTRACTED_WORDS_FILE="${SANDBOX_DIRECTORY}/extracted-words.txt"
VIOLATIONS_FILE="${SANDBOX_DIRECTORY}/violations.txt"

printf '%s\n' '[cargo-citation-contract] phase=detector-self-check status=start'
cat >"$FIXTURE_MARKDOWN" <<'FIXTURE'
Prose mentions must stay invisible: cargo proseword never happened here.

```sh
cargo build --release
cargo retired-fixture-alias --flag
```

Run `cargo verify` before pushing, then `cargo another-retired-alias`.
FIXTURE

extract_cited_command_words "$FIXTURE_MARKDOWN" >"$EXTRACTED_WORDS_FILE"
expected_extracted_words="$(printf '%s\n' build retired-fixture-alias verify another-retired-alias)"
actual_extracted_words="$(sort "$EXTRACTED_WORDS_FILE")"
if [ "$actual_extracted_words" != "$(printf '%s\n' "$expected_extracted_words" | sort)" ]; then
    print_error "extractor self-check failed"
    print_error "expected words: ${expected_extracted_words}"
    print_error "actual words:   ${actual_extracted_words}"
    exit 1
fi
if is_citable_command_word "retired-fixture-alias"; then
    print_error "validator self-check failed: a retired alias must not resolve"
    exit 1
fi
if ! is_citable_command_word "build"; then
    print_error "validator self-check failed: a stock subcommand must resolve"
    exit 1
fi
if ! is_citable_command_word "verify"; then
    print_error "validator self-check failed: a repository alias must resolve"
    exit 1
fi
printf '%s\n' '[cargo-citation-contract] phase=detector-self-check status=success'

printf '%s\n' '[cargo-citation-contract] phase=repository-sweep status=start'
citation_count=0
: >"$VIOLATIONS_FILE"
for markdown_file in $(git -C "$repository_root" ls-files '*.md' | sort); do
    extract_cited_command_words "${repository_root}/${markdown_file}" >"$EXTRACTED_WORDS_FILE"
    while IFS= read -r command_word; do
        [ -n "$command_word" ] || continue
        citation_count=$((citation_count + 1))
        if ! is_citable_command_word "$command_word"; then
            printf '%s cites unknown cargo command "%s"\n' "$markdown_file" "$command_word" >>"$VIOLATIONS_FILE"
        fi
    done <"$EXTRACTED_WORDS_FILE"
done

# Floor: the live sweep must still see a substantial citation population, so a
# silent extractor regression cannot pass the contract vacuously.
MINIMUM_EXPECTED_CITATIONS=40
if [ "$citation_count" -lt "$MINIMUM_EXPECTED_CITATIONS" ]; then
    print_error "sweep found only ${citation_count} citations; the extractor is likely broken (floor is ${MINIMUM_EXPECTED_CITATIONS})"
    exit 1
fi

violation_line_count="$(wc -l <"$VIOLATIONS_FILE" | tr -d ' ')"
if [ "$violation_line_count" -ne 0 ]; then
    print_error "${violation_line_count} markdown cargo citation(s) do not resolve:"
    cat "$VIOLATIONS_FILE" >&2
    exit 1
fi
printf '%s\n' "[cargo-citation-contract] phase=repository-sweep status=success citations=${citation_count}"

printf '%s\n' "cargo-citation contract tests passed"
