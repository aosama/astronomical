#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# ModernBERT Tokenizer-Gap Acceptance Journey — Issue #596
#
# Purpose: Prove that the artifact's tokenizer.json declares a
#          TemplateProcessing post-processor wrapping text with [CLS]/[SEP],
#          yet Astronomical encodes with add_special_tokens=false so those
#          tokens never enter the encoder sequence.
#
# Preconditions:
#   • A ModernBERT checkpoint with tokenizer.json is locatable through
#     $MODERNBERT_CHECKPOINT_PATH or the standard Astronomical model roots.
#   • Python 3 with the `tokenizers` wheel installed.
#
# Exit codes:
#   0  post-processor has no effect on any example → no gap detected
#   1  one-or-more examples differ → gap confirmed (DEFECT)
#   2  environment failure (missing dependency / checkpoint)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd -P)"

WORKED_EXAMPLE_QUERY_TSNE="search_query: What is TSNE?"
WORKED_EXAMPLE_QUERY_LAURENS="search_query: Who is Laurens van der Maaten?"
WORKED_EXAMPLE_DOC="search_document: TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten"

log() { printf '%s\n' "[tokenizer-gap] $*" >&2; }
die() { log "FATAL: $*"; exit 2; }

locate_checkpoint() {
    if [ -n "${MODERNBERT_CHECKPOINT_PATH:-}" ]; then
        [ -d "$MODERNBERT_CHECKPOINT_PATH" ] || die "override path does not exist: ${MODERNBERT_CHECKPOINT_PATH}"
        printf '%s' "$MODERNBERT_CHECKPOINT_PATH"
        return
    fi
    local candidate
    for candidate in \
        "$HOME/.astronomical-dev/models/mlx-community/nomicai-modernbert-embed-base-8bit" \
        "$HOME/.astronomical-dev/models/nomicai-modernbert-embed-base-8bit" \
        "$HOME/.astronomical/models/mlx-community/nomicai-modernbert-embed-base-8bit" \
        "$HOME/.astronomical/models/nomicai-modernbert-embed-base-8bit"
    do
        if [ -d "$candidate" ]; then
            printf '%s' "$candidate"
            return
        fi
    done
    die "ModernBERT checkpoint not found; set \$MODERNBERT_CHECKPOINT_PATH or install the artifact"
}

main() {
    log "phase=start"

    local checkpoint_directory
    checkpoint_directory="$(locate_checkpoint)"
    log "phase=preflight checkpoint=${checkpoint_directory}"

    [ -f "${checkpoint_directory}/tokenizer.json" ] || die "missing ${checkpoint_directory}/tokenizer.json"

    python3 -c "import tokenizers" 2>/dev/null || die "Python package 'tokenizers' not installed; run: pip install tokenizers"
    log "phase=preflight tokenizers-library confirmed"

    local result
    result=$(python3 - "$checkpoint_directory/tokenizer.json" \
        "$WORKED_EXAMPLE_QUERY_TSNE" \
        "$WORKED_EXAMPLE_QUERY_LAURENS" \
        "$WORKED_EXAMPLE_DOC" << 'PYEOF'
import json, sys
from tokenizers import Tokenizer as HfTokenizer

tokenizer_path = sys.argv[1]
queries = sys.argv[2:5]

with open(tokenizer_path) as tokenizer_file:
    tokenizer_document = json.load(tokenizer_file)

tokenizer = HfTokenizer.from_str(json.dumps(tokenizer_document))

special_tokens = tokenizer_document.get("post_processor", {}).get("special_tokens", {})
if not special_tokens:
    special_tokens = tokenizer_document.get("special_tokens", {})
cls_token_id = special_tokens.get("[CLS]", {}).get("ids", [None])[0]
sep_token_id = special_tokens.get("[SEP]", {}).get("ids", [None])[0]
pad_token_id = special_tokens.get("[PAD]", {}).get("ids", [None])[0]

has_post_processor = tokenizer_document.get("post_processor") is not None

table_rows = []
gap_detected = False

for query_text in queries:
    # add_special_tokens=False reproduces Astronomical's encode(text, false).
    encoding_without_specials = tokenizer.encode(query_text, add_special_tokens=False)
    # add_special_tokens=True applies the artifact-declared post-processor.
    encoding_with_specials = tokenizer.encode(query_text, add_special_tokens=True)

    ids_without = encoding_without_specials.ids
    ids_with = encoding_with_specials.ids
    token_count_difference = len(ids_with) - len(ids_without)

    specials_absent_without = cls_token_id not in ids_without and sep_token_id not in ids_without
    specials_present_with = cls_token_id in ids_with and sep_token_id in ids_with

    if token_count_difference > 0 and specials_absent_without and specials_present_with:
        gap_detected = True

    table_rows.append((
        query_text[:48],
        len(ids_without),
        len(ids_with),
        token_count_difference,
        str(specials_present_with).lower(),
        str(specials_absent_without).lower(),
    ))

print(f"HAS_POST_PROCESSOR={str(has_post_processor).lower()}")
print(f"CLS_TOKEN_ID={cls_token_id}")
print(f"SEP_TOKEN_ID={sep_token_id}")
print(f"PAD_TOKEN_ID={pad_token_id}")
print("---TABLE_START---")
for row in table_rows:
    print("|".join(str(value) for value in row))
print("---TABLE_END---")
print(f"GAP_DETECTED={str(gap_detected).lower()}")
PYEOF
    ) || die "Python verifier failed"

    local has_post_processor cls_token_id sep_token_id pad_token_id gap_detected table_data
    has_post_processor=$(printf '%s\n' "$result" | grep '^HAS_POST_PROCESSOR=' | cut -d= -f2)
    cls_token_id=$(printf '%s\n' "$result" | grep '^CLS_TOKEN_ID=' | cut -d= -f2)
    sep_token_id=$(printf '%s\n' "$result" | grep '^SEP_TOKEN_ID=' | cut -d= -f2)
    pad_token_id=$(printf '%s\n' "$result" | grep '^PAD_TOKEN_ID=' | cut -d= -f2)
    gap_detected=$(printf '%s\n' "$result" | grep '^GAP_DETECTED=' | cut -d= -f2)
    table_data=$(printf '%s\n' "$result" | sed -n '/^---TABLE_START---/,/^---TABLE_END---/p' | grep -v '^---')

    log "phase=analysis post_processor=${has_post_processor} cls=${cls_token_id} sep=${sep_token_id} pad=${pad_token_id}"

    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  ModernBERT Tokenizer-Gap Analysis — Issue #596"
    echo "════════════════════════════════════════════════════════════════════════"
    echo ""
    printf "  %-50s %11s %13s %6s %9s %11s\n" \
        "Input" "Served(none)" "Declared(+)" "Diff" "Specials+" "Absent-none"
    echo "  ────────────────────────────────────────────────────────────────────────────────"
    while IFS='|' read -r query_text served_count declared_count difference specials_with absent_without; do
        printf "  %-50s %11s %13s %6s %9s %11s\n" \
            "${query_text}" "${served_count}" "${declared_count}" "${difference}" "${specials_with}" "${absent_without}"
    done <<< "$table_data"
    echo ""
    echo "  Artifact post-processor declared: ${has_post_processor}"
    echo "  [CLS]=${cls_token_id}  [SEP]=${sep_token_id}  [PAD]=${pad_token_id}"
    echo ""

    if [ "$gap_detected" = "true" ]; then
        echo "  Verdict: GAP CONFIRMED — served sequences omit [CLS]/[SEP] while the"
        echo "           artifact declares TemplateProcessing [CLS] → content → [SEP]."
        echo ""
        echo "  Served path:   crates/model-serving/src/modernbert/tokenizer.rs encode(text, false)"
        echo "  Declared path: tokenizer.json post_processor (encode(text, true))"
        echo ""
        exit 1
    fi

    echo "  Verdict: no gap — served encoding matches the artifact declaration."
    echo ""
    exit 0
}

main "$@"
