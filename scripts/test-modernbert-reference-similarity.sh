#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# ModernBERT Reference-Similarity Acceptance Journey — Issue #596
#
# Purpose: Reproduce the upstream similarity-gap defect by sending the
#           exact worked-example inputs from the nomicai-modernbert-embed-base
#           model card to Astronomical's /v1/embeddings endpoint and measuring
#           whether served vectors preserve the published cosine ordering.
#
# Preconditions:
#   • One Development-instance checkpoint is reachable:
#       $ASTRONOMICAL_DEV_STATE_DIR/models/nomicai-modernbert-embed-base-8bit
#     Defaults to ~/.astronomical-dev/models if unset.
#   • An active astronomicald instance listens on port 6733.
#   • No other Astronomical process holds wired GPU memory.
#
# Outputs (written to STDOUT; also recorded in the JSON evidence file):
#   step  message
#   -------------------------------------------
#   probe   public /ready health-check
#   query   embedding request sent
#   cosine  measured cosines for every pair
#   compare published vs observed separation
#   verdict DEFECT or RESOLVED per criterion
#
# Exit codes:
#   0  all criteria pass (no defect detected)
#   1  one-or-more criteria fail (defect reproduced)
#   2  environment or runtime failure (infrastructure)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# --------------- constants ---------------
SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd -P)"

DEFAULT_DEV_STATE="$HOME/.astronomical-dev"
DEV_STATE_DIR="${ASTRONOMICAL_DEV_STATE_DIR:-$DEFAULT_DEV_STATE}"
DAEMON_HOST="http://127.0.0.1:6733"
EVIDENCE_JSON_FILE="${ASTRONOMICAL_ACCEPTANCE_EVIDENCE_DIRECTORY:-}"

UPSTREAM_QUERY_TSNE="search_query: What is TSNE?"
UPSTREAM_QUERY_LAURENS="search_query: Who is Laurens van der Maaten?"
UPSTREAM_DOC="search_document: TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten"

UPSTREAM_TSDOC_SIMILARITY="0.7214"     # TSNE-query vs doc
UPSTREAM_LADOC_SIMILARITY="0.3260"     # Laurens-query vs doc
UPSTREAM_SEPARATION="0.3954"           # first minus second (positive → ordered)

TIMESTAMP_MS="$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))')"
EVIDENCE_PATH=""

# --------------- helpers ---------------
log() { eprintf "  [ref-sim] $*"; }

eprintf() { printf '%s\n' "$*" >&2; }

die() { log "FATAL: $*"; exit 2; }

json_body_from_curl() {
    # Strip HTTP headers (everything before first blank line)
    echo "$1" | sed '/^$/,$d' | tail -n +2
}

compute_cosine() {
    # Given three CSV files: vecA, vecB, vecC (one number per line)
    # Outputs "a_vs_b|a_vs_c|b_vs_c" using awk for precision.
    awk -v fa="$1" -v fb="$2" -v fc="$3" '
        BEGIN {
            split(fa,a," "); split(fb,b," "); split(fc,c," ")
        }
        {
            for(i=1;i<=NF;i++){
                v_a[NR,i]=a[i]; v_b[NR,i]=b[i]; v_c[NR,i]=c[i]
            }
            n=NR
        }
        END {
            d_ab=0; d_ac=0; d_bc=0; na=0; nb=0; nc=0
            for(i=1;i<=n;i++){
                d_ab += v_a[i]*v_b[i]; d_ac += v_a[i]*v_c[i]; d_bc += v_b[i]*v_c[i]
                for(k=i;k<=n;k++) na+=v_a[k]*v_a[k]; na=v_a[i]^2
                for(k=i;k<=n;k++) nb+=v_b[k]*v_b[k]; nb=v_b[i]^2
            }
            sq_a=0; sq_b=0; sq_c=0
            for(i=1;i<=n;i++){sq_a+=v_a[i]^2; sq_b+=v_b[i]^2; sq_c+=v_c[i]^2}
            if(sq_a==0||sq_b==0||sq_c==0){print "NaN|NaN|NaN"; exit}
            c_ab=sqrt(sq_a)*sqrt(sq_b); c_ac=sqrt(sq_a)*sqrt(sq_c); c_bc=sqrt(sq_b)*sqrt(sq_c)
            printf "%.6f|%.6f|%.6f\n", d_ab/c_ab, d_ac/c_ac, d_bc/c_bc
        }
    ' < /dev/null
}

write_json_evidence() {
    local ts="$1"; shift
    local up_tsdoc="$1"; shift
    local up_ladoc="$1"; shift
    local up_sep="$1"; shift
    local obs_tsdoc="$1"; shift
    local obs_ladoc="$1"; shift
    local obs_sep="$1"; shift
    local verdict_tsne="$1"; shift
    local verdict_lauren="$1"; shift
    local verdict_sep="$1"; shift
    local overall="$1"; shift

    mkdir -p "$(dirname "$EVIDENCE_PATH")"

    cat > "$EVIDENCE_PATH" <<EOF
{
  "journey": "modernbert-reference-similarity",
  "version": "0.2.36",
  "timestamp_ms": ${ts},
  "upstream_published": {
    "tsne_vs_doc_similarity": ${up_tsdoc},
    "laurens_vs_doc_similarity": ${up_ladoc},
    "separation_first_minus_second": ${up_sep}
  },
  "observed_onsite": {
    "tsne_vs_doc_similarity": ${obs_tsdoc},
    "laurens_vs_doc_similarity": ${obs_ladoc},
    "separation_first_minus_second": ${obs_sep}
  },
  "criteria": {
    "tsne_similarity_within_tolerance": {"expected_range": "[0.70..0.74]", "actual": ${obs_tsdoc}, "passes": ${verdict_tsne}},
    "laurens_similarity_within_tolerance": {"expected_range": "[0.30..0.36]", "actual": ${obs_ladoc}, "passes": ${verdict_lauren}},
    "ordering_preserved": {"expected_sign": "+", "actual_sign": $(if [ "${obs_sep%.*}" = "-0" ] || [[ "${obs_sep}" == -* ]]; then echo "negative"; else echo "positive"; fi), "expected_value": ${up_sep}, "actual_value": ${obs_sep}, "passes": ${verdict_sep}}
  },
  "overall_verdict": "${overall}",
  "issue_ref": "#596"
}
EOF
}

# --------------- pre-flight checks ---------------
preflight() {
    # Check daemon is alive
    log "phase=preflight probe=/ready"
    local ready_response
    ready_response=$(curl -s --max-time 5 "${DAEMON_HOST}/ready" 2>/dev/null) || true
    if ! echo "$ready_response" | grep -q "HTTP/1.1 200 OK"; then
        die "daemon not healthy on ${DAEMON_HOST}/ready (got: ${ready_response})"
    fi
    log "/ready returned 200 OK — daemon confirmed"
}

# --------------- main journey ---------------
main() {
    log "phase=start model=nomicai-modernbert-embed-base-8bit"

    # Evidence directory
    EVIDENCE_PATH="${ASTRONOMICAL_ACCEPTANCE_EVIDENCE_DIRECTORY:-${REPO_ROOT}/target/acceptance-evidence/modernbert-ref-sim}/${TIMESTAMP_MS}/reference-similarity.json"

    preflight

    # ---- Step 1: Send the three worked-example embeddings ----
    log "phase=query send-three-worked-examples"
    local embed_response
    embed_response=$(curl -s \
        --max-time 120 \
        --header "Content-Type: application/json" \
        -d "{
            \"model\": \"nomicai-modernbert-embed-base-8bit\",
            \"input\": [
                \"${UPSTREAM_QUERY_TSNE}\",
                \"${UPSTREAM_QUERY_LAURENS}\",
                \"${UPSTREAM_DOC}\"
            ]
        }" \
        "${DAEMON_HOST}/v1/embeddings" 2>/dev/null) || die "curl /v1/embeddings failed"

    # Extract raw float arrays from JSON response body
    local json_body
    json_body=$(echo "$embed_response" | python3 -c "import sys,json;d=json.load(sys.stdin);[print(' '.join(str(x) for x in r['embedding'])) for r in d['data']]" 2>/dev/null) || die "could not parse /v1/embeddings response body"

    local tsne_vec ladoc_vec doc_vec
    tsne_vec=$(echo "$json_body" | sed -n '1p')
    ladoc_vec=$(echo "$json_body" | sed -n '2p')
    doc_vec=$(echo "$json_body" | sed -n '3p')

    log "phase=query extracted three 768-dim vectors from response"

    # ---- Step 2: Compute pairwise cosines ----
    log "phase=cosine compute-pairwise-cosines"
    local cosines
    cosines=$(python3 -c "
import math, sys

def dot(a,b): return sum(x*y for x,y in zip(a,b))
def norm(v): return math.sqrt(sum(x*x for x in v))
def cosine(a,b): return dot(a,b)/(norm(a)*norm(b))

vecs = []
for line in sys.stdin:
    vecs.append([float(x) for x in line.strip().split()])

c01 = cosine(vecs[0], vecs[1])  # TSNE vs Laurens
c02 = cosine(vecs[0], vecs[2])  # TSNE vs doc
c12 = cosine(vecs[1], vecs[2])  # Laurens vs doc

print(f'{c02:.6f}|{c12:.6f}|{c02-c12:+.6f}')
" <<< "$doc_vec" <<< "$ladoc_vec" <<< "$tsne_vec" 2>/dev/null) || die "cosine computation failed"

    # Reconstruct using proper multi-input for python3
    local computed
    computed=$(python3 << PYEOF
import math, sys

def dot(a,b): return sum(x*y for x,y in zip(a,b))
def norm(v): return math.sqrt(sum(x*x for x in v))
def cosine(a,b): return dot(a,b)/(norm(a)*norm(b))

lines = """${tsne_vec}
${ladoc_vec}
${doc_vec}""".strip().split('\n')
vecs = [[float(x) for x in line.split()] for line in lines if line.strip()]

if len(vecs) < 3:
    print("NaN|NaN|NaN", file=sys.stderr)
    sys.exit(1)

c02 = cosine(vecs[0], vecs[2])  # TSNE vs doc
c12 = cosine(vecs[1], vecs[2])  # Laurens vs doc

print(f'{c02:.6f}|{c12:.6f}|{c02-c12:+.6f}')
PYEOF
    ) || die "cosine computation failed"

    local obs_tsdoc obs_ladoc obs_sep
    obs_tsdoc=$(echo "$computed" | cut -d'|' -f1)
    obs_ladoc=$(echo "$computed" | cut -d'|' -f2)
    obs_sep=$(echo "$computed" | cut -d'|' -f3)

    log "phase=cosine results: tsne-vs-doc=${obs_tsdoc} laurens-vs-doc=${obs_ladoc} separation=${obs_sep}"

    # ---- Step 3: Compare against upstream reference ----
    log "phase=compare evaluate-upstream-tolerance"

    # Criterion 1: TSNE vs doc within tolerance band (0.7214 ± 0.03)
    local verdict_tsne
    verdict_tsne=$(python3 -c "
val=float('${obs_tsdoc}'); lo=0.7214-0.03; hi=0.7214+0.03
print('true' if lo<=val<=hi else 'false')
" 2>/dev/null) || verdict_tsne="false"

    # Criterion 2: Laurens vs doc within tolerance band (0.3260 ± 0.03)
    local verdict_lauren
    verdict_lauren=$(python3 -c "
val=float('${obs_ladoc}'); lo=0.3260-0.03; hi=0.3260+0.03
print('true' if lo<=val<=hi else 'false')
" 2>/dev/null) || verdict_lauren="false"

    # Criterion 3: Ordering preserved (separation positive, ≥ 0.30)
    local verdict_sep
    verdict_sep=$(python3 -c "
val=float('${obs_sep}')
print('true' if val>=0.30 else 'false')
" 2>/dev/null) || verdict_sep="false"

    log "phase=compare criteria: tsne_tol=${verdict_tsne} lauren_tol=${verdict_lauren} sep_ordered=${verdict_sep}"

    # Determine overall verdict
    local overall
    if [ "$verdict_tsne" = "true" ] && [ "$verdict_lauren" = "true" ] && [ "$verdict_sep" = "true" ]; then
        overall="RESOLVED"
    else
        overall="DEFECT"
    fi

    # Write evidence
    write_json_evidence \
        "$TIMESTAMP_MS" \
        "$UPSTREAM_TSDOC_SIMILARITY" \
        "$UPSTREAM_LADOC_SIMILARITY" \
        "$UPSTREAM_SEPARATION" \
        "$obs_tsdoc" \
        "$obs_ladoc" \
        "$obs_sep" \
        "$verdict_tsne" \
        "$verdict_lauren" \
        "$verdict_sep" \
        "$overall"

    log "phase=evidence written ${EVIDENCE_PATH}"

    # ---- Step 4: Print summary table ----
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  ModernBERT Reference-Similarity Journey Results"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    printf "%-30s  %-10s  %-10s\n" "Pair" "Published" "Observed"
    echo "───────────────────────────────────────────────────"
    printf "%-30s  %-10s  %-10s\n" "TSNE vs Doc" "${UPSTREAM_TSDOC_SIMILARITY}" "${obs_tsdoc}"
    printf "%-30s  %-10s  %-10s\n" "Laurens vs Doc" "${UPSTREAM_LADOC_SIMILARITY}" "${obs_ladoc}"
    printf "%-30s  %-10s  %-10s\n" "Separation (+)" "${UPSTREAM_SEPARATION}" "${obs_sep}"
    echo ""
    printf "%-30s  %-10s  %-10s\n" "Criterion" "Expected" "Pass"
    echo "───────────────────────────────────────────────────"
    printf "%-30s  %-10s  %-10s\n" "TSNE within tolerance" "[0.69..0.75]" "${verdict_tsne}"
    printf "%-30s  %-10s  %-10s\n" "Laurens within tolerance" "[0.29..0.36]" "${verdict_lauren}"
    printf "%-30s  %-10s  %-10s\n" "Ordering preserved" "sep >= +0.30" "${verdict_sep}"
    echo ""
    printf "Overall: %-10s\n" "$overall"
    echo "Evidence: $EVIDENCE_PATH"
    echo ""

    case "$overall" in
        RESOLVED) exit 0 ;;
        DEFECT)   exit 1 ;;
        *)        exit 2 ;;
    esac
}

main "$@"
