// The reply summary on Under the hood: a headline naming the model, the
// memory limit and the speed, three plain facts, and one column per token of
// the last reply. Its developer details keep the whole Mac's memory on one
// bar with the kernel's wire limit drawn on it and four engineering counters.
// The shareable picture (memory-page-image.js) and Home's "Right now" card
// (overview-right-now.js) are drawn from the same two documents. Text is set
// through textContent only; never innerHTML.

const MEMORY_PAGE_BYTES_PER_GIGABYTE = 1073741824;

let memoryPageStatusDocument = null;
let memoryPageExpertStatsDocument = null;

// What each part of the bar is called on screen, in the order drawn.
const MEMORY_PAGE_BAR_PARTS = [
    { key: "other", label: "macOS and other apps" },
    { key: "fixed", label: "Always in memory" },
    { key: "experts", label: "Experts kept" },
    { key: "conversation", label: "This conversation" },
    { key: "working", label: "Working room" }
];

function formatMemoryPageGigabytes(byteCount) {
    return (byteCount / MEMORY_PAGE_BYTES_PER_GIGABYTE).toFixed(1) + " GB";
}

// The memory limit in force, in whole gigabytes, or 0 when not reported.
// The limit, not the Mac's size, is what the claim is about: most people give
// a model the 8 GB a 16 GB Mac can spare, and a "48 GB Mac" headline told them
// the trick needed a large Mac.
function memoryPageLimitGigabytes(statusDocument) {
    const limitBytes = statusDocument.mlx_memory_ceiling_bytes || 0;
    return limitBytes > 0 ? Math.round(limitBytes / MEMORY_PAGE_BYTES_PER_GIGABYTE) : 0;
}

// The claim at the top, built only from what is known: a part with no
// figure behind it is left out, never shown as zero. Without a model there is
// no claim, and the page shows its invitation instead of a bare "8 GB limit".
function memoryPageHeadline(statusDocument) {
    if (!(statusDocument.ready_model_size_bytes > 0)) return "";
    const headlineParts = [formatMemoryPageGigabytes(statusDocument.ready_model_size_bytes) + " model"];
    const limitGigabytes = memoryPageLimitGigabytes(statusDocument);
    if (limitGigabytes > 0) {
        headlineParts.push(limitGigabytes + " GB limit");
    }
    const tokensPerSecond = statusDocument.serving_session?.average_generation_tok_per_second || 0;
    if (tokensPerSecond > 0) {
        headlineParts.push(Math.round(tokensPerSecond) + " tokens a second");
    }
    return headlineParts.join(", ");
}

// How the Mac's memory is split, from the engine's ledger and what macOS
// reports as wired. Other apps are whatever is wired that MLX does not hold.
function memoryPageBarBytes(mlxMemorySnapshot) {
    const memoryLedger = mlxMemorySnapshot.memory_ledger;
    const activeBytes = mlxMemorySnapshot.active_memory_bytes || 0;
    const mlxBufferCacheBytes = mlxMemorySnapshot.mlx_buffer_cache_bytes || 0;
    const fixedBytes = Math.min(memoryLedger.core_weight_bytes || 0, activeBytes);
    const expertBytes = Math.min(memoryLedger.held_kept_weight_bytes || 0, activeBytes - fixedBytes);
    const conversationBytes = Math.min(memoryLedger.held_context_bytes || 0, activeBytes - fixedBytes - expertBytes);
    const workingBytes = activeBytes - fixedBytes - expertBytes - conversationBytes + mlxBufferCacheBytes;
    const otherBytes = Math.max(0, (memoryLedger.system_wired_bytes || 0) - activeBytes - mlxBufferCacheBytes);
    return { other: otherBytes, fixed: fixedBytes, experts: expertBytes, conversation: conversationBytes, working: workingBytes };
}

function memoryPageBar(statusDocument) {
    const mlxMemorySnapshot = statusDocument.mlx_memory_snapshot;
    const machineMemoryBytes = statusDocument.machine_mlx_memory_ceiling_bytes || 0;
    if (!mlxMemorySnapshot?.memory_ledger || !(machineMemoryBytes > 0)) return null;
    const barBytes = memoryPageBarBytes(mlxMemorySnapshot);
    const wireLimitBytes = mlxMemorySnapshot.memory_ledger.system_wire_limit_bytes || 0;
    return {
        parts: MEMORY_PAGE_BAR_PARTS.map((barPart) => ({
            key: barPart.key,
            label: barPart.label,
            text: formatMemoryPageGigabytes(barBytes[barPart.key]),
            // Under a twentieth of a gigabyte the part would read "0.0 GB".
            isEmpty: barBytes[barPart.key] < MEMORY_PAGE_BYTES_PER_GIGABYTE / 20,
            widthPercent: Math.min(100, (barBytes[barPart.key] / machineMemoryBytes) * 100)
        })),
        wireLimitPercent: wireLimitBytes > 0 ? Math.min(100, (wireLimitBytes / machineMemoryBytes) * 100) : null,
        wireLimitText: wireLimitBytes > 0 ? "macOS wire limit · " + formatMemoryPageGigabytes(wireLimitBytes) : ""
    };
}

function memoryPageWireLimitFact(memoryLedger) {
    const wireLimitBytes = memoryLedger?.system_wire_limit_bytes || 0;
    if (!(wireLimitBytes > 0)) return { label: "Under the wire limit", value: "—" };
    const roomBytes = wireLimitBytes - (memoryLedger.system_wired_bytes || 0);
    return roomBytes >= 0
        ? { label: "Under the wire limit", value: formatMemoryPageGigabytes(roomBytes) }
        : { label: "Over the wire limit", value: formatMemoryPageGigabytes(-roomBytes) };
}

function memoryPageFacts(statusDocument, expertStatsDocument) {
    const residency = expertStatsDocument?.residency;
    const lastReply = expertStatsDocument?.last_reply;
    const averageStorageReads = memoryPageAverageStorageReads(lastReply);
    const routedExpertCount = lastReply?.routed_expert_count_per_token || 0;
    const sampledTokenCount = lastReply?.mlx_buffer_cache_sampled_token_count || 0;
    return [
        memoryPageWireLimitFact(statusDocument.mlx_memory_snapshot?.memory_ledger),
        {
            label: "Expert library in memory",
            value: residency?.complete_expert_payload_byte_count > 0
                ? Math.round((residency.resident_expert_payload_bytes / residency.complete_expert_payload_byte_count) * 100) + "%"
                : "—"
        },
        {
            label: "From the SSD per token",
            value: averageStorageReads === null ? "—"
                : Math.round(averageStorageReads) + (routedExpertCount > 0 ? " of " + routedExpertCount + " experts" : " experts")
        },
        {
            label: "Buffers reused",
            value: sampledTokenCount > 0
                ? (sampledTokenCount - (lastReply.mlx_buffer_cache_empty_token_count || 0)) + " of " + sampledTokenCount
                : "—"
        }
    ];
}

// The three facts everyone reads, in plain words: the experts kept in
// memory, what a typical token reads from the SSD, and the limit with a way
// to change it. The engineering counters stay in the developer details.
function memoryPagePlainFacts(statusDocument, expertStatsDocument) {
    const residency = expertStatsDocument?.residency;
    const lastReply = expertStatsDocument?.last_reply;
    const libraryBytes = residency?.complete_expert_payload_byte_count || 0;
    const residentBytes = residency?.resident_expert_payload_bytes || 0;
    const averageStorageReads = memoryPageAverageStorageReads(lastReply);
    const routedExpertCount = lastReply?.routed_expert_count_per_token || 0;
    const limitGigabytes = memoryPageLimitGigabytes(statusDocument);
    return {
        kept: libraryBytes > 0
            // The total is left out here: the lede above prints it to two
            // decimals, and a second, differently rounded copy read as a mismatch.
            ? { value: formatMemoryPageGigabytes(residentBytes),
                detail: Math.round((residentBytes / libraryBytes) * 100) + "% of all its experts, the ones used most" }
            : { value: "—", detail: "Appears for a model that streams experts" },
        read: averageStorageReads !== null && routedExpertCount > 0
            // The share already in memory is said in words, not drawn per
            // token: across a reply it barely moves, so a column for each
            // token showed hundreds of near-identical columns and said less
            // than this one sentence.
            ? { value: Math.round(averageStorageReads) + " of " + routedExpertCount,
                detail: "experts for a typical token; the other " + memoryPageMemoryServedPercent(lastReply) + "% were already in memory" }
            : { value: "—", detail: "Appears after the first reply" },
        limit: { value: limitGigabytes > 0 ? limitGigabytes + " GB" : "—" }
    };
}

function memoryPageAverageStorageReads(lastReply) {
    const storageReadCounts = lastReply?.storage_read_expert_counts_by_token || [];
    if (storageReadCounts.length === 0) return null;
    return storageReadCounts.reduce((total, readCount) => total + readCount, 0) / storageReadCounts.length;
}

// The share of each token's experts already in memory, averaged over the
// last reply, or null before a reply on a model that streams experts.
function memoryPageMemoryServedPercent(lastReply) {
    const averageStorageReads = memoryPageAverageStorageReads(lastReply);
    const routedExpertCount = lastReply?.routed_expert_count_per_token || 0;
    if (averageStorageReads === null || routedExpertCount === 0) return null;
    return Math.round((1 - averageStorageReads / routedExpertCount) * 100);
}

// How the model's experts are split between memory and the SSD, for the
// one bar the shared picture draws, or null for a model that keeps no
// experts apart.
function memoryPageLibrarySplit(expertStatsDocument) {
    const libraryBytes = expertStatsDocument?.residency?.complete_expert_payload_byte_count || 0;
    const residentBytes = Math.min(expertStatsDocument?.residency?.resident_expert_payload_bytes || 0, libraryBytes);
    if (!(libraryBytes > 0)) return null;
    return {
        residentPercent: (residentBytes / libraryBytes) * 100,
        caption: formatMemoryPageGigabytes(residentBytes) + " of its experts in memory · "
            + formatMemoryPageGigabytes(libraryBytes - residentBytes) + " on the SSD"
    };
}

function memoryPagePresentation(statusDocument, expertStatsDocument) {
    const statusDocumentOrEmpty = statusDocument || {};
    return {
        modelLine: statusDocumentOrEmpty.ready_model_id || "No model loaded",
        headline: memoryPageHeadline(statusDocumentOrEmpty) || "Load a model to see how it fits in your memory limit",
        bar: memoryPageBar(statusDocumentOrEmpty),
        facts: memoryPageFacts(statusDocumentOrEmpty, expertStatsDocument),
        plainFacts: memoryPagePlainFacts(statusDocumentOrEmpty, expertStatsDocument)
    };
}

function rememberMemoryPageStatus(statusDocument) {
    memoryPageStatusDocument = statusDocument;
    renderReplySummaries();
}

function rememberMemoryPageExpertStats(expertStatsDocument) {
    memoryPageExpertStatsDocument = expertStatsDocument;
    renderReplySummaries();
}

// Under the hood and Home's "Right now" card read the same two documents, so
// both redraw whenever either document arrives.
function renderReplySummaries() {
    renderMemoryPage();
    if (typeof renderOverviewRightNow === "function") {
        renderOverviewRightNow(memoryPageStatusDocument, memoryPageExpertStatsDocument);
    }
}

function renderMemoryPagePlainFacts(plainFacts) {
    document.getElementById("memory-page-kept-value").textContent = plainFacts.kept.value;
    document.getElementById("memory-page-kept-detail").textContent = plainFacts.kept.detail;
    document.getElementById("memory-page-read-value").textContent = plainFacts.read.value;
    document.getElementById("memory-page-read-detail").textContent = plainFacts.read.detail;
    document.getElementById("memory-page-limit-value").textContent = plainFacts.limit.value;
}

function renderMemoryPageBar(barPresentation) {
    const barSection = document.getElementById("memory-page-bar-section");
    barSection.hidden = !barPresentation;
    if (!barPresentation) return;
    for (const barPart of barPresentation.parts) {
        document.getElementById("memory-page-seg-" + barPart.key).style.width = barPart.widthPercent.toFixed(2) + "%";
        const barKeyLabel = document.getElementById("memory-page-key-" + barPart.key);
        barKeyLabel.textContent = barPart.label + " " + barPart.text;
        // Between replies the conversation and working room are given back;
        // an empty part leaves the key rather than reading "0.0 GB".
        if (barKeyLabel.parentElement) barKeyLabel.parentElement.hidden = barPart.isEmpty;
    }
    const wireLimitMarker = document.getElementById("memory-page-wire-limit");
    wireLimitMarker.hidden = barPresentation.wireLimitPercent === null;
    if (barPresentation.wireLimitPercent !== null) {
        wireLimitMarker.style.left = barPresentation.wireLimitPercent.toFixed(2) + "%";
        document.getElementById("memory-page-wire-limit-label").textContent = barPresentation.wireLimitText;
    }
}

function renderMemoryPage() {
    if (!document.getElementById("memory-page-headline")) return;
    const presentation = memoryPagePresentation(memoryPageStatusDocument, memoryPageExpertStatsDocument);
    document.getElementById("memory-page-model").textContent = presentation.modelLine;
    document.getElementById("memory-page-headline").textContent = presentation.headline;
    renderMemoryPagePlainFacts(presentation.plainFacts);
    renderMemoryPageBar(presentation.bar);
    presentation.facts.forEach((fact, factIndex) => {
        document.getElementById("memory-page-fact-label-" + factIndex).textContent = fact.label;
        document.getElementById("memory-page-fact-value-" + factIndex).textContent = fact.value;
    });
}

function wireMemoryPage() {
    document.getElementById("memory-page-copy")?.addEventListener("click", () => {
        void copyMemoryPageImage("memory-page-copy-feedback");
    });
    renderMemoryPage();
}
