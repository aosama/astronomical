// The Memory page: the whole Mac's memory on one bar with the kernel's wire
// limit drawn on it, four facts about the reply, and one column per token.
// The shareable picture is drawn by memory-page-image.js from the same
// presentation. Text is set through textContent only; never innerHTML.

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

// The claim at the top, built only from what is known: a part with no
// figure behind it is left out, never shown as zero.
function memoryPageHeadline(statusDocument) {
    const headlineParts = [];
    if (statusDocument.ready_model_size_bytes > 0) {
        headlineParts.push(formatMemoryPageGigabytes(statusDocument.ready_model_size_bytes) + " model");
    }
    if (statusDocument.machine_mlx_memory_ceiling_bytes > 0) {
        headlineParts.push(
            Math.round(statusDocument.machine_mlx_memory_ceiling_bytes / MEMORY_PAGE_BYTES_PER_GIGABYTE) + " GB Mac");
    }
    const tokensPerSecond = statusDocument.serving_session?.average_generation_tok_per_second || 0;
    if (tokensPerSecond > 0) {
        headlineParts.push(Math.round(tokensPerSecond) + " tokens per second");
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

function memoryPageAverageStorageReads(lastReply) {
    const storageReadCounts = lastReply?.storage_read_expert_counts_by_token || [];
    if (storageReadCounts.length === 0) return null;
    return storageReadCounts.reduce((total, readCount) => total + readCount, 0) / storageReadCounts.length;
}

// One column per token. A column stands for every expert the token routed
// to, and its violet part is the share read from the SSD: drawn to that
// scale, a token that read 22 of its 384 experts looks as it was, almost
// wholly served from memory.
function memoryPageTokens(lastReply) {
    const storageReadCounts = lastReply?.storage_read_expert_counts_by_token || [];
    const largestReadCount = storageReadCounts.reduce((largest, readCount) => Math.max(largest, readCount), 0);
    const routedExpertCount = Math.max(lastReply?.routed_expert_count_per_token || 0, largestReadCount, 1);
    return storageReadCounts.map((readCount) => ({
        readCount,
        routedExpertCount,
        storageSharePercent: (readCount / routedExpertCount) * 100
    }));
}

function memoryPageTokenNote(lastReply) {
    const averageStorageReads = memoryPageAverageStorageReads(lastReply);
    const routedExpertCount = lastReply?.routed_expert_count_per_token || 0;
    if (averageStorageReads === null || routedExpertCount === 0) {
        return "Finish a reply on a model that streams experts to see every token here.";
    }
    const memoryServedPercent = Math.round((1 - averageStorageReads / routedExpertCount) * 100);
    return "Each token used " + routedExpertCount + " experts. On average " + memoryServedPercent
        + "% of them were already in memory; the rest were read from the SSD.";
}

function memoryPagePresentation(statusDocument, expertStatsDocument) {
    const statusDocumentOrEmpty = statusDocument || {};
    const lastReply = expertStatsDocument?.last_reply;
    return {
        modelLine: statusDocumentOrEmpty.ready_model_id || "No model loaded",
        headline: memoryPageHeadline(statusDocumentOrEmpty) || "Load a model to see how it fits in this Mac's memory",
        bar: memoryPageBar(statusDocumentOrEmpty),
        facts: memoryPageFacts(statusDocumentOrEmpty, expertStatsDocument),
        tokens: memoryPageTokens(lastReply),
        tokenNote: memoryPageTokenNote(lastReply)
    };
}

function rememberMemoryPageStatus(statusDocument) {
    memoryPageStatusDocument = statusDocument;
    renderMemoryPage();
}

function rememberMemoryPageExpertStats(expertStatsDocument) {
    memoryPageExpertStatsDocument = expertStatsDocument;
    renderMemoryPage();
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

function renderMemoryPageTokens(tokenPresentations) {
    const tokenStrip = document.getElementById("memory-page-token-strip");
    const stripSignature = tokenPresentations.map((token) => token.readCount).join(",");
    if (tokenStrip.dataset.signature === stripSignature) return;
    tokenStrip.dataset.signature = stripSignature;
    tokenStrip.replaceChildren(...tokenPresentations.map((token, tokenIndex) => {
        const tokenCell = document.createElement("span");
        tokenCell.className = "memory-page-token";
        tokenCell.title = "Token " + (tokenIndex + 1) + " · " + (token.routedExpertCount - token.readCount)
            + " experts from memory, " + token.readCount + " from the SSD";
        const storageShare = document.createElement("span");
        storageShare.className = "memory-page-token-ssd";
        storageShare.style.height = token.storageSharePercent.toFixed(2) + "%";
        tokenCell.append(storageShare);
        return tokenCell;
    }));
}

function renderMemoryPage() {
    if (!document.getElementById("memory-page-headline")) return;
    const presentation = memoryPagePresentation(memoryPageStatusDocument, memoryPageExpertStatsDocument);
    document.getElementById("memory-page-model").textContent = presentation.modelLine;
    document.getElementById("memory-page-headline").textContent = presentation.headline;
    renderMemoryPageBar(presentation.bar);
    presentation.facts.forEach((fact, factIndex) => {
        document.getElementById("memory-page-fact-label-" + factIndex).textContent = fact.label;
        document.getElementById("memory-page-fact-value-" + factIndex).textContent = fact.value;
    });
    renderMemoryPageTokens(presentation.tokens);
    document.getElementById("memory-page-token-note").textContent = presentation.tokenNote;
}

function wireMemoryPage() {
    document.getElementById("memory-page-copy")?.addEventListener("click", () => { void copyMemoryPageImage(); });
    renderMemoryPage();
}
