// Home's "Right now" card: one plain sentence everyone sees first, naming the
// model's size, the memory limit and the speed, a second sentence on how much
// of each word's work was already in memory, and one bar splitting the model
// between memory and the SSD. It reads the same two documents as Under the
// hood (memory-page.js holds them and asks this card to redraw), so the two
// pages never disagree. Text is set through textContent only; never innerHTML.

// The sentence and bar for the card, or null while no model is loaded or the
// limit is not reported: the card then stays hidden rather than guessing.
function overviewRightNowPresentation(statusDocument, expertStatsDocument) {
    const modelSizeBytes = statusDocument?.ready_model_size_bytes || 0;
    const limitGigabytes = memoryPageLimitGigabytes(statusDocument || {});
    if (!statusDocument?.ready_model_id || !(modelSizeBytes > 0) || !(limitGigabytes > 0)) return null;
    return {
        modelLine: statusDocument.ready_model_id,
        headline: overviewRightNowHeadline(statusDocument, formatMemoryPageGigabytes(modelSizeBytes), limitGigabytes),
        detail: overviewRightNowDetail(statusDocument, expertStatsDocument, limitGigabytes),
        split: overviewRightNowSplit(statusDocument, expertStatsDocument)
    };
}

// The speed shown and whether a reply is being written now: a reply in
// progress reports its own pace, otherwise the session's average does.
function overviewRightNowSpeed(statusDocument) {
    const progress = statusDocument.progress;
    const isWriting = Boolean(statusDocument.activity) && statusDocument.activity !== "idle"
        && progress?.phase === "generation" && (progress.elapsed_ms || 0) > 0;
    if (isWriting) {
        return { isWriting: true, tokensPerSecond: ((progress.processed_tokens || 0) / progress.elapsed_ms) * 1000 };
    }
    return { isWriting: false, tokensPerSecond: statusDocument.serving_session?.average_generation_tok_per_second || 0 };
}

function overviewRightNowHeadline(statusDocument, modelSizeText, limitGigabytes) {
    const speed = overviewRightNowSpeed(statusDocument);
    const subject = "This " + modelSizeText + " model ";
    const limitPhrase = "inside your " + limitGigabytes + " GB limit";
    if (speed.tokensPerSecond > 0) {
        const verb = speed.isWriting ? "is answering " : "last answered ";
        return subject + verb + limitPhrase + ", at " + Math.round(speed.tokensPerSecond) + " tokens a second.";
    }
    return subject + "is loaded " + limitPhrase + ".";
}

function overviewRightNowDetail(statusDocument, expertStatsDocument, limitGigabytes) {
    const memoryServedPercent = memoryPageMemoryServedPercent(expertStatsDocument?.last_reply);
    if (memoryServedPercent !== null) {
        return memoryServedPercent + "% of what each word needs is already in memory. "
            + "The rest is read from your SSD as it goes.";
    }
    if (statusDocument.ready_model_size_bytes <= limitGigabytes * MEMORY_PAGE_BYTES_PER_GIGABYTE) {
        return "The whole model fits in your limit, so nothing has to be read from your SSD.";
    }
    return "Astronomical keeps what each word needs most in memory and reads the rest from your SSD.";
}

// How much of the model sits in memory and how much stays on the SSD, from
// the weights every step uses plus the experts kept. Null when the expert
// figures are missing, so the bar never shows a split it cannot back.
function overviewRightNowSplit(statusDocument, expertStatsDocument) {
    const modelSizeBytes = statusDocument.ready_model_size_bytes;
    const coreWeightBytes = statusDocument.mlx_memory_snapshot?.memory_ledger?.core_weight_bytes || 0;
    const residentExpertBytes = expertStatsDocument?.residency?.resident_expert_payload_bytes;
    if (!(coreWeightBytes > 0) || typeof residentExpertBytes !== "number") return null;
    const inMemoryBytes = Math.min(modelSizeBytes, coreWeightBytes + residentExpertBytes);
    const onSsdBytes = modelSizeBytes - inMemoryBytes;
    return {
        inMemoryPercent: (inMemoryBytes / modelSizeBytes) * 100,
        inMemoryText: "In memory · " + formatMemoryPageGigabytes(inMemoryBytes),
        onSsdText: "On your SSD, read as needed · " + formatMemoryPageGigabytes(onSsdBytes)
    };
}

function renderOverviewRightNowSplit(splitPresentation) {
    document.getElementById("overview-right-now-split").hidden = !splitPresentation;
    if (!splitPresentation) return;
    document.getElementById("overview-right-now-memory").style.width = splitPresentation.inMemoryPercent.toFixed(2) + "%";
    document.getElementById("overview-right-now-memory-text").textContent = splitPresentation.inMemoryText;
    document.getElementById("overview-right-now-ssd-text").textContent = splitPresentation.onSsdText;
}

function renderOverviewRightNow(statusDocument, expertStatsDocument) {
    const card = document.getElementById("overview-right-now");
    if (!card) return;
    const presentation = overviewRightNowPresentation(statusDocument, expertStatsDocument);
    card.hidden = !presentation;
    if (!presentation) return;
    document.getElementById("overview-right-now-model").textContent = presentation.modelLine;
    document.getElementById("overview-right-now-headline").textContent = presentation.headline;
    document.getElementById("overview-right-now-detail").textContent = presentation.detail;
    renderOverviewRightNowSplit(presentation.split);
}

function wireOverviewRightNow() {
    document.getElementById("overview-right-now-copy")?.addEventListener("click", () => {
        void copyMemoryPageImage("overview-right-now-copy-feedback");
    });
}
