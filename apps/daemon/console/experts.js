// Observatory Experts view. Polls status for the live bars and
// /v1/experts/stats for the session that settles at the end of a generation.
// All model output goes through textContent and createElement only.

const EXPERTS_STATS_URL = "/v1/experts/stats";
const EXPERTS_STATUS_URL = "/v1/status";
const EXPERTS_POLL_INTERVAL_MILLIS = 1000;
const EXPERTS_FEED_ENTRY_LIMIT = 8;

let latestExpertStatusDocument = null;
let latestExpertStatsDocument = null;
let selectedExpertLayerIndex = null;
let renderedExpertLayerSignature = "";
let renderedExpertSlotSignature = "";

// The Observatory shell boots once from console.js; this view starts from
// that single DOMContentLoaded handler like Library and Connect do.
function wireExpertObservatory() {
    pollExpertObservatory();
    setInterval(pollExpertObservatory, EXPERTS_POLL_INTERVAL_MILLIS);
}

async function pollExpertObservatory() {
    await Promise.all([pollExpertStatus(), pollExpertStats()]);
    renderExpertObservatory();
}

async function pollExpertStatus() {
    try {
        const statusResponse = await fetch(EXPERTS_STATUS_URL);
        if (statusResponse.ok) {
            latestExpertStatusDocument = await statusResponse.json();
        }
    } catch (statusFetchError) {
        return;
    }
}

async function pollExpertStats() {
    try {
        const statsResponse = await fetch(EXPERTS_STATS_URL);
        if (statsResponse.ok) {
            latestExpertStatsDocument = await statsResponse.json();
        }
    } catch (statsFetchError) {
        return;
    }
}

function renderExpertObservatory() {
    const statusDocument = latestExpertStatusDocument || {};
    const statsDocument = latestExpertStatsDocument;
    const headlineFacts = expertHeadlineFacts(statusDocument, statsDocument);
    renderExpertPlacement(statusDocument, headlineFacts);
    renderExpertMemoryBar(statusDocument);
    renderExpertLibraryBar(headlineFacts);
    renderExpertSession(statusDocument, statsDocument, headlineFacts);
    renderExpertLayers(statusDocument, statsDocument, headlineFacts);
    renderExpertGeneration(statsDocument);
    renderExpertModeFeed(statsDocument ? statsDocument.mode_transitions || [] : []);
    renderExpertMethod(statsDocument);
}

function renderExpertPlacement(statusDocument, headlineFacts) {
    const placement = expertPlacementCopy(statusDocument);
    setExpertText(
        "experts-headline",
        placement.stockSentence || expertHeadline(headlineFacts)
    );
    const replyElement = document.getElementById("experts-reply");
    if (!replyElement) {
        return;
    }
    replyElement.hidden = placement.replySentence.length === 0;
    replyElement.textContent = placement.replySentence;
}

function expertHeadlineFacts(statusDocument, statsDocument) {
    const residency = expertResidencyFacts(statusDocument, statsDocument);
    const modelId = (statsDocument && statsDocument.model_id)
        || statusDocument.ready_model_id
        || "";
    const expertMemoryMode = (statsDocument && statsDocument.available && statsDocument.expert_memory_mode)
        || statusDocument.expert_memory_mode
        || "";
    return {
        modelId,
        expertMemoryMode,
        completeExpertPayloadByteCount: residency.completeExpertPayloadByteCount,
        residentExpertPayloadBytes: residency.residentExpertPayloadBytes,
        totalLayerCount: residency.totalLayerCount,
        expertCapacity: residency.expertCapacity,
        ceilingBytes: statusDocument.mlx_memory_ceiling_bytes || 0
    };
}

function expertResidencyFacts(statusDocument, statsDocument) {
    const statusResidency = statusDocument.expert_residency || null;
    const statsResidency = statsDocument && statsDocument.residency ? statsDocument.residency : {};
    const library = statsDocument && statsDocument.library ? statsDocument.library : {};
    const completeFromStatus = statusResidency
        ? statusResidency.complete_expert_payload_byte_count || 0
        : 0;
    const completeFromStats = library.complete_expert_payload_byte_count
        || statsResidency.complete_expert_payload_byte_count
        || 0;
    return {
        completeExpertPayloadByteCount: completeFromStatus || completeFromStats,
        residentExpertPayloadBytes: statusResidency
            ? statusResidency.resident_expert_payload_bytes || 0
            : statsResidency.resident_expert_payload_bytes || 0,
        totalLayerCount: statusResidency
            ? statusResidency.total_layer_count || 0
            : statsResidency.total_layer_count || 0,
        expertCapacity: (statusResidency && statusResidency.expert_capacity)
            || library.expert_capacity
            || statsResidency.expert_capacity
            || 0
    };
}

function renderExpertMemoryBar(statusDocument) {
    const ceilingBytes = statusDocument.mlx_memory_ceiling_bytes || 0;
    const segments = reconciledExpertMemorySegments(
        statusDocument.mlx_memory_snapshot,
        ceilingBytes
    );
    setExpertBarWidth("experts-mem-experts", segments.expertBytes, ceilingBytes);
    setExpertBarWidth("experts-mem-model-core", segments.modelCoreBytes, ceilingBytes);
    setExpertBarWidth("experts-mem-context", segments.contextStateBytes, ceilingBytes);
    setExpertBarWidth("experts-mem-runtime", segments.runtimeWorkBytes, ceilingBytes);
    setExpertBarWidth("experts-mem-available", segments.availableBytes, ceilingBytes);
    setExpertText(
        "experts-memory-caption",
        "Experts "
            + formatExpertGigabytes(segments.expertBytes)
            + " · ceiling "
            + formatExpertGigabytes(ceilingBytes)
    );
}

function renderExpertLibraryBar(headlineFacts) {
    const libraryPanel = document.getElementById("experts-library-panel");
    const leftOnSsdBytes = expertPayloadLeftOnSsdBytes(
        headlineFacts.completeExpertPayloadByteCount,
        headlineFacts.residentExpertPayloadBytes
    );
    if (!libraryPanel) {
        return;
    }
    if (leftOnSsdBytes === null) {
        libraryPanel.hidden = true;
        return;
    }
    libraryPanel.hidden = false;
    const completeBytes = headlineFacts.completeExpertPayloadByteCount;
    setExpertBarWidth("experts-lib-ram", headlineFacts.residentExpertPayloadBytes, completeBytes);
    setExpertBarWidth("experts-lib-ssd", leftOnSsdBytes, completeBytes);
    setExpertText(
        "experts-library-caption",
        formatExpertGigabytes(headlineFacts.residentExpertPayloadBytes)
            + " in RAM · "
            + formatExpertGigabytes(leftOnSsdBytes)
            + " on the SSD"
    );
}

function renderExpertSession(statusDocument, statsDocument, headlineFacts) {
    const sessionPanel = document.getElementById("experts-session-panel");
    const emptyNote = document.getElementById("experts-session-empty");
    const sessionGrid = document.getElementById("experts-session-grid");
    const hasSession = Boolean(statsDocument && statsDocument.available);
    if (sessionPanel) {
        sessionPanel.hidden = false;
    }
    if (emptyNote) {
        emptyNote.hidden = hasSession;
    }
    if (sessionGrid) {
        sessionGrid.hidden = !hasSession;
    }
    const openSentence = expertSessionScopeSentence(statusDocument.activity);
    const openNote = document.getElementById("experts-session-open");
    if (openNote) {
        openNote.hidden = false;
        openNote.textContent = openSentence;
    }
    if (!hasSession) {
        return;
    }
    const piles = expertSessionPiles(statsDocument.value);
    const leftOnSsdBytes = expertPayloadLeftOnSsdBytes(
        headlineFacts.completeExpertPayloadByteCount,
        headlineFacts.residentExpertPayloadBytes
    );
    setExpertText("experts-session-prompt", formatExpertGigabytes(piles.promptBytes));
    setExpertText("experts-session-generation", formatExpertGigabytes(piles.generationBytes));
    setExpertText(
        "experts-session-physical",
        piles.physicalDiskReadBytes === null
            ? "Not measured yet"
            : formatExpertGigabytes(piles.physicalDiskReadBytes)
    );
    setExpertText(
        "experts-session-avoided",
        formatExpertGigabytes(piles.avoidedCompleteLayerBytes)
    );
    setExpertText(
        "experts-session-left",
        leftOnSsdBytes === null ? "Not measured" : formatExpertGigabytes(leftOnSsdBytes)
    );
}

function renderExpertLayers(statusDocument, statsDocument, headlineFacts) {
    const layerMap = document.getElementById("experts-layer-map");
    if (!layerMap) {
        return;
    }
    const totalLayerCount = statsDocument && statsDocument.residency
        ? statsDocument.residency.total_layer_count || headlineFacts.totalLayerCount
        : headlineFacts.totalLayerCount;
    const layerRows = expertLayerRowsForObservatory(
        statusDocument.expert_residency,
        statsDocument && statsDocument.layer_topology ? statsDocument.layer_topology : [],
        totalLayerCount,
        headlineFacts.expertCapacity
    );
    if (!findExpertLayerRow(layerRows, selectedExpertLayerIndex) && layerRows.length > 0) {
        selectedExpertLayerIndex = layerRows[0].layerIndex;
    }
    setExpertText("experts-layer-summary", expertLayerOverview(layerRows));
    setExpertText(
        "experts-layer-axis-start",
        layerRows.length > 0 ? "Layer " + layerRows[0].layerIndex : ""
    );
    setExpertText(
        "experts-layer-axis-end",
        layerRows.length > 1 ? "Layer " + layerRows[layerRows.length - 1].layerIndex : ""
    );
    // Polling every second must not rebuild the columns while the pointer
    // is on one, or the click lands on a column that no longer exists.
    const layerSignature = expertLayerListSignature(layerRows);
    if (layerSignature !== renderedExpertLayerSignature) {
        renderedExpertLayerSignature = layerSignature;
        layerMap.textContent = "";
        for (const layerRow of layerRows) {
            layerMap.appendChild(expertLayerColumn(layerRow));
        }
    }
    renderExpertSlots(layerRows);
}

function expertLayerListSignature(layerRows) {
    const signatureParts = [String(selectedExpertLayerIndex)];
    for (const layerRow of layerRows) {
        signatureParts.push(
            layerRow.layerIndex
                + ":"
                + layerRow.retainedCount
                + ":"
                + layerRow.retainedExpertIds.join(",")
                + ":"
                + (layerRow.isStreaming ? "streaming" : "retained")
                + ":"
                + (layerRow.identitiesDeferred ? "deferred" : "named")
        );
    }
    return signatureParts.join("|");
}

function expertLayerColumn(layerRow) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "experts-layer-column";
    button.setAttribute(
        "aria-pressed",
        layerRow.layerIndex === selectedExpertLayerIndex ? "true" : "false"
    );
    button.setAttribute("aria-label", expertLayerDetailSentence(layerRow));
    button.addEventListener("click", function selectExpertLayer() {
        selectedExpertLayerIndex = layerRow.layerIndex;
        renderedExpertSlotSignature = "";
        renderExpertObservatory();
    });
    const track = document.createElement("span");
    track.className = "experts-layer-column-track";
    const fill = document.createElement("span");
    fill.className = "experts-layer-column-fill";
    fill.style.height = expertLayerFillPercent(
        layerRow.retainedCount,
        layerRow.expertCapacity
    ).toFixed(1) + "%";
    track.appendChild(fill);
    button.appendChild(track);
    return button;
}

function renderExpertSlots(layerRows) {
    const slotGrid = document.getElementById("experts-slot-grid");
    const slotCaption = document.getElementById("experts-slot-caption");
    if (!slotGrid) {
        return;
    }
    const selectedRow = findExpertLayerRow(layerRows, selectedExpertLayerIndex);
    if (!selectedRow) {
        slotGrid.textContent = "";
        renderedExpertSlotSignature = "";
        setExpertText("experts-slot-caption", "");
        return;
    }
    const signature = selectedRow.layerIndex
        + ":"
        + selectedRow.retainedCount
        + ":"
        + selectedRow.retainedExpertIds.join(",")
        + ":"
        + (selectedRow.identitiesDeferred ? "deferred" : "named");
    setExpertText("experts-slot-caption", expertLayerCaption(selectedRow));
    if (selectedRow.identitiesDeferred) {
        if (signature !== renderedExpertSlotSignature) {
            renderedExpertSlotSignature = signature;
            slotGrid.textContent = "";
        }
        return;
    }
    if (signature === renderedExpertSlotSignature) {
        return;
    }
    renderedExpertSlotSignature = signature;
    slotGrid.textContent = "";
    const slotStates = expertSlotStates(selectedRow.retainedExpertIds, selectedRow.expertCapacity);
    for (const slotState of slotStates) {
        const slot = document.createElement("span");
        slot.className = "experts-slot";
        slot.dataset.warm = slotState.isWarm ? "true" : "false";
        slot.title = "Expert " + slotState.expertId + (slotState.isWarm ? " in RAM" : " on the SSD");
        slotGrid.appendChild(slot);
    }
    if (slotCaption) {
        slotCaption.hidden = false;
    }
}

function findExpertLayerRow(layerRows, layerIndex) {
    for (const layerRow of layerRows) {
        if (layerRow.layerIndex === layerIndex) {
            return layerRow;
        }
    }
    return null;
}

function renderExpertGeneration(statsDocument) {
    const sentence = statsDocument && statsDocument.available
        ? expertGenerationSentence(statsDocument.value)
        : "Decode layer-token totals appear when the first reply finishes.";
    setExpertText("experts-generation-sentence", sentence);
}

function renderExpertModeFeed(modeTransitions) {
    const modeFeed = document.getElementById("experts-mode-feed");
    if (!modeFeed) {
        return;
    }
    modeFeed.textContent = "";
    const newestFirst = modeTransitions.slice().reverse().slice(0, EXPERTS_FEED_ENTRY_LIMIT);
    if (newestFirst.length === 0) {
        const emptyEntry = document.createElement("li");
        emptyEntry.className = "experts-mode-feed-entry experts-mode-feed-empty";
        emptyEntry.textContent = "No residency changes recorded yet.";
        modeFeed.appendChild(emptyEntry);
        return;
    }
    for (const modeTransition of newestFirst) {
        const entry = document.createElement("li");
        entry.className = "experts-mode-feed-entry";
        entry.textContent = modeTransitionLine(modeTransition);
        modeFeed.appendChild(entry);
    }
}

function renderExpertMethod(statsDocument) {
    const valueSection = statsDocument && statsDocument.available ? statsDocument.value : null;
    setExpertText("experts-method-live", expertMethodLiveLine(valueSection));
    setExpertText(
        "experts-method-mixed",
        valueSection ? expertMixedRouteSentence(valueSection) : ""
    );
}

function setExpertBarWidth(elementId, byteCount, totalBytes) {
    const element = document.getElementById(elementId);
    if (!element) {
        return;
    }
    element.style.width = expertBarPercent(byteCount, totalBytes).toFixed(1) + "%";
}

function setExpertText(elementId, textValue) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = textValue;
    }
}
