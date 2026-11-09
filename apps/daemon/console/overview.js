// Astronomical Observatory Overview: what is running now, memory beside the
// loaded model, and totals since launch. Each part has a pure function that
// decides what to show from the daemon's documents and a small renderer that
// writes it with textContent only, never markup. Reuses neuralEnginePhaseSuffix
// from neural-engine-panel.js.

// The phases /v1/status reports. "target" is prompt processing: the worker names
// the model doing the work rather than the stage. A phase missing from this table
// shows as "Working", because an internal name means nothing on this page.
const OVERVIEW_PHASE_LABELS = new Map([
    ["target", "Reading prompt"],
    ["generation_preparation", "Preparing"],
    ["generation", "Generating"],
    ["preparing", "Preparing"],
    ["encoding_prompt", "Reading prompt"],
    ["denoising", "Drawing"],
    ["decoding", "Finishing image"],
    ["encoding_image", "Saving image"],
]);
const OVERVIEW_IMAGE_PHASES = new Set(["preparing", "encoding_prompt", "denoising", "decoding", "encoding_image"]);

// The model advertised by /v1/models under the loaded id, kept between polls
// because limits come from the model list and the rest from /v1/status.
let overviewLoadedModelListing = null;
// The SSD prompt cache tile joins two polls: whether the cache is on and how
// big it may grow come from /v1/status (the worker reports a zero maximum
// while no model is loaded), and use and hit rate from /v1/cache/stats.
let overviewPromptCacheSetting = null;
let overviewPromptCacheStats = null;

function formatOverviewSeconds(elapsedMilliseconds) {
    return (elapsedMilliseconds / 1000).toFixed(1) + " s";
}

// Pure reconciliation of MLX memory ownership segments against a ceiling, in
// the order the bar draws them. A null snapshot is an all-zero measurement.
function reconciledMlxMemorySegmentBytes(mlxMemorySnapshot, mlxMemoryCeilingBytes) {
    if (!mlxMemorySnapshot) {
        return {
            activeMemoryBytes: 0,
            expertBytes: 0,
            modelCoreBytes: 0,
            contextStateBytes: 0,
            runtimeWorkBytes: 0,
            availableBytes: Math.max(0, mlxMemoryCeilingBytes || 0)
        };
    }
    const activeMemoryBytes = mlxMemorySnapshot.active_memory_bytes || 0;
    const expertBytes = Math.min(mlxMemorySnapshot.expert_payload_bytes || 0, activeMemoryBytes);
    const activeAfterExperts = Math.max(0, activeMemoryBytes - expertBytes);
    const modelCoreBytes = Math.min(mlxMemorySnapshot.model_core_payload_bytes || 0, activeAfterExperts);
    const activeAfterModelCore = Math.max(0, activeAfterExperts - modelCoreBytes);
    const contextStateBytes = Math.min(mlxMemorySnapshot.context_state_payload_bytes || 0, activeAfterModelCore);
    return {
        activeMemoryBytes,
        expertBytes,
        modelCoreBytes,
        contextStateBytes,
        runtimeWorkBytes: Math.max(0, activeAfterModelCore - contextStateBytes),
        availableBytes: Math.max(0, mlxMemoryCeilingBytes - activeMemoryBytes)
    };
}

function memoryPressurePresentationForState(memoryPressureState) {
    switch (memoryPressureState) {
    case "normal":
        return { state: "normal", title: "Normal" };
    case "warning":
        return { state: "warning", title: "Warning" };
    case "critical":
        return { state: "critical", title: "Critical" };
    default:
        return { state: "unavailable", title: "Unavailable" };
    }
}

// What the band at the top says: a state, the model, a progress fraction, one
// line of detail in the work's own units, and when an idle model unloads.
function overviewNowPresentation(statusDocument) {
    if (!statusDocument) {
        return { state: "unavailable", stateLabel: "Unavailable", modelTitle: "The local server is not answering",
            fillPercent: 0, detail: "Observatory retries every second.", unloadText: "" };
    }
    const modelTitle = statusDocument.ready_model_id || "No model loaded";
    const progress = statusDocument.progress;
    if (progress) {
        return overviewWorkingPresentation(statusDocument, progress, modelTitle);
    }
    // A swap keeps the old model ready while the new one loads, so a load in
    // flight counts as loading whatever the status word says.
    const loadInFlight = statusDocument.model_load_in_flight;
    if (statusDocument.status === "loading" || loadInFlight) {
        return { state: "loading", stateLabel: "Loading", modelTitle: loadInFlight?.model_id || modelTitle,
            fillPercent: 0,
            detail: loadInFlight ? "Loading the model into memory." : "Starting the model worker.",
            unloadText: "" };
    }
    if (statusDocument.status === "unavailable") {
        return { state: "unavailable", stateLabel: "Unavailable", modelTitle, fillPercent: 0,
            detail: "The model worker is not running; Astronomical restarts it.", unloadText: "" };
    }
    if (!statusDocument.ready_model_id) {
        return { state: "idle", stateLabel: "Idle", modelTitle, fillPercent: 0,
            detail: "The next request loads the model it asks for.", unloadText: "" };
    }
    return { state: "idle", stateLabel: "Idle", modelTitle, fillPercent: 0,
        detail: "Ready for the next request.", unloadText: overviewUnloadText(statusDocument.idle_model_unload) };
}

// Minutes read better than "Unloads in 1742 s" once the span is long.
function overviewUnloadText(idleModelUnload) {
    if (!idleModelUnload) return "";
    if (!idleModelUnload.after_seconds) return "Stays loaded";
    const remainingSeconds = idleModelUnload.remaining_seconds;
    if (typeof remainingSeconds !== "number") return "";
    if (remainingSeconds === 0) return "Unloading now";
    if (remainingSeconds >= 120) return "Unloads in " + Math.ceil(remainingSeconds / 60) + " min";
    return "Unloads in " + remainingSeconds + " s";
}

function overviewWorkingPresentation(statusDocument, progress, modelTitle) {
    const stateLabel = (OVERVIEW_PHASE_LABELS.get(progress.phase) || "Working")
        + (progress.phase === "target" && typeof neuralEnginePhaseSuffix === "function"
            ? neuralEnginePhaseSuffix(statusDocument) : "");
    const elapsedMilliseconds = progress.elapsed_ms || 0;
    if (OVERVIEW_IMAGE_PHASES.has(progress.phase)) {
        const completedSteps = progress.completed_steps || 0;
        const totalSteps = progress.total_steps || 0;
        return { state: "working", stateLabel, modelTitle,
            fillPercent: totalSteps > 0 ? Math.min(100, (completedSteps / totalSteps) * 100) : 0,
            detail: (totalSteps > 0 ? "Step " + completedSteps + " of " + totalSteps + " · " : "")
                + formatOverviewSeconds(elapsedMilliseconds),
            unloadText: "" };
    }
    if (progress.phase === "generation_preparation") {
        return { state: "working", stateLabel, modelTitle, fillPercent: 0,
            detail: "Getting ready to write · " + formatOverviewSeconds(elapsedMilliseconds), unloadText: "" };
    }
    const processedTokens = progress.processed_tokens || 0;
    const totalTokens = progress.total_tokens || 0;
    const tokensPerSecond = elapsedMilliseconds > 0 ? (processedTokens / elapsedMilliseconds) * 1000 : 0;
    const speedAndTime = " · " + Math.round(tokensPerSecond) + " tok/s · " + formatOverviewSeconds(elapsedMilliseconds);
    if (progress.phase === "generation") {
        // The total is the output ceiling, not a target: most replies end well
        // before it, so it is shown as a limit and the bar stays empty.
        return { state: "working", stateLabel, modelTitle, fillPercent: 0,
            detail: processedTokens.toLocaleString() + " tokens written · up to "
                + totalTokens.toLocaleString() + speedAndTime, unloadText: "" };
    }
    return { state: "working", stateLabel, modelTitle,
        fillPercent: totalTokens > 0 ? Math.min(100, (processedTokens / totalTokens) * 100) : 0,
        detail: processedTokens.toLocaleString() + " of " + totalTokens.toLocaleString() + " prompt tokens read" + speedAndTime,
        unloadText: "" };
}

// Rewrites only text that changed, so assistive technology announcing a status
// element does not repeat identical words every poll.
function setOverviewText(elementId, nextText) {
    const element = document.getElementById(elementId);
    if (element.textContent !== nextText) element.textContent = nextText;
}

function renderOverviewNow(statusDocument) {
    const nowPresentation = overviewNowPresentation(statusDocument);
    setOverviewText("overview-now-state", nowPresentation.stateLabel);
    document.getElementById("overview-now-state").dataset.state = nowPresentation.state;
    setOverviewText("overview-now-model", nowPresentation.modelTitle);
    setOverviewText("overview-now-unload", nowPresentation.unloadText);
    document.getElementById("overview-now-fill").style.width = nowPresentation.fillPercent.toFixed(1) + "%";
    document.getElementById("overview-now-detail").textContent = nowPresentation.detail;
}

// One decimal is enough at a glance; the gigabyte is Apple's 2^30 bytes.
function formatOverviewGigabytes(byteCount) {
    return (byteCount / 1073741824).toFixed(1) + " GB";
}

const OVERVIEW_MEMORY_PARTS = [
    { key: "experts", segmentBytes: "expertBytes" },
    { key: "model-core", segmentBytes: "modelCoreBytes" },
    { key: "context-state", segmentBytes: "contextStateBytes" },
    { key: "runtime-work", segmentBytes: "runtimeWorkBytes" }
];

function renderOverviewMemory(statusDocument) {
    const ceilingBytes = statusDocument.mlx_memory_ceiling_bytes || 0;
    const segmentBytes = reconciledMlxMemorySegmentBytes(statusDocument.mlx_memory_snapshot, ceilingBytes);
    document.getElementById("overview-memory-used").textContent = formatOverviewGigabytes(segmentBytes.activeMemoryBytes);
    document.getElementById("overview-memory-ceiling").textContent = "of " + formatOverviewGigabytes(ceilingBytes) + " ceiling";
    for (const memoryPart of OVERVIEW_MEMORY_PARTS) {
        const partBytes = segmentBytes[memoryPart.segmentBytes];
        const widthPercent = ceilingBytes > 0 ? Math.min(100, (partBytes / ceilingBytes) * 100) : 0;
        document.getElementById("overview-memory-seg-" + memoryPart.key).style.width = widthPercent.toFixed(1) + "%";
        document.getElementById("overview-memory-" + memoryPart.key).textContent = formatOverviewGigabytes(partBytes);
    }
    document.getElementById("overview-memory-free").textContent = formatOverviewGigabytes(segmentBytes.availableBytes);
}

function renderOverviewMemoryPressure(memoryPressureState) {
    const pressurePresentation = memoryPressurePresentationForState(memoryPressureState);
    setOverviewText("overview-memory-pressure", "macOS pressure: " + pressurePresentation.title.toLowerCase());
    document.getElementById("overview-memory-pressure").dataset.pressureState = pressurePresentation.state;
}

// Switches unit by the rounded value, so 999,500 reads "1M", not "1000K".
function formatOverviewContextTokens(tokenCount) {
    if (Math.round(tokenCount / 1_000) >= 1_000) return Math.round(tokenCount / 1_000_000) + "M tokens";
    if (tokenCount >= 1_000) return Math.round(tokenCount / 1_000) + "K tokens";
    return tokenCount + " tokens";
}

function overviewModelKind(modelListing) {
    const supportedEndpoints = modelListing?.supported_endpoints || [];
    if (supportedEndpoints.includes("/v1/images/generations")) return "Image generation";
    if (supportedEndpoints.includes("/v1/embeddings")) return "Embeddings";
    return supportedEndpoints.length > 0 ? "Chat" : null;
}

// Only the facts that apply to the loaded kind of model: an image model has no
// context window, so it gets no "—" row for one. Empty when nothing is loaded.
function overviewModelFacts(statusDocument, modelListing, expertPlacement) {
    if (!statusDocument || !statusDocument.ready_model_id) return [];
    const modelFacts = [];
    const modelKind = overviewModelKind(modelListing);
    if (modelKind) modelFacts.push({ label: "Kind", value: modelKind });
    if (modelListing?.context_window) {
        modelFacts.push({ label: "Context", value: formatOverviewContextTokens(modelListing.context_window) });
    }
    if (modelKind === "Chat") {
        const readsImages = (modelListing.input_modalities || []).includes("image");
        modelFacts.push({ label: "Reads", value: readsImages ? "Text and images" : "Text" });
    }
    if (typeof statusDocument.ready_model_size_bytes === "number") {
        modelFacts.push({ label: "On disk", value: formatOverviewGigabytes(statusDocument.ready_model_size_bytes) });
    }
    if (expertPlacement?.stockSentence) modelFacts.push({ label: "In memory", value: expertPlacement.stockSentence });
    if (expertPlacement?.replySentence) modelFacts.push({ label: "Experts", value: expertPlacement.replySentence });
    return modelFacts;
}

function rememberOverviewLoadedModelListing(modelListing) {
    overviewLoadedModelListing = modelListing;
}

function renderOverviewModel(statusDocument, expertPlacement) {
    const listingForLoadedModel = overviewLoadedModelListing
        && overviewLoadedModelListing.id === statusDocument?.ready_model_id ? overviewLoadedModelListing : null;
    const modelFacts = overviewModelFacts(statusDocument, listingForLoadedModel, expertPlacement);
    const factList = document.getElementById("overview-model-facts");
    const factRows = [];
    for (const modelFact of modelFacts) {
        const factLabel = document.createElement("dt");
        factLabel.textContent = modelFact.label;
        const factValue = document.createElement("dd");
        factValue.textContent = modelFact.value;
        factRows.push(factLabel, factValue);
    }
    factList.replaceChildren(...factRows);
    factList.hidden = modelFacts.length === 0;
    document.getElementById("overview-model-empty").hidden = modelFacts.length > 0;
}

// The two reuse figures measure different things, so each says which: prompt
// tokens the loaded model kept in memory, and prompts found in the SSD cache.
function overviewSessionPresentation(servingSession) {
    const completedRequestCount = servingSession?.completed_request_count || 0;
    const averageTokensPerSecond = servingSession?.average_generation_tok_per_second || 0;
    const totalPromptTokens = servingSession?.total_prompt_token_count || 0;
    const reusedPromptTokens = Math.min(servingSession?.total_reused_prompt_token_count || 0, totalPromptTokens);
    return {
        requests: completedRequestCount.toLocaleString(),
        speed: averageTokensPerSecond > 0 ? Math.round(averageTokensPerSecond) + " tok/s" : "—",
        memoryReuse: totalPromptTokens > 0 ? Math.round((reusedPromptTokens / totalPromptTokens) * 100) + "%" : "—",
        memoryReuseHint: totalPromptTokens > 0
            ? reusedPromptTokens.toLocaleString() + " of " + totalPromptTokens.toLocaleString() + " prompt tokens"
            : "No prompts yet"
    };
}

function renderOverviewSession(statusDocument) {
    overviewPromptCacheSetting = statusDocument.configuration?.prompt_cache || null;
    renderOverviewPromptCacheTile();
    const sessionPresentation = overviewSessionPresentation(statusDocument.serving_session);
    document.getElementById("overview-session-requests").textContent = sessionPresentation.requests;
    document.getElementById("overview-session-speed").textContent = sessionPresentation.speed;
    document.getElementById("overview-session-memory-reuse").textContent = sessionPresentation.memoryReuse;
    document.getElementById("overview-session-memory-reuse-hint").textContent = sessionPresentation.memoryReuseHint;
}

function overviewPromptCachePresentation(promptCacheSetting, cacheStatsDocument) {
    if (!promptCacheSetting) return { usage: "—", hint: "" };
    // Effective values stay null until the worker acknowledges its settings;
    // the configured value, then the default, stands in until then.
    const enabledSetting = promptCacheSetting.enabled || {};
    const isEnabled = enabledSetting.effective ?? enabledSetting.configured ?? enabledSetting.default;
    if (isEnabled === false) {
        return { usage: "Off", hint: "Turned off in Configuration" };
    }
    const capacitySetting = promptCacheSetting.capacity_bytes || {};
    const capacityBytes = capacitySetting.effective ?? capacitySetting.configured ?? capacitySetting.default ?? 0;
    const usedBytes = cacheStatsDocument?.persistent_prompt_cache_total_size_bytes || 0;
    const promptLookups = (cacheStatsDocument?.persistent_prompt_cache_hits || 0)
        + (cacheStatsDocument?.persistent_prompt_cache_misses || 0);
    return {
        usage: formatOverviewGigabytes(usedBytes) + " of " + formatOverviewGigabytes(capacityBytes),
        hint: promptLookups > 0
            ? Math.round((cacheStatsDocument.persistent_prompt_cache_hit_rate || 0) * 100) + "% of prompts found there"
            : "No prompts looked up yet"
    };
}

function renderOverviewPromptCacheTile() {
    const promptCachePresentation = overviewPromptCachePresentation(overviewPromptCacheSetting, overviewPromptCacheStats);
    document.getElementById("overview-session-ssd-cache").textContent = promptCachePresentation.usage;
    document.getElementById("overview-session-ssd-cache-hint").textContent = promptCachePresentation.hint;
}

function renderOverviewPromptCache(cacheStatsDocument) {
    overviewPromptCacheStats = cacheStatsDocument;
    renderOverviewPromptCacheTile();
}
