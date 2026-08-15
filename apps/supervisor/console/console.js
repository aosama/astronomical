// Astronomical Observatory telemetry and controls.
// All model output is rendered via textContent only; never innerHTML.

const STATUS_URL = "/v1/status";
const MODELS_URL = "/v1/models";
const CACHE_STATS_URL = "/v1/cache/stats";
const SYSTEM_TELEMETRY_URL = "/v1/system/telemetry";
const CONFIG_RELOAD_URL = "/v1/config/reload";
const SERVER_SHUTDOWN_URL = "/v1/control/shutdown";
const POLL_INTERVAL_MILLIS = 1000;
const SPARKLINE_BUFFER_SIZE = 60;

const OBSERVATORY_PATH_MAP = {
    overview: "/overview",
    chat: "/chat",
    optimizer: "/optimizer",
    model: "/model",
    settings: "/settings"
};

const OBSERVATORY_PATH_TO_DESTINATION_MAP = Object.fromEntries(
    Object.entries(OBSERVATORY_PATH_MAP).map(([destination, path]) => [path, destination])
);

const OBSERVATORY_DEFAULT_DESTINATION = "overview";
const OBSERVATORY_DEFAULT_PATH = "/overview";

let selectedModelId = null;
let selectedModelMaximumInputTokens = null;
const sparklineHitRateBuffer = [];

document.addEventListener("DOMContentLoaded", () => {
    console.log("observatory boot");
    wireObservatoryNavigation();
    pollStatus();
    pollCacheStats();
    pollModels();
    setInterval(pollStatus, POLL_INTERVAL_MILLIS);
    setInterval(pollCacheStats, POLL_INTERVAL_MILLIS);
    setInterval(pollModels, POLL_INTERVAL_MILLIS);
    wirePlayground();
    wireServerControls();
    wireMemoryLimitControl();
});

function wireObservatoryNavigation() {
    const navigationButtons = Array.from(
        document.querySelectorAll("[data-observatory-destination]")
    );
    const observatoryViews = Array.from(document.querySelectorAll("[data-observatory-view]"));
    const activateCurrentObservatoryPath = () => {
        const currentObservatoryPath = window.location.pathname;
        const matchedDestination = OBSERVATORY_PATH_TO_DESTINATION_MAP[currentObservatoryPath]
            || OBSERVATORY_DEFAULT_DESTINATION;
        activateObservatoryView(matchedDestination, navigationButtons, observatoryViews, false);
        if (!OBSERVATORY_PATH_TO_DESTINATION_MAP[currentObservatoryPath]) {
            history.replaceState(null, "", OBSERVATORY_DEFAULT_PATH);
        }
    };
    navigationButtons.forEach((navigationButton) => {
        navigationButton.addEventListener("click", () => {
            activateObservatoryView(
                navigationButton.dataset.observatoryDestination,
                navigationButtons,
                observatoryViews,
                /* updateHistory */ true
            );
        });
    });
    window.addEventListener("popstate", activateCurrentObservatoryPath);
    activateCurrentObservatoryPath();
}

function activateObservatoryView(requestedViewIdentifier, navigationButtons, observatoryViews, updateHistory) {
    const requestedViewExists = observatoryViews.some(
        (observatoryView) => observatoryView.dataset.observatoryView === requestedViewIdentifier
    );
    const activeViewIdentifier = requestedViewExists ? requestedViewIdentifier : "overview";
    observatoryViews.forEach((observatoryView) => {
        observatoryView.hidden = observatoryView.dataset.observatoryView !== activeViewIdentifier;
    });
    navigationButtons.forEach((navigationButton) => {
        if (navigationButton.dataset.observatoryDestination === activeViewIdentifier) {
            navigationButton.setAttribute("aria-current", "page");
        } else {
            navigationButton.removeAttribute("aria-current");
        }
    });
    const targetPath = OBSERVATORY_PATH_MAP[activeViewIdentifier] || OBSERVATORY_DEFAULT_PATH;
    if (updateHistory) {
        history.pushState(null, "", targetPath);
    }
    return activeViewIdentifier;
}

async function pollStatus() {
    void pollSystemTelemetry();
    try {
        const response = await fetch(STATUS_URL);
        if (!response.ok) { setStatusUnavailable(); return; }
        const data = await response.json();
        renderApplicationIdentity(data.application);
        renderStatusHeader(data);
        renderNowStrip(data);
        renderAboutFromStatus(data);
        renderCompactMlxMemory(data);
        renderMemoryLimitControl(data);
        renderSession(data);
        renderAboutEnhanced(data);
        renderPromptProcessingOptimizer(
            data.prompt_processing_chunk_size_optimizer,
            selectedModelMaximumInputTokens
        );
    } catch (fetchError) {
        setStatusUnavailable();
    }
}

function applicationIdentityTitle(applicationIdentity) {
    if (!applicationIdentity) { return "Version unavailable"; }
    const dirtySuffix = applicationIdentity.is_dirty ? "-dirty" : "";
    const channelTitle = applicationIdentity.channel_display_name || applicationIdentity.channel || "Unknown";
    return `${applicationIdentity.version || "unknown"} · ${channelTitle} · ${applicationIdentity.commit || "unknown"}${dirtySuffix}`;
}

function renderApplicationIdentity(applicationIdentity) {
    const identityTitle = applicationIdentityTitle(applicationIdentity);
    document.getElementById("application-build-identity").textContent = identityTitle;
    document.getElementById("footer-application-identity").textContent = identityTitle;
    if (!applicationIdentity) { return; }
    document.getElementById("application-channel").textContent =
        applicationIdentity.channel_display_name || applicationIdentity.channel;
    document.getElementById("about-instance-note").textContent =
        `Single-user local console. Served by astronomicald on this machine. Config: ${applicationIdentity.state_directory}/config.json`;
    document.body.dataset.applicationChannel = applicationIdentity.channel || "unknown";
}

async function pollSystemTelemetry() {
    try {
        const response = await fetch(SYSTEM_TELEMETRY_URL);
        if (!response.ok) {
            renderCompactMemoryPressure(null);
            return;
        }
        const telemetryDocument = await response.json();
        renderCompactMemoryPressure(telemetryDocument.memory_pressure);
    } catch (fetchError) {
        renderCompactMemoryPressure(null);
    }
}

function setStatusUnavailable() {
    const header = document.getElementById("status-header");
    header.className = "status-header unavailable";
    document.getElementById("status-word").textContent = "Unavailable";
    document.getElementById("status-model-id").textContent = "No model loaded";
    document.getElementById("optimizer-configuration-note").hidden = true;
    resetNowStrip();
}

function renderStatusHeader(data) {
    const header = document.getElementById("status-header");
    const statusWord = document.getElementById("status-word");
    const modelIdLabel = document.getElementById("status-model-id");
    const optimizerConfigurationNote = document.getElementById("optimizer-configuration-note");
    const status = data.status || "unavailable";
    const activity = data.activity || "idle";
    const isActive = activity === "prompt_processing" || activity === "generating";
    let className = "status-header " + status;
    if (isActive && status === "ready") { className += " active"; }
    header.className = className;
    if (status === "loading") {
        statusWord.textContent = "Loading model…";
    } else if (status === "ready") {
        statusWord.textContent = isActive ? describeActivity(activity) : "Ready";
    } else {
        statusWord.textContent = "Unavailable";
    }
    const readyModelId = data.ready_model_id || null;
    if (readyModelId) {
        modelIdLabel.textContent = readyModelId;
        selectedModelId = readyModelId;
    } else if (status !== "ready") {
        modelIdLabel.textContent = "No model loaded";
    }
    if (data.config_warning) {
        optimizerConfigurationNote.textContent = data.config_warning;
        optimizerConfigurationNote.hidden = false;
    } else {
        optimizerConfigurationNote.hidden = true;
    }
}

function describeActivity(activity) {
    if (activity === "prompt_processing") { return "Processing prompt"; }
    if (activity === "generating") { return "Generating"; }
    return "Active";
}

async function pollModels() {
    try {
        const response = await fetch(MODELS_URL);
        if (!response.ok) { return; }
        const data = await response.json();
        const models = data.data || [];
        const selectedModel = selectAdvertisedModel(models, selectedModelId);
        if (selectedModel) {
            selectedModelId = selectedModel.id;
            selectedModelMaximumInputTokens = Number(selectedModel.max_input_tokens || 0) || null;
            renderAboutFromModels(selectedModel);
        }
    } catch (fetchError) {
        // Status polling handles the unavailable display; stay quiet here.
    }
}

function selectAdvertisedModel(advertisedModels, preferredModelId) {
    return advertisedModels.find((model) => model.id === preferredModelId)
        || advertisedModels[0]
        || null;
}

function renderNowStrip(data) {
    const phaseLabel = document.getElementById("phase-label");
    const phaseFill = document.getElementById("phase-fill");
    const phaseTokens = document.getElementById("phase-tokens");
    const phaseTps = document.getElementById("phase-tps");
    const phaseElapsed = document.getElementById("phase-elapsed");
    const progress = data.progress;
    if (!progress) { resetNowStrip(); return; }
    const processedTokens = progress.processed_tokens || 0;
    const totalTokens = progress.total_tokens || 0;
    const elapsedMs = progress.elapsed_ms || 0;
    const fillPercent = totalTokens > 0 ? Math.min(100, (processedTokens / totalTokens) * 100) : 0;
    const tokPerSecond = elapsedMs > 0 ? (processedTokens / elapsedMs) * 1000 : 0;
    if (progress.phase === "prefill") {
        phaseLabel.textContent = "Prefilling";
        phaseFill.classList.remove("generating");
    } else if (progress.phase === "generation") {
        phaseLabel.textContent = "Generating";
        phaseFill.classList.add("generating");
    } else {
        phaseLabel.textContent = progress.phase || "Active";
    }
    phaseFill.style.width = fillPercent.toFixed(1) + "%";
    phaseTokens.textContent = processedTokens.toLocaleString() + " / " + totalTokens.toLocaleString();
    phaseTps.textContent = tokPerSecond.toFixed(0) + " tok/s";
    phaseElapsed.textContent = elapsedMs.toLocaleString() + " ms";
}

function resetNowStrip() {
    document.getElementById("phase-label").textContent = "Idle";
    document.getElementById("phase-fill").style.width = "0%";
    document.getElementById("phase-fill").classList.remove("generating");
    document.getElementById("phase-tokens").textContent = "—";
    document.getElementById("phase-tps").textContent = "— tok/s";
    document.getElementById("phase-elapsed").textContent = "— ms";
}

async function pollCacheStats() {
    try {
        const response = await fetch(CACHE_STATS_URL);
        if (!response.ok) { return; }
        const data = await response.json();
        renderCompactCachePanel(data);
        const cacheEfficacyDocument = data.speculative_prefill_cache_efficacy || {};
        const hitRate = (cacheEfficacyDocument.combined || {}).reuse_rate || 0;
        pushSparklineSample(hitRate);
        renderSparkline();
    } catch (fetchError) {
        // Cache panel stays at last-known values on transient fetch errors.
    }
}

function formatGigabytes(bytes) {
    return (bytes / 1e9).toFixed(2) + " GB";
}

function pushSparklineSample(hitRate) {
    sparklineHitRateBuffer.push(hitRate);
    if (sparklineHitRateBuffer.length > SPARKLINE_BUFFER_SIZE) {
        sparklineHitRateBuffer.shift();
    }
}

function renderSparkline() {
    const compactLine = document.getElementById("compact-cache-sparkline-line");
    if (sparklineHitRateBuffer.length < 2) {
        compactLine.setAttribute("points", "");
        return;
    }
    const width = 120;
    const height = 24;
    const stepX = width / (SPARKLINE_BUFFER_SIZE - 1);
    const points = sparklineHitRateBuffer.map((value, index) => {
        const x = (index * stepX).toFixed(1);
        const y = (height - value * height).toFixed(1);
        return x + "," + y;
    }).join(" ");
    compactLine.setAttribute("points", points);
}

function renderAboutFromStatus(data) {
    if (data.ready_model_id) {
        document.getElementById("about-model-id").textContent = data.ready_model_id;
    }
    const expertMemory = data.expert_memory_mode;
    const expertMemoryElement = document.getElementById("about-expert-memory");
    if (expertMemory) {
        expertMemoryElement.textContent =
            expertMemory.charAt(0).toUpperCase() + expertMemory.slice(1);
    } else { expertMemoryElement.textContent = "—"; }
    const mtpEnabled = data.mtp_enabled;
    const mtpRuntimeState = data.mtp_runtime_state;
    const mtpElement = document.getElementById("about-mtp");
    if (data.ready_model_id && mtpEnabled !== undefined && mtpRuntimeState !== undefined) {
        mtpElement.textContent = mtpEnabled ? String(mtpRuntimeState) : "disabled (config)";
    } else if (!data.ready_model_id) {
        mtpElement.textContent = "Not loaded";
    }
}

function renderAboutFromModels(model) {
    document.getElementById("about-context-window").textContent =
        (model.context_window || 0).toLocaleString();
    document.getElementById("about-max-input").textContent =
        (model.max_input_tokens || 0).toLocaleString();
    document.getElementById("about-max-output").textContent =
        (model.max_output_tokens || 0).toLocaleString();
    const inputModalities = model.input_modalities || [];
    const hasImage = inputModalities.indexOf("image") !== -1;
    document.getElementById("about-modalities").textContent =
        hasImage ? "Text + Image" : "Text only";
    const maximumOutputTokens = model.max_output_tokens || 1;
    const maximumOutputSlider = document.getElementById("sampling-max-tokens");
    maximumOutputSlider.max = String(maximumOutputTokens);
    if (parseInt(maximumOutputSlider.value, 10) > maximumOutputTokens) {
        maximumOutputSlider.value = String(maximumOutputTokens);
        document.getElementById("sampling-max-tokens-value").textContent =
            maximumOutputTokens.toLocaleString();
    }
}

function renderAboutEnhanced(data) {
    const readyModelSizeBytes = data.ready_model_size_bytes;
    if (readyModelSizeBytes !== undefined && readyModelSizeBytes !== null) {
        document.getElementById("about-disk-size").textContent = formatGigabytes(readyModelSizeBytes);
    } else if (data.ready_model_id) {
        document.getElementById("about-disk-size").textContent = "Not measured";
    }
    const expertMemoryMode = data.expert_memory_mode;
    const residencyElement = document.getElementById("about-residency");
    if (expertMemoryMode) {
        residencyElement.textContent =
            expertMemoryMode === "paged" ? "RAM + SSD streaming"
            : expertMemoryMode === "resident" ? "Fully in memory"
            : "Not loaded";
    } else if (data.status !== "ready") {
        residencyElement.textContent = "Not loaded";
    }
    const mtpUnavailableReason = data.mtp_unavailable_reason;
    const mtpReasonElement = document.getElementById("about-mtp-reason");
    if (mtpUnavailableReason) {
        mtpReasonElement.textContent = mtpUnavailableReason;
    } else {
        mtpReasonElement.textContent = "—";
    }
}

function setMlxSegmentWidth(elementId, byteCount, ceilingBytes) {
    const element = document.getElementById(elementId);
    if (!element) { return; }
    const fraction = ceilingBytes > 0 ? Math.min(100, (byteCount / ceilingBytes) * 100) : 0;
    element.style.width = fraction.toFixed(1) + "%";
}

function renderSession(data) {
    const servingSession = data.serving_session;
    if (!servingSession) { return; }
    const completedRequestCount = servingSession.completed_request_count || 0;
    document.getElementById("session-requests").textContent =
        completedRequestCount + (completedRequestCount === 1 ? " request" : " requests");
    const averageGenerationTokensPerSecond = servingSession.average_generation_tok_per_second || 0;
    document.getElementById("session-avg-gen").textContent =
        Math.round(averageGenerationTokensPerSecond) + " tok/s";
    const totalPromptTokenCount = servingSession.total_prompt_token_count || 0;
    const totalReusedPromptTokenCount = servingSession.total_reused_prompt_token_count || 0;
    const reusePercentageElement = document.getElementById("session-reuse-pct");
    const reuseFill = document.getElementById("reuse-fill");
    const reuseBreakdownElement = document.getElementById("session-reuse-breakdown");
    if (totalPromptTokenCount === 0) {
        reusePercentageElement.textContent = "Not measured";
        reuseFill.style.width = "0%";
        reuseBreakdownElement.textContent = "No completed prompts";
        return;
    }
    const boundedReusedCount = Math.min(totalReusedPromptTokenCount, totalPromptTokenCount);
    const reuseFraction = boundedReusedCount / totalPromptTokenCount;
    const reusePercentage = Math.round(reuseFraction * 100);
    reusePercentageElement.textContent = reusePercentage + "%";
    reuseFill.style.width = (reuseFraction * 100) + "%";
    const newTokenCount = totalPromptTokenCount - boundedReusedCount;
    reuseBreakdownElement.textContent =
        boundedReusedCount.toLocaleString() + " reused · " + newTokenCount.toLocaleString() + " new";
}

function wireServerControls() {
    const reloadButton = document.getElementById("control-reload");
    const stopButton = document.getElementById("control-stop");
    reloadButton.addEventListener("click", reloadConfig);
    stopButton.addEventListener("click", stopServer);
}

async function reloadConfig() {
    const reloadButton = document.getElementById("control-reload");
    reloadButton.disabled = true;
    showControlFeedback("Reloading configuration…", "progress");
    try {
        const response = await fetch(CONFIG_RELOAD_URL, { method: "POST" });
        const responseBody = await response.text();
        let parsedResponse;
        try { parsedResponse = JSON.parse(responseBody); } catch (parseError) { parsedResponse = null; }
        if (response.ok) {
            if (parsedResponse && parsedResponse.status === "worker_restart_started") {
                showControlFeedback(parsedResponse.message || "Config reloaded; worker restarting…", "progress");
            } else if (parsedResponse && parsedResponse.rest_api_restart_required) {
                showControlFeedback(
                    (parsedResponse.message || "Config reloaded") +
                    " — a full server restart is required. Use the menu bar app to restart.",
                    "progress"
                );
            } else {
                showControlFeedback(parsedResponse ? parsedResponse.message : "Config reloaded", "success");
            }
        } else if (response.status === 409) {
            showControlFeedback("A generation is active; reload aborted. Wait for it to finish.", "error");
        } else if (parsedResponse && parsedResponse.message) {
            showControlFeedback(parsedResponse.message, "error");
        } else {
            showControlFeedback("Config reload failed (HTTP " + response.status + ")", "error");
        }
    } catch (fetchError) {
        showControlFeedback("Network error: " + fetchError.message, "error");
    } finally {
        reloadButton.disabled = false;
    }
}

async function stopServer() {
    const stopButton = document.getElementById("control-stop");
    stopButton.disabled = true;
    showControlFeedback("Stopping server… The console will go offline. Restart via the menu bar app or run astronomicald again.", "progress");
    try {
        await fetch(SERVER_SHUTDOWN_URL, { method: "POST" });
    } catch (fetchError) {
        // The server is shutting down; the fetch may fail. That's expected.
    }
    // The status polling will detect the daemon going offline and show "Unavailable".
}

function showControlFeedback(message, feedbackKind) {
    const feedbackElement = document.getElementById("control-feedback");
    feedbackElement.textContent = message;
    feedbackElement.className = "control-feedback feedback-" + feedbackKind;
    feedbackElement.hidden = false;
}
