// Astronomical Observatory telemetry and controls.
// All model output is rendered via textContent only; never innerHTML.

const STATUS_URL = "/v1/status";
const MODELS_URL = "/v1/models";
const CACHE_STATS_URL = "/v1/cache/stats";
const SYSTEM_TELEMETRY_URL = "/v1/system/telemetry";
const CONFIG_REVEAL_URL = "/v1/config/reveal";
const POLL_INTERVAL_MILLIS = 1000;

const OBSERVATORY_PATH_MAP = {
    overview: "/overview",
    chat: "/chat",
    experts: "/experts",
    "download-models": "/download-models",
    connect: "/connect",
    configuration: "/configuration"
};

// Retired addresses still open the page that replaced them: the Model page was
// folded into Overview, the Library page was renamed Download Models and later Models (keeping its route), and the
// Settings page was folded into Configuration.
const OBSERVATORY_RETIRED_PATH_DESTINATIONS = {
    "/model": "overview",
    "/library": "download-models",
    "/settings": "configuration"
};

const OBSERVATORY_PATH_TO_DESTINATION_MAP = Object.fromEntries(
    Object.entries(OBSERVATORY_PATH_MAP).map(([destination, path]) => [path, destination])
);

const OBSERVATORY_DEFAULT_DESTINATION = "overview";
const OBSERVATORY_DEFAULT_PATH = "/overview";

let selectedModelId = null;
// The model the worker reports as loaded (status ready_model_id), or null.
let loadedModelId = null;

document.addEventListener("DOMContentLoaded", () => {
    console.log("observatory boot");
    // Navigation runs first: it shows the view for the address (retired ones
    // included), and the Configuration editor loads only when its view is shown.
    wireObservatoryNavigation();
    wireDownloadModelsCatalog();
    wireConnectMaterial();
    wireExpertObservatory();
    wireConfigurationEditor();
    pollStatus();
    pollCacheStats();
    pollModels();
    setInterval(pollStatus, POLL_INTERVAL_MILLIS);
    setInterval(pollCacheStats, POLL_INTERVAL_MILLIS);
    setInterval(pollModels, POLL_INTERVAL_MILLIS);
    wireConfigRevealControl();
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
            || OBSERVATORY_RETIRED_PATH_DESTINATIONS[currentObservatoryPath]
            || OBSERVATORY_DEFAULT_DESTINATION;
        activateObservatoryView(matchedDestination, navigationButtons, observatoryViews, false);
        if (!OBSERVATORY_PATH_TO_DESTINATION_MAP[currentObservatoryPath]) {
            history.replaceState(null, "", OBSERVATORY_PATH_MAP[matchedDestination] || OBSERVATORY_DEFAULT_PATH);
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
        loadedModelId = data.ready_model_id || null;
        renderOverviewNow(data);
        renderOverviewMemory(data);
        renderOverviewModel(data, readExpertPlacement(data));
        renderOverviewSession(data);
        renderMemoryLimitControl(data);
        renderConfigurationNeuralEngineStatus(data);
        renderNeuralEnginePanel(data);
        renderModelDiscoveryDiagnostic(data.configuration);
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

function configurationRequiresRestart(configurationDocument) {
    if (!configurationDocument) { return false; }
    return configurationDocument.restart_required === true
        || (configurationDocument.configured_generation
            && configurationDocument.configured_generation
                !== configurationDocument.effective_generation);
}

function renderModelDiscoveryDiagnostic(configurationDocument) {
    const warningElement = document.getElementById("model-discovery-warning");
    const messageElement = document.getElementById("model-discovery-warning-message");
    const diagnostic = configurationDocument
        && Array.isArray(configurationDocument.model_discovery_diagnostics)
        ? configurationDocument.model_discovery_diagnostics[0]
        : null;
    if (!diagnostic || diagnostic.code !== "ambiguous_model_identity"
        || typeof diagnostic.model_id !== "string"
        || !Array.isArray(diagnostic.configured_root_numbers)) {
        warningElement.hidden = true;
        messageElement.textContent = "";
        // A reveal failure belongs to the warning it was shown in, not to the next one.
        const revealFeedback = document.getElementById("model-discovery-reveal-feedback");
        if (revealFeedback) { revealFeedback.textContent = ""; }
        return;
    }
    const configuredRootNumbers = diagnostic.configured_root_numbers.join(", ");
    messageElement.textContent = `Model ${diagnostic.model_id} appears in model_directories entries ${configuredRootNumbers}. Remove one duplicate root.`;
    warningElement.hidden = false;
}

// Each channel opens Thin Talk through its own scheme, because macOS hands a shared
// one to whichever installed app it prefers. There is deliberately no default: a
// channel missing here gets no link rather than a link to another instance.
// The daemon reports only these two channels, and an App Store build reports stable.
// Mirrors ApplicationChannel.chatURLScheme and xtask's channel profiles.
const THIN_TALK_CHAT_LINKS = new Map([
    ["development", "astronomical-dev://chat"],
    ["stable", "astronomical://chat"],
]);

function thinTalkChatLink(applicationChannel) {
    return THIN_TALK_CHAT_LINKS.get(applicationChannel) || null;
}

function renderThinTalkLink(applicationChannel) {
    const openThinTalkLink = document.getElementById("open-thin-talk");
    const chatLink = thinTalkChatLink(applicationChannel);
    if (chatLink) {
        openThinTalkLink.setAttribute("href", chatLink);
    } else {
        openThinTalkLink.removeAttribute("href");
    }
}

function renderApplicationIdentity(applicationIdentity) {
    const identityTitle = applicationIdentityTitle(applicationIdentity);
    document.getElementById("application-build-identity").textContent = identityTitle;
    document.getElementById("footer-application-identity").textContent = identityTitle;
    if (!applicationIdentity) { return; }
    document.getElementById("application-channel").textContent =
        applicationIdentity.channel_display_name || applicationIdentity.channel;
    document.body.dataset.applicationChannel = applicationIdentity.channel || "unknown";
    renderThinTalkLink(applicationIdentity.channel);
}

async function pollSystemTelemetry() {
    try {
        const response = await fetch(SYSTEM_TELEMETRY_URL);
        if (!response.ok) {
            renderOverviewMemoryPressure(null);
            return;
        }
        const telemetryDocument = await response.json();
        renderOverviewMemoryPressure(telemetryDocument.memory_pressure);
    } catch (fetchError) {
        renderOverviewMemoryPressure(null);
    }
}

function setStatusUnavailable() {
    const header = document.getElementById("status-header");
    header.className = "status-header unavailable";
    document.getElementById("status-word").textContent = "Unavailable";
    document.getElementById("status-model-id").textContent = "No model loaded";
    // With no answer, nothing on Overview may keep describing the last model.
    renderOverviewNow(null);
    renderOverviewModel(null, {});
    renderOverviewMemory({});
}

function renderStatusHeader(data) {
    const header = document.getElementById("status-header");
    const statusWord = document.getElementById("status-word");
    const modelIdLabel = document.getElementById("status-model-id");
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
    } else {
        // A ready worker with no model is the normal state after an idle
        // unload, so the header must stop naming the model it let go.
        modelIdLabel.textContent = "No model loaded";
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
        renderConnectionMaterial({
            consoleOrigin: connectionConsoleOrigin(),
            modelIdentifiers: connectionModelIdentifiers(data),
            selectedModelIdentifier: selectedModel ? selectedModel.id : ""
        });
        if (selectedModel) {
            selectedModelId = selectedModel.id;
        }
        rememberOverviewLoadedModelListing(models.find((model) => model.id === loadedModelId) || null);
    } catch (fetchError) {
        // Status polling handles the unavailable display; stay quiet here.
    }
}

function selectAdvertisedModel(advertisedModels, preferredModelId) {
    return advertisedModels.find((model) => model.id === preferredModelId)
        || advertisedModels[0]
        || null;
}

async function pollCacheStats() {
    try {
        const response = await fetch(CACHE_STATS_URL);
        if (!response.ok) { return; }
        const data = await response.json();
        renderOverviewPromptCache(data);
    } catch (fetchError) {
        // Cache panel stays at last-known values on transient fetch errors.
    }
}

function formatGigabytes(bytes) {
    // The user-facing gigabyte is the one Apple prints on the box: 2³⁰ bytes.
    return (bytes / 1073741824).toFixed(2) + " GB";
}

function readExpertPlacement(statusDocument) {
    if (typeof expertPlacementCopy !== "function") {
        return { stockSentence: "", replySentence: "" };
    }
    return expertPlacementCopy(statusDocument);
}

function wireConfigRevealControl() {
    document.getElementById("model-discovery-reveal-config").addEventListener("click", revealConfig);
}

// The failure is shown inside the discovery warning that holds the button, so
// it appears where the person clicked.
async function revealConfig() {
    const revealFeedback = document.getElementById("model-discovery-reveal-feedback");
    revealFeedback.textContent = "";
    try {
        const response = await fetch(CONFIG_REVEAL_URL, { method: "POST" });
        if (!response.ok) {
            revealFeedback.textContent = "Could not reveal the configuration file.";
        }
    } catch (fetchError) {
        revealFeedback.textContent = "Could not reveal the configuration file.";
    }
}
