// Astronomical Observatory Download Models catalog coordinator.
// Owns the one-shot catalog fetch, download state polling, search/filter state,
// and the rendering call into download-models-render.js. Catalog values enter the
// document only through textContent and created elements.

const LIBRARY_CATALOG_URL = "/v1/library/catalog";
const LIBRARY_DOWNLOAD_URL = "/v1/library/download";
// A local immutable endpoint should resolve promptly; the bound prevents a stale
// loading state when a daemon connection is interrupted without an immediate
// fetch error.
const LIBRARY_CATALOG_TIMEOUT_MILLIS = 10_000;
const DOWNLOAD_MODELS_STATUS_URL = "/v1/status";
const LIBRARY_REVEAL_URL = "/v1/library/reveal";
// Schema 3 adds the role, get-started, experimental, and memory fields that
// the Get started and More models pages are built from.
const LIBRARY_CATALOG_SCHEMA_VERSION = 3;
const DOWNLOAD_MODELS_ROLES = ["chat", "coding", "image_generation", "helper"];

let downloadModelsCatalogLoadPromise = null;
let downloadModelsCatalogDocument = null;
let downloadModelsCurrentDownload = { state: "idle" };
// Fingerprint and outcome of the last full catalog render; the 1s poll skips
// re-rendering when nothing changed (see renderDownloadModelsCatalogDocument).
let downloadModelsRenderedStateFingerprint = null;
let downloadModelsRenderedStateOutcome = "unavailable";
let downloadModelsPollingTimer = null;
let downloadModelsRefreshPromise = null;
let downloadModelsProgressSample = null;
let downloadModelsMeasuredBytesPerSecond = null;
// Active filter state: search query, family filter, and readiness filter.
let downloadModelsSearchQuery = "";
let downloadModelsFamilyFilter = "all";
let downloadModelsReadinessFilter = "all";
let downloadModelsExpandedModelId = null;
// Serving memory ceiling from /v1/status: the limit the engine loads models
// within, which can sit below the machine's RAM when the user configured a lower
// maximum. Null means unknown, and the cards then show no fit label rather than a guess.
let downloadModelsServingCeilingBytes = null;
// True when the user set a maximum below what this Mac allows. A model that
// cannot fit under such a limit is held back by the user's own choice, so its
// label says so instead of promising it will merely run slowly.
let downloadModelsCeilingIsUserLimit = false;
let downloadModelsServingCeilingReadAtMillis = 0;
const DOWNLOAD_MODELS_CEILING_REFRESH_MILLIS = 30_000;
// Which Download Models page is showing: "get-started" (default) or "more-models".
let downloadModelsActiveTab = "get-started";

function wireDownloadModelsCatalog() {
    if (downloadModelsCatalogLoadPromise !== null) {
        return downloadModelsCatalogLoadPromise;
    }
    const catalogContainer = document.getElementById("download-models-catalog");
    if (!catalogContainer) {
        return Promise.resolve("unavailable");
    }
    wireDownloadModelsControls();
    downloadModelsCatalogLoadPromise = loadDownloadModelsCatalog(catalogContainer);
    if (typeof setInterval === "function" && downloadModelsPollingTimer === null) {
        downloadModelsPollingTimer = setInterval(refreshDownloadModelsDownloadState, 1_000);
    }
    return downloadModelsCatalogLoadPromise;
}

function wireDownloadModelsControls() {
    const searchInput = document.getElementById("download-models-search");
    if (searchInput) {
        searchInput.addEventListener("input", () => {
            downloadModelsSearchQuery = searchInput.value.trim().toLowerCase();
            renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
        });
    }
    const familyFilterSelect = document.getElementById("download-models-family-filter");
    if (familyFilterSelect) {
        familyFilterSelect.addEventListener("change", () => {
            downloadModelsFamilyFilter = familyFilterSelect.value;
            renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
        });
    }
    const readinessFilterSelect = document.getElementById("download-models-readiness-filter");
    if (readinessFilterSelect) {
        readinessFilterSelect.addEventListener("change", () => {
            downloadModelsReadinessFilter = readinessFilterSelect.value;
            renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
        });
    }
}

async function loadDownloadModelsCatalog(
    catalogContainer,
    catalogTimeoutMillis = LIBRARY_CATALOG_TIMEOUT_MILLIS
) {
    const catalogAbortController = new AbortController();
    const catalogTimeout = setTimeout(
        () => catalogAbortController.abort(),
        catalogTimeoutMillis
    );
    try {
        const catalogResponse = await fetch(
            LIBRARY_CATALOG_URL,
            { signal: catalogAbortController.signal }
        );
        if (!catalogResponse.ok) {
            return renderDownloadModelsUnavailableState(catalogContainer);
        }
        const catalogDocument = await catalogResponse.json();
        downloadModelsCatalogDocument = catalogDocument;
        await refreshDownloadModelsServingCeiling();
        downloadModelsServingCeilingReadAtMillis = Date.now();
        return renderDownloadModelsCatalogDocument(catalogDocument, catalogContainer);
    } catch {
        return renderDownloadModelsUnavailableState(catalogContainer);
    } finally {
        clearTimeout(catalogTimeout);
    }
}

function renderDownloadModelsCatalogDocument(catalogDocument, catalogContainer) {
    const resolvedCatalogContainer = catalogContainer
        || document.getElementById("download-models-catalog");
    if (!resolvedCatalogContainer) {
        return "unavailable";
    }
    // The 1s download poll re-enters this function every tick; when no render
    // input moved, skip the rebuild so in-progress text selection and focus
    // survive (replacing nodes cuts selections out from under an active drag).
    const renderStateFingerprint = JSON.stringify([
        catalogDocument,
        downloadModelsCurrentDownload,
        downloadModelsSearchQuery,
        downloadModelsFamilyFilter,
        downloadModelsReadinessFilter,
        downloadModelsExpandedModelId,
        downloadModelsActiveTab,
        downloadModelsServingCeilingBytes,
        downloadModelsCeilingIsUserLimit,
        ...downloadModelsRemovalRenderState(), // changes no catalog field, so it counts here
    ]);
    if (renderStateFingerprint === downloadModelsRenderedStateFingerprint) {
        return downloadModelsRenderedStateOutcome;
    }
    downloadModelsRenderedStateFingerprint = renderStateFingerprint;
    const catalogRows = downloadModelsCatalogRowsFromDocument(catalogDocument);
    if (catalogRows === null) {
        downloadModelsRenderedStateOutcome = "unavailable";
        return renderDownloadModelsUnavailableState(resolvedCatalogContainer);
    }
    if (catalogRows.length === 0) {
        downloadModelsRenderedStateOutcome = "empty";
        return renderDownloadModelsEmptyState(resolvedCatalogContainer);
    }

    const statusMessage = resolveDownloadModelsStatusMessage();
    statusMessage.className = "download-models-catalog-state download-models-catalog-status-only";
    statusMessage.textContent = "Model catalog loaded.";
    resolvedCatalogContainer.dataset.downloadModelsState = "ready";

    // Snapshot the active search input BEFORE building the replacement bar:
    // the element being replaced is still the one the user is typing into,
    // and elements created for this render can shadow it in id lookups.
    // Capturing after the rebuild would observe the fresh, unfocused input
    // and silently drop focus on every keystroke.
    const searchSnapshot = captureDownloadModelsSearchSnapshot();
    resolvedCatalogContainer.replaceChildren(
        statusMessage,
        createDownloadModelsTabBar(),
        createDownloadModelsActivePage(catalogRows, catalogDocument)
    );
    // Restore focus and caret only after the replacement bar is connected to
    // the container, so the id lookup resolves the fresh input the user is
    // meant to keep typing into.
    restoreDownloadModelsSearchSnapshot(searchSnapshot);
    downloadModelsRenderedStateOutcome = "ready";
    return "ready";
}

// Switching pages re-renders at once and moves focus to the newly selected tab,
// because the rebuild replaces the tab the keyboard user was on.
function selectDownloadModelsTab(tabName) {
    downloadModelsActiveTab = tabName;
    renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
    const selectedTabButton = document.getElementById("download-models-tab-" + tabName);
    if (selectedTabButton) {
        selectedTabButton.focus({ preventScroll: true });
    }
}

function captureDownloadModelsSearchSnapshot() {
    const searchInput = document.getElementById("download-models-search");
    if (!searchInput) {
        return null;
    }
    return {
        focused: document.activeElement === searchInput,
        value: typeof searchInput.value === "string" ? searchInput.value : "",
        selectionStart: Number.isInteger(searchInput.selectionStart)
            ? searchInput.selectionStart
            : null
    };
}

function downloadModelsSearchCaretIndex(capturedCaretIndex, valueLength) {
    const caretIndex = capturedCaretIndex != null ? capturedCaretIndex : valueLength;
    return Math.min(Math.max(caretIndex, 0), valueLength);
}

function restoreDownloadModelsSearchSnapshot(searchSnapshot) {
    if (!searchSnapshot?.focused) {
        return;
    }
    const searchInput = document.getElementById("download-models-search");
    if (!searchInput || typeof searchInput.focus !== "function") {
        return;
    }
    // The filter bar can be above the viewport while the user is browsing
    // results; restoring keyboard focus must not move the page to that bar.
    searchInput.focus({ preventScroll: true });
    if (typeof searchInput.setSelectionRange === "function") {
        // Keep the caret where the user placed it so typing mid-query keeps
        // inserting there instead of jumping to the end of the text.
        const caretIndex = downloadModelsSearchCaretIndex(
            searchSnapshot.selectionStart,
            searchInput.value.length
        );
        searchInput.setSelectionRange(caretIndex, caretIndex);
    }
}

function filterDownloadModelsRows(catalogRows) {
    return catalogRows.filter((catalogRow) => {
        if (downloadModelsReadinessFilter === "ready" && !catalogRow.readyOnThisMac) return false;
        if (downloadModelsReadinessFilter === "downloadable" && catalogRow.readyOnThisMac) return false;
        if (downloadModelsFamilyFilter !== "all" && catalogRow.family !== downloadModelsFamilyFilter) return false;
        if (downloadModelsSearchQuery) {
            const haystack = [
                catalogRow.displayName,
                catalogRow.huggingfaceId,
                catalogRow.family,
                catalogRow.description || "",
                catalogRow.quantizationLabel || ""
            ].join(" ").toLowerCase();
            if (!haystack.includes(downloadModelsSearchQuery)) return false;
        }
        return true;
    });
}

function downloadModelsCatalogRowsFromDocument(catalogDocument) {
    if (!catalogDocument
        || catalogDocument.schema_version !== LIBRARY_CATALOG_SCHEMA_VERSION
        || !Array.isArray(catalogDocument.entries)) {
        return null;
    }
    const catalogRows = [];
    for (const catalogEntry of catalogDocument.entries) {
        if (!isRenderableDownloadModelsCatalogEntry(catalogEntry)) {
            return null;
        }
        catalogRows.push(downloadModelsCatalogRowFromEntry(catalogEntry));
    }
    return catalogRows;
}

function downloadModelsCatalogRowFromEntry(catalogEntry) {
    const capabilities = catalogEntry.capabilities || {};
    return {
        huggingfaceId: catalogEntry.huggingface_id,
        displayName: catalogEntry.display_name,
        family: catalogEntry.family,
        approximateSize: formatDownloadModelsSizeGigabytes(catalogEntry.approximate_size_bytes),
        approximateSizeBytes: catalogEntry.approximate_size_bytes,
        approximateMemoryBytes: catalogEntry.approximate_memory_bytes,
        role: catalogEntry.role,
        getStarted: catalogEntry.get_started === true,
        experimental: catalogEntry.experimental === true,
        retiring: catalogEntry.retiring === true,
        minimumMemoryBytes: isDownloadModelsStreamingFloor(catalogEntry.minimum_memory_bytes)
            ? catalogEntry.minimum_memory_bytes
            : null,
        readyOnThisMac: catalogEntry.ready_on_this_mac === true,
        removableFromDisk: catalogEntry.removable_from_disk === true,
        destinationDirectory: typeof catalogEntry.destination_directory === "string"
            ? catalogEntry.destination_directory
            : null,
        downloadState: catalogEntry.download_state || null,
        description: typeof catalogEntry.description === "string"
            ? catalogEntry.description
            : null,
        quantizationLabel: typeof catalogEntry.quantization_label === "string"
            ? catalogEntry.quantization_label
            : null,
        upstreamLicense: typeof catalogEntry.upstream_license === "string"
            ? catalogEntry.upstream_license
            : null,
        requestableModelId: typeof catalogEntry.requestable_model_id === "string"
            ? catalogEntry.requestable_model_id
            : null,
        supportsReasoning: capabilities.supports_reasoning === true,
        supportsVision: capabilities.supports_vision === true,
        supportsToolCalls: capabilities.supports_tool_calls === true,
        supportsImageGeneration: capabilities.supports_image_generation === true,
        supportsEmbeddings: capabilities.supports_embeddings === true,
        contextWindow: typeof capabilities.context_window === "number"
            ? capabilities.context_window
            : null,
        maxOutputTokens: typeof capabilities.max_output_tokens === "number"
            ? capabilities.max_output_tokens
            : null
    };
}

function isRenderableDownloadModelsCatalogEntry(catalogEntry) {
    return catalogEntry !== null
        && typeof catalogEntry === "object"
        && typeof catalogEntry.huggingface_id === "string"
        && catalogEntry.huggingface_id.trim().length > 0
        && typeof catalogEntry.revision === "string"
        && /^[0-9a-f]{40}$/i.test(catalogEntry.revision)
        && typeof catalogEntry.display_name === "string"
        && catalogEntry.display_name.trim().length > 0
        && typeof catalogEntry.family === "string"
        && catalogEntry.family.trim().length > 0
        && Number.isSafeInteger(catalogEntry.approximate_size_bytes)
        && catalogEntry.approximate_size_bytes > 0
        && Number.isSafeInteger(catalogEntry.approximate_memory_bytes)
        && catalogEntry.approximate_memory_bytes > 0
        && DOWNLOAD_MODELS_ROLES.includes(catalogEntry.role)
        && catalogEntry.public === true;
}

async function refreshDownloadModelsServingCeiling() {
    const statusDocument = await fetchDownloadModelsStatusDocument();
    downloadModelsServingCeilingBytes = readDownloadModelsServingCeilingBytes(statusDocument);
    downloadModelsCeilingIsUserLimit = readDownloadModelsCeilingIsUserLimit(statusDocument);
}

// Memory ceiling is advisory: any failure leaves it unknown rather than
// failing the catalog, so the page still renders without fit labels.
async function fetchDownloadModelsStatusDocument() {
    const statusAbortController = new AbortController();
    const statusTimeout = setTimeout(
        () => statusAbortController.abort(),
        LIBRARY_CATALOG_TIMEOUT_MILLIS
    );
    try {
        const statusResponse = await fetch(
            DOWNLOAD_MODELS_STATUS_URL,
            { signal: statusAbortController.signal }
        );
        if (!statusResponse.ok) return null;
        return await statusResponse.json();
    } catch {
        return null;
    } finally {
        clearTimeout(statusTimeout);
    }
}

// Prefer the effective ceiling: the engine serves within the user's configured
// maximum, so a model above it will not run in memory even on a Mac with more RAM.
// The machine maximum is only a fallback for status documents that omit the effective value.
function readDownloadModelsServingCeilingBytes(statusDocument) {
    const effectiveCeilingBytes = statusDocument?.mlx_memory_ceiling_bytes;
    if (isPositiveFiniteByteCount(effectiveCeilingBytes)) return effectiveCeilingBytes;
    const machineCeilingBytes = statusDocument?.machine_mlx_memory_ceiling_bytes;
    return isPositiveFiniteByteCount(machineCeilingBytes) ? machineCeilingBytes : null;
}

// The daemon omits the floor for models that must fit whole; anything that is
// not a positive safe integer is treated the same way, so a malformed floor can
// never turn into a streaming promise.
function isDownloadModelsStreamingFloor(minimumMemoryBytes) {
    return Number.isSafeInteger(minimumMemoryBytes) && minimumMemoryBytes > 0;
}

// A configured maximum below the machine's own ceiling is a limit the user
// chose; a missing configuration, or one at or above the machine, is not.
function readDownloadModelsCeilingIsUserLimit(statusDocument) {
    const configuredGigabytes = statusDocument?.configured_maximum_mlx_memory_gb;
    const machineCeilingBytes = statusDocument?.machine_mlx_memory_ceiling_bytes;
    if (!Number.isFinite(configuredGigabytes) || !isPositiveFiniteByteCount(machineCeilingBytes)) {
        return false;
    }
    return configuredGigabytes * 1073741824 < machineCeilingBytes;
}

function isPositiveFiniteByteCount(byteCount) {
    return Number.isFinite(byteCount) && byteCount > 0;
}

function formatDownloadModelsSizeGigabytes(approximateSizeBytes) {
    // The user-facing gigabyte is the one Apple prints on the box: 2³⁰ bytes.
    return (approximateSizeBytes / 1073741824).toFixed(2) + " GB";
}

function downloadModelsDestinationPath(catalogRow) {
    const activeDownload = downloadModelsCurrentDownload.huggingface_id === catalogRow.huggingfaceId
        ? downloadModelsCurrentDownload : null;
    return activeDownload?.destination_directory || catalogRow.destinationDirectory || "";
}

function downloadModelsProgressPercent(activeDownload) {
    if (!activeDownload || !(activeDownload.bytes_total > 0)) return 0;
    return Math.min(
        99,
        Math.floor(activeDownload.bytes_completed * 100 / activeDownload.bytes_total)
    );
}

/** The download state one row shows. The status poll is the live truth for the model it names.
 * The catalog's copy is only a fallback from before that poll, and it is ignored once the poll
 * names another model: only one download holds the download slot, so the copy is stale, and its
 * Resume and Cancel buttons would act on the other model's download. */
function downloadModelsRowDownloadState(catalogRow) {
    const currentModelId = downloadModelsCurrentDownload.huggingface_id;
    if (currentModelId === catalogRow.huggingfaceId) return downloadModelsCurrentDownload.state;
    return currentModelId ? null : catalogRow.downloadState;
}

function downloadModelsStateTitle(catalogRow) {
    if (catalogRow.readyOnThisMac) return "Ready to use";
    const activeDownload = downloadModelsCurrentDownload.huggingface_id === catalogRow.huggingfaceId
        ? downloadModelsCurrentDownload : null;
    const downloadState = downloadModelsRowDownloadState(catalogRow);
    if (!downloadState) return "Available to download";
    if ((downloadState === "downloading" || downloadState === "paused")
        && activeDownload?.bytes_total > 0) {
        const percent = downloadModelsProgressPercent(activeDownload);
        const completedSize = formatDownloadModelsSizeGigabytes(activeDownload.bytes_completed);
        const totalSize = formatDownloadModelsSizeGigabytes(activeDownload.bytes_total);
        const prefix = downloadState === "paused" ? "Paused at " : "Downloading ";
        return prefix + percent + "% · " + completedSize + " of " + totalSize;
    }
    const stateTitles = {
        checking_disk: "Checking disk space",
        fetching_manifest: "Preparing download",
        downloading: "Downloading",
        paused: "Paused",
        verifying: "Download complete. Checking the files…",
        publishing: "Download complete. Adding it to your models…",
        failed: downloadModelsFailureTitle(activeDownload?.error_code)
    };
    return stateTitles[downloadState] || "Preparing download";
}

function downloadModelsProgressDetail(catalogRow) {
    const activeDownload = downloadModelsCurrentDownload.huggingface_id === catalogRow.huggingfaceId
        ? downloadModelsCurrentDownload : null;
    if (!activeDownload || activeDownload.state !== "downloading"
        || !(activeDownload.bytes_total > 0)) {
        return "";
    }
    const currentFile = activeDownload.current_file_relative_path
        ? "Now saving " + activeDownload.current_file_relative_path
        : "";
    const bytesPerSecond = downloadModelsMeasuredBytesPerSecond;
    if (!(bytesPerSecond > 0)) {
        return currentFile || "Measuring transfer rate…";
    }
    const remainingBytes = activeDownload.bytes_total - activeDownload.bytes_completed;
    const rateDetail = formatDownloadModelsRate(bytesPerSecond)
        + " · "
        + formatDownloadModelsRemainingTime(remainingBytes / bytesPerSecond);
    return currentFile ? currentFile + " · " + rateDetail : rateDetail;
}

function smoothDownloadModelsTransferRate(previousBytesPerSecond, measuredBytesPerSecond) {
    if (!(previousBytesPerSecond > 0)) return measuredBytesPerSecond;
    // A quarter of each new sample reacts within a few polls while preventing one CDN burst or
    // quiet interval from turning a large model's estimate into misleading hours.
    return previousBytesPerSecond * 0.75 + measuredBytesPerSecond * 0.25;
}

function recordDownloadModelsProgressMeasurement(activeDownload, sampledAtMillis = Date.now()) {
    const previousSample = downloadModelsProgressSample;
    const canMeasure = activeDownload?.state === "downloading"
        && typeof activeDownload.huggingface_id === "string"
        && Number.isFinite(activeDownload.bytes_completed)
        && previousSample?.huggingfaceId === activeDownload.huggingface_id
        && activeDownload.bytes_completed >= previousSample.bytesCompleted;
    if (canMeasure) {
        const elapsedSeconds = (sampledAtMillis - previousSample.sampledAtMillis) / 1_000;
        const receivedBytes = activeDownload.bytes_completed - previousSample.bytesCompleted;
        if (elapsedSeconds > 0 && receivedBytes > 0) {
            downloadModelsMeasuredBytesPerSecond = smoothDownloadModelsTransferRate(
                downloadModelsMeasuredBytesPerSecond,
                receivedBytes / elapsedSeconds
            );
        }
    } else {
        downloadModelsMeasuredBytesPerSecond = null;
    }
    downloadModelsProgressSample = activeDownload?.state === "downloading"
        && activeDownload.huggingface_id
        && Number.isFinite(activeDownload.bytes_completed)
        ? {
            huggingfaceId: activeDownload.huggingface_id,
            bytesCompleted: activeDownload.bytes_completed,
            sampledAtMillis
        }
        : null;
}

function downloadModelsFailureTitle(errorCode) {
    const messages = {
        insufficient_disk: "Not enough disk space",
        download_gated: "This model requires Hugging Face access",
        checksum_mismatch: "Downloaded files did not pass verification",
        model_already_present: "Another model already uses this download location. Move it elsewhere, then cancel this download and try again.",
        model_not_executable: "This version of Astronomical cannot run this model. Cancel to free the download."
    };
    return messages[errorCode] || "Download failed. You can resume it.";
}

// Failures that resuming cannot fix: the same files would be refused again, so
// the card offers only Cancel instead of a Resume that fails the same way.
const DOWNLOAD_MODELS_CANCEL_ONLY_ERROR_CODES = ["model_already_present", "model_not_executable"];

function downloadModelsActionButtons(catalogRow) {
    if (catalogRow.readyOnThisMac) return [];
    const isCurrentDownload = downloadModelsCurrentDownload.huggingface_id === catalogRow.huggingfaceId;
    const state = downloadModelsRowDownloadState(catalogRow);
    if (state === "verifying" || state === "publishing") return [];
    if (isCurrentDownload && state === "failed"
        && DOWNLOAD_MODELS_CANCEL_ONLY_ERROR_CODES.includes(downloadModelsCurrentDownload.error_code)) {
        return [downloadModelsActionButton("Cancel", "cancel")];
    }
    if (state === "paused" || state === "failed") {
        return [downloadModelsActionButton("Resume", "resume"), downloadModelsActionButton("Cancel", "cancel")];
    }
    if (isCurrentDownload && state !== "idle") {
        return [downloadModelsActionButton("Pause", "pause"), downloadModelsActionButton("Cancel", "cancel")];
    }
    const downloadButton = downloadModelsActionButton(
        "Download", "start", catalogRow.huggingfaceId, "download-models-download-button");
    // A failed download of another model does not hold the download slot: the daemon discards it when
    // this download starts, so the button stays usable and nobody has to cancel it first.
    downloadButton.disabled = downloadModelsCurrentDownload.state !== "idle"
        && downloadModelsCurrentDownload.state !== "failed";
    return [downloadButton];
}

function downloadModelsActionButton(label, action, huggingfaceId = null, extraClassName = "") {
    const button = document.createElement("button");
    const buttonStyle = action === "cancel" ? "button button-danger" : "button button-primary";
    button.className = extraClassName ? buttonStyle + " " + extraClassName : buttonStyle;
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", () => performDownloadModelsAction(action, huggingfaceId));
    return button;
}

async function performDownloadModelsAction(action, huggingfaceId) {
    const actionPath = action === "start" ? LIBRARY_DOWNLOAD_URL : LIBRARY_DOWNLOAD_URL + "/" + action;
    const request = { method: "POST", headers: {} };
    if (action === "start") {
        request.headers["Content-Type"] = "application/json";
        request.body = JSON.stringify({ huggingface_id: huggingfaceId });
    }
    try {
        const actionResponse = await fetch(actionPath, request);
        const actionDocument = await actionResponse.json();
        if (!actionResponse.ok) {
            throw new Error(actionDocument?.error?.message || "Download control failed");
        }
        downloadModelsCurrentDownload = actionDocument;
    } catch {
        downloadModelsCurrentDownload = { state: "failed", error_code: "download_failed" };
    }
    renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
}

function refreshDownloadModelsDownloadState() {
    if (downloadModelsRefreshPromise !== null) return downloadModelsRefreshPromise;
    downloadModelsRefreshPromise = refreshDownloadModelsDownloadStateOnce()
        .finally(() => { downloadModelsRefreshPromise = null; });
    return downloadModelsRefreshPromise;
}

async function refreshDownloadModelsDownloadStateOnce() {
    try {
        if (!downloadModelsCatalogDocument) {
            const catalogContainer = document.getElementById("download-models-catalog");
            if (!catalogContainer) return;
            if (downloadModelsCatalogLoadPromise !== null) {
                await downloadModelsCatalogLoadPromise;
            }
            if (downloadModelsCatalogDocument) {
                renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument, catalogContainer);
            } else {
                await loadDownloadModelsCatalog(catalogContainer);
            }
            if (!downloadModelsCatalogDocument) return;
        }
        const downloadResponse = await fetch(LIBRARY_DOWNLOAD_URL);
        if (!downloadResponse.ok) return;
        downloadModelsCurrentDownload = await downloadResponse.json();
        recordDownloadModelsProgressMeasurement(downloadModelsCurrentDownload);
        if (downloadModelsCurrentDownload.state === "idle"
            || downloadModelsCurrentDownload.state === "verifying"
            || downloadModelsCurrentDownload.state === "publishing") {
            await refetchDownloadModelsCatalogDocument();
        }
        renderDownloadModelsCatalogDocument(downloadModelsCatalogDocument);
    } catch {
        // Polling is self-healing and must never cancel daemon-owned work.
    }
}

// The ceiling is re-read while still unknown (the worker reports it after
// startup) and otherwise at most every DOWNLOAD_MODELS_CEILING_REFRESH_MILLIS, so a
// memory limit changed on the Configuration page reaches the fit labels within
// seconds without doubling every request of the idle 1s catalog poll.
async function refetchDownloadModelsCatalogDocument(nowMillis = Date.now()) {
    const catalogResponse = await fetch(LIBRARY_CATALOG_URL);
    if (!catalogResponse.ok) return;
    downloadModelsCatalogDocument = await catalogResponse.json();
    const isCeilingStale = nowMillis - downloadModelsServingCeilingReadAtMillis
        >= DOWNLOAD_MODELS_CEILING_REFRESH_MILLIS;
    if (downloadModelsServingCeilingBytes === null || isCeilingStale) {
        await refreshDownloadModelsServingCeiling();
        downloadModelsServingCeilingReadAtMillis = nowMillis;
    }
}

// The page names the model, never a path: the daemon resolves the folder from
// its own discovery and download state and asks Finder to select it. Returns
// whether Finder accepted the request.
async function revealDownloadModelsModelInFinder(huggingfaceId) {
    try {
        const revealResponse = await fetch(LIBRARY_REVEAL_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ huggingface_id: huggingfaceId })
        });
        return revealResponse.ok;
    } catch {
        return false;
    }
}

function renderDownloadModelsEmptyState(catalogContainer) {
    return renderDownloadModelsStatusState(
        catalogContainer,
        "empty",
        "No catalog entries are available in this release."
    );
}

function renderDownloadModelsUnavailableState(catalogContainer) {
    return renderDownloadModelsStatusState(
        catalogContainer,
        "unavailable",
        "The model catalog is unavailable."
    );
}

function renderDownloadModelsStatusState(catalogContainer, downloadModelsState, message) {
    const statusMessage = resolveDownloadModelsStatusMessage();
    statusMessage.className = "download-models-catalog-state";
    catalogContainer.dataset.downloadModelsState = downloadModelsState;
    // Preserve one live-region node so assistive technology observes state
    // transitions reliably.
    catalogContainer.replaceChildren(statusMessage);
    statusMessage.textContent = message;
    return downloadModelsState;
}

function resolveDownloadModelsStatusMessage() {
    const existingStatusMessage = document.getElementById("download-models-catalog-status");
    if (existingStatusMessage) {
        return existingStatusMessage;
    }
    const statusMessage = document.createElement("p");
    statusMessage.id = "download-models-catalog-status";
    statusMessage.setAttribute("role", "status");
    statusMessage.setAttribute("aria-live", "polite");
    return statusMessage;
}
