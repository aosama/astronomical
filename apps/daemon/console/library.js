// Astronomical Observatory Library catalog coordinator.
// Owns the one-shot catalog fetch, download state polling, search/filter state,
// and the rendering call into library-render.js. Catalog values enter the
// document only through textContent and created elements.

const LIBRARY_CATALOG_URL = "/v1/library/catalog";
const LIBRARY_DOWNLOAD_URL = "/v1/library/download";
// A local immutable endpoint should resolve promptly; the bound prevents a stale
// loading state when a daemon connection is interrupted without an immediate
// fetch error.
const LIBRARY_CATALOG_TIMEOUT_MILLIS = 10_000;
const LIBRARY_STATUS_URL = "/v1/status";
const LIBRARY_REVEAL_URL = "/v1/library/reveal";
// Schema 3 adds the role, get-started, experimental, and memory fields that
// the Get started and More models pages are built from.
const LIBRARY_CATALOG_SCHEMA_VERSION = 3;
const LIBRARY_ROLES = ["chat", "coding", "image_generation", "helper"];

let libraryCatalogLoadPromise = null;
let libraryCatalogDocument = null;
let libraryCurrentDownload = { state: "idle" };
// Fingerprint and outcome of the last full catalog render; the 1s poll skips
// re-rendering when nothing changed (see renderLibraryCatalogDocument).
let libraryRenderedStateFingerprint = null;
let libraryRenderedStateOutcome = "unavailable";
let libraryPollingTimer = null;
let libraryRefreshPromise = null;
let libraryProgressSample = null;
let libraryMeasuredBytesPerSecond = null;
// Active filter state: search query, family filter, and readiness filter.
let librarySearchQuery = "";
let libraryFamilyFilter = "all";
let libraryReadinessFilter = "all";
let libraryExpandedModelId = null;
// Serving memory ceiling from /v1/status: the limit the engine loads models
// within, which can sit below the machine's RAM when the user configured a lower
// maximum. Null means unknown, and the cards then show no fit label rather than a guess.
let libraryServingCeilingBytes = null;
// True when the user set a maximum below what this Mac allows. A model that
// cannot fit under such a limit is held back by the user's own choice, so its
// label says so instead of promising it will merely run slowly.
let libraryCeilingIsUserLimit = false;
let libraryServingCeilingReadAtMillis = 0;
const LIBRARY_CEILING_REFRESH_MILLIS = 30_000;
// Which Library page is showing: "get-started" (default) or "more-models".
let libraryActiveTab = "get-started";

function wireLibraryCatalog() {
    if (libraryCatalogLoadPromise !== null) {
        return libraryCatalogLoadPromise;
    }
    const catalogContainer = document.getElementById("library-catalog");
    if (!catalogContainer) {
        return Promise.resolve("unavailable");
    }
    wireLibraryControls();
    libraryCatalogLoadPromise = loadLibraryCatalog(catalogContainer);
    if (typeof setInterval === "function" && libraryPollingTimer === null) {
        libraryPollingTimer = setInterval(refreshLibraryDownloadState, 1_000);
    }
    return libraryCatalogLoadPromise;
}

function wireLibraryControls() {
    const searchInput = document.getElementById("library-search");
    if (searchInput) {
        searchInput.addEventListener("input", () => {
            librarySearchQuery = searchInput.value.trim().toLowerCase();
            renderLibraryCatalogDocument(libraryCatalogDocument);
        });
    }
    const familyFilterSelect = document.getElementById("library-family-filter");
    if (familyFilterSelect) {
        familyFilterSelect.addEventListener("change", () => {
            libraryFamilyFilter = familyFilterSelect.value;
            renderLibraryCatalogDocument(libraryCatalogDocument);
        });
    }
    const readinessFilterSelect = document.getElementById("library-readiness-filter");
    if (readinessFilterSelect) {
        readinessFilterSelect.addEventListener("change", () => {
            libraryReadinessFilter = readinessFilterSelect.value;
            renderLibraryCatalogDocument(libraryCatalogDocument);
        });
    }
}

async function loadLibraryCatalog(
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
            return renderLibraryUnavailableState(catalogContainer);
        }
        const catalogDocument = await catalogResponse.json();
        libraryCatalogDocument = catalogDocument;
        await refreshLibraryServingCeiling();
        libraryServingCeilingReadAtMillis = Date.now();
        return renderLibraryCatalogDocument(catalogDocument, catalogContainer);
    } catch {
        return renderLibraryUnavailableState(catalogContainer);
    } finally {
        clearTimeout(catalogTimeout);
    }
}

function renderLibraryCatalogDocument(catalogDocument, catalogContainer) {
    const resolvedCatalogContainer = catalogContainer
        || document.getElementById("library-catalog");
    if (!resolvedCatalogContainer) {
        return "unavailable";
    }
    // The 1s download poll re-enters this function every tick; when no render
    // input moved, skip the rebuild so in-progress text selection and focus
    // survive (replacing nodes cuts selections out from under an active drag).
    const renderStateFingerprint = JSON.stringify([
        catalogDocument,
        libraryCurrentDownload,
        librarySearchQuery,
        libraryFamilyFilter,
        libraryReadinessFilter,
        libraryExpandedModelId,
        libraryActiveTab,
        libraryServingCeilingBytes,
        libraryCeilingIsUserLimit,
    ]);
    if (renderStateFingerprint === libraryRenderedStateFingerprint) {
        return libraryRenderedStateOutcome;
    }
    libraryRenderedStateFingerprint = renderStateFingerprint;
    const catalogRows = libraryCatalogRowsFromDocument(catalogDocument);
    if (catalogRows === null) {
        libraryRenderedStateOutcome = "unavailable";
        return renderLibraryUnavailableState(resolvedCatalogContainer);
    }
    if (catalogRows.length === 0) {
        libraryRenderedStateOutcome = "empty";
        return renderLibraryEmptyState(resolvedCatalogContainer);
    }

    const statusMessage = resolveLibraryStatusMessage();
    statusMessage.className = "library-catalog-state library-catalog-status-only";
    statusMessage.textContent = "Model catalog loaded.";
    resolvedCatalogContainer.dataset.libraryState = "ready";

    // Snapshot the active search input BEFORE building the replacement bar:
    // the element being replaced is still the one the user is typing into,
    // and elements created for this render can shadow it in id lookups.
    // Capturing after the rebuild would observe the fresh, unfocused input
    // and silently drop focus on every keystroke.
    const searchSnapshot = captureLibrarySearchSnapshot();
    resolvedCatalogContainer.replaceChildren(
        statusMessage,
        createLibraryTabBar(),
        createLibraryActivePage(catalogRows, catalogDocument)
    );
    // Restore focus and caret only after the replacement bar is connected to
    // the container, so the id lookup resolves the fresh input the user is
    // meant to keep typing into.
    restoreLibrarySearchSnapshot(searchSnapshot);
    libraryRenderedStateOutcome = "ready";
    return "ready";
}

// Switching pages re-renders at once and moves focus to the newly selected tab,
// because the rebuild replaces the tab the keyboard user was on.
function selectLibraryTab(tabName) {
    libraryActiveTab = tabName;
    renderLibraryCatalogDocument(libraryCatalogDocument);
    const selectedTabButton = document.getElementById("library-tab-" + tabName);
    if (selectedTabButton) {
        selectedTabButton.focus({ preventScroll: true });
    }
}

function captureLibrarySearchSnapshot() {
    const searchInput = document.getElementById("library-search");
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

function librarySearchCaretIndex(capturedCaretIndex, valueLength) {
    const caretIndex = capturedCaretIndex != null ? capturedCaretIndex : valueLength;
    return Math.min(Math.max(caretIndex, 0), valueLength);
}

function restoreLibrarySearchSnapshot(searchSnapshot) {
    if (!searchSnapshot?.focused) {
        return;
    }
    const searchInput = document.getElementById("library-search");
    if (!searchInput || typeof searchInput.focus !== "function") {
        return;
    }
    // The filter bar can be above the viewport while the user is browsing
    // results; restoring keyboard focus must not move the page to that bar.
    searchInput.focus({ preventScroll: true });
    if (typeof searchInput.setSelectionRange === "function") {
        // Keep the caret where the user placed it so typing mid-query keeps
        // inserting there instead of jumping to the end of the text.
        const caretIndex = librarySearchCaretIndex(
            searchSnapshot.selectionStart,
            searchInput.value.length
        );
        searchInput.setSelectionRange(caretIndex, caretIndex);
    }
}

function filterLibraryRows(catalogRows) {
    return catalogRows.filter((catalogRow) => {
        if (libraryReadinessFilter === "ready" && !catalogRow.readyOnThisMac) return false;
        if (libraryReadinessFilter === "downloadable" && catalogRow.readyOnThisMac) return false;
        if (libraryFamilyFilter !== "all" && catalogRow.family !== libraryFamilyFilter) return false;
        if (librarySearchQuery) {
            const haystack = [
                catalogRow.displayName,
                catalogRow.huggingfaceId,
                catalogRow.family,
                catalogRow.description || "",
                catalogRow.quantizationLabel || ""
            ].join(" ").toLowerCase();
            if (!haystack.includes(librarySearchQuery)) return false;
        }
        return true;
    });
}

function libraryCatalogRowsFromDocument(catalogDocument) {
    if (!catalogDocument
        || catalogDocument.schema_version !== LIBRARY_CATALOG_SCHEMA_VERSION
        || !Array.isArray(catalogDocument.entries)) {
        return null;
    }
    const catalogRows = [];
    for (const catalogEntry of catalogDocument.entries) {
        if (!isRenderableLibraryCatalogEntry(catalogEntry)) {
            return null;
        }
        catalogRows.push(libraryCatalogRowFromEntry(catalogEntry));
    }
    return catalogRows;
}

function libraryCatalogRowFromEntry(catalogEntry) {
    const capabilities = catalogEntry.capabilities || {};
    return {
        huggingfaceId: catalogEntry.huggingface_id,
        displayName: catalogEntry.display_name,
        family: catalogEntry.family,
        approximateSize: formatLibrarySizeGigabytes(catalogEntry.approximate_size_bytes),
        approximateSizeBytes: catalogEntry.approximate_size_bytes,
        approximateMemoryBytes: catalogEntry.approximate_memory_bytes,
        role: catalogEntry.role,
        getStarted: catalogEntry.get_started === true,
        experimental: catalogEntry.experimental === true,
        minimumMemoryBytes: isLibraryStreamingFloor(catalogEntry.minimum_memory_bytes)
            ? catalogEntry.minimum_memory_bytes
            : null,
        readyOnThisMac: catalogEntry.ready_on_this_mac === true,
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

function isRenderableLibraryCatalogEntry(catalogEntry) {
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
        && LIBRARY_ROLES.includes(catalogEntry.role)
        && catalogEntry.public === true;
}

async function refreshLibraryServingCeiling() {
    const statusDocument = await fetchLibraryStatusDocument();
    libraryServingCeilingBytes = readLibraryServingCeilingBytes(statusDocument);
    libraryCeilingIsUserLimit = readLibraryCeilingIsUserLimit(statusDocument);
}

// Memory ceiling is advisory: any failure leaves it unknown rather than
// failing the catalog, so the page still renders without fit labels.
async function fetchLibraryStatusDocument() {
    const statusAbortController = new AbortController();
    const statusTimeout = setTimeout(
        () => statusAbortController.abort(),
        LIBRARY_CATALOG_TIMEOUT_MILLIS
    );
    try {
        const statusResponse = await fetch(
            LIBRARY_STATUS_URL,
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
function readLibraryServingCeilingBytes(statusDocument) {
    const effectiveCeilingBytes = statusDocument?.mlx_memory_ceiling_bytes;
    if (isPositiveFiniteByteCount(effectiveCeilingBytes)) return effectiveCeilingBytes;
    const machineCeilingBytes = statusDocument?.machine_mlx_memory_ceiling_bytes;
    return isPositiveFiniteByteCount(machineCeilingBytes) ? machineCeilingBytes : null;
}

// The daemon omits the floor for models that must fit whole; anything that is
// not a positive safe integer is treated the same way, so a malformed floor can
// never turn into a streaming promise.
function isLibraryStreamingFloor(minimumMemoryBytes) {
    return Number.isSafeInteger(minimumMemoryBytes) && minimumMemoryBytes > 0;
}

// A configured maximum below the machine's own ceiling is a limit the user
// chose; a missing configuration, or one at or above the machine, is not.
function readLibraryCeilingIsUserLimit(statusDocument) {
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

function formatLibrarySizeGigabytes(approximateSizeBytes) {
    // The user-facing gigabyte is the one Apple prints on the box: 2³⁰ bytes.
    return (approximateSizeBytes / 1073741824).toFixed(2) + " GB";
}

function libraryDestinationPath(catalogRow) {
    const activeDownload = libraryCurrentDownload.huggingface_id === catalogRow.huggingfaceId
        ? libraryCurrentDownload : null;
    return activeDownload?.destination_directory || catalogRow.destinationDirectory || "";
}

function libraryProgressPercent(activeDownload) {
    if (!activeDownload || !(activeDownload.bytes_total > 0)) return 0;
    return Math.min(
        99,
        Math.floor(activeDownload.bytes_completed * 100 / activeDownload.bytes_total)
    );
}

/** The download state one row shows. The status poll is the live truth for the model it names.
 * The catalog's copy is only a fallback from before that poll, and it is ignored once the poll
 * names another model: only one download holds the Library, so the copy is stale, and its
 * Resume and Cancel buttons would act on the other model's download. */
function libraryRowDownloadState(catalogRow) {
    const currentModelId = libraryCurrentDownload.huggingface_id;
    if (currentModelId === catalogRow.huggingfaceId) return libraryCurrentDownload.state;
    return currentModelId ? null : catalogRow.downloadState;
}

function libraryStateTitle(catalogRow) {
    if (catalogRow.readyOnThisMac) return "Ready to use";
    const activeDownload = libraryCurrentDownload.huggingface_id === catalogRow.huggingfaceId
        ? libraryCurrentDownload : null;
    const downloadState = libraryRowDownloadState(catalogRow);
    if (!downloadState) return "Available to download";
    if ((downloadState === "downloading" || downloadState === "paused")
        && activeDownload?.bytes_total > 0) {
        const percent = libraryProgressPercent(activeDownload);
        const completedSize = formatLibrarySizeGigabytes(activeDownload.bytes_completed);
        const totalSize = formatLibrarySizeGigabytes(activeDownload.bytes_total);
        const prefix = downloadState === "paused" ? "Paused at " : "Downloading ";
        return prefix + percent + "% · " + completedSize + " of " + totalSize;
    }
    const stateTitles = {
        checking_disk: "Checking disk space",
        fetching_manifest: "Preparing download",
        downloading: "Downloading",
        paused: "Paused",
        verifying: "Download complete. Checking the files…",
        publishing: "Download complete. Adding it to Library…",
        failed: libraryFailureTitle(activeDownload?.error_code)
    };
    return stateTitles[downloadState] || "Preparing download";
}

function libraryProgressDetail(catalogRow) {
    const activeDownload = libraryCurrentDownload.huggingface_id === catalogRow.huggingfaceId
        ? libraryCurrentDownload : null;
    if (!activeDownload || activeDownload.state !== "downloading"
        || !(activeDownload.bytes_total > 0)) {
        return "";
    }
    const currentFile = activeDownload.current_file_relative_path
        ? "Now saving " + activeDownload.current_file_relative_path
        : "";
    const bytesPerSecond = libraryMeasuredBytesPerSecond;
    if (!(bytesPerSecond > 0)) {
        return currentFile || "Measuring transfer rate…";
    }
    const remainingBytes = activeDownload.bytes_total - activeDownload.bytes_completed;
    const rateDetail = formatLibraryRate(bytesPerSecond)
        + " · "
        + formatLibraryRemainingTime(remainingBytes / bytesPerSecond);
    return currentFile ? currentFile + " · " + rateDetail : rateDetail;
}

function smoothLibraryTransferRate(previousBytesPerSecond, measuredBytesPerSecond) {
    if (!(previousBytesPerSecond > 0)) return measuredBytesPerSecond;
    // A quarter of each new sample reacts within a few polls while preventing one CDN burst or
    // quiet interval from turning a large model's estimate into misleading hours.
    return previousBytesPerSecond * 0.75 + measuredBytesPerSecond * 0.25;
}

function recordLibraryProgressMeasurement(activeDownload, sampledAtMillis = Date.now()) {
    const previousSample = libraryProgressSample;
    const canMeasure = activeDownload?.state === "downloading"
        && typeof activeDownload.huggingface_id === "string"
        && Number.isFinite(activeDownload.bytes_completed)
        && previousSample?.huggingfaceId === activeDownload.huggingface_id
        && activeDownload.bytes_completed >= previousSample.bytesCompleted;
    if (canMeasure) {
        const elapsedSeconds = (sampledAtMillis - previousSample.sampledAtMillis) / 1_000;
        const receivedBytes = activeDownload.bytes_completed - previousSample.bytesCompleted;
        if (elapsedSeconds > 0 && receivedBytes > 0) {
            libraryMeasuredBytesPerSecond = smoothLibraryTransferRate(
                libraryMeasuredBytesPerSecond,
                receivedBytes / elapsedSeconds
            );
        }
    } else {
        libraryMeasuredBytesPerSecond = null;
    }
    libraryProgressSample = activeDownload?.state === "downloading"
        && activeDownload.huggingface_id
        && Number.isFinite(activeDownload.bytes_completed)
        ? {
            huggingfaceId: activeDownload.huggingface_id,
            bytesCompleted: activeDownload.bytes_completed,
            sampledAtMillis
        }
        : null;
}

function formatLibraryRate(bytesPerSecond) {
    if (bytesPerSecond >= 1073741824) {
        return (bytesPerSecond / 1073741824).toFixed(1) + " GB/s";
    }
    if (bytesPerSecond >= 1_000_000) {
        return (bytesPerSecond / 1_000_000).toFixed(1) + " MB/s";
    }
    return Math.max(1, Math.round(bytesPerSecond / 1_000)) + " KB/s";
}

function formatLibraryRemainingTime(remainingSeconds) {
    if (!(remainingSeconds > 0) || !Number.isFinite(remainingSeconds)) {
        return "Calculating time remaining";
    }
    if (remainingSeconds < 60) return "Less than a minute left";
    const remainingMinutes = Math.round(remainingSeconds / 60);
    if (remainingMinutes < 60) {
        return remainingMinutes === 1
            ? "About 1 minute left"
            : "About " + remainingMinutes + " minutes left";
    }
    const remainingHours = Math.round(remainingMinutes / 60);
    return remainingHours === 1
        ? "About 1 hour left"
        : "About " + remainingHours + " hours left";
}

function libraryFailureTitle(errorCode) {
    const messages = {
        insufficient_disk: "Not enough disk space",
        download_gated: "This model requires Hugging Face access",
        checksum_mismatch: "Downloaded files did not pass verification",
        model_already_present: "Another model uses this Library location. Move it elsewhere, then cancel this download and try again."
    };
    return messages[errorCode] || "Download failed. You can resume it.";
}

function libraryActionButtons(catalogRow) {
    if (catalogRow.readyOnThisMac) return [];
    const isCurrentDownload = libraryCurrentDownload.huggingface_id === catalogRow.huggingfaceId;
    const state = libraryRowDownloadState(catalogRow);
    if (state === "verifying" || state === "publishing") return [];
    if (isCurrentDownload && state === "failed"
        && libraryCurrentDownload.error_code === "model_already_present") {
        return [libraryActionButton("Cancel", "cancel")];
    }
    if (state === "paused" || state === "failed") {
        return [libraryActionButton("Resume", "resume"), libraryActionButton("Cancel", "cancel")];
    }
    if (isCurrentDownload && state !== "idle") {
        return [libraryActionButton("Pause", "pause"), libraryActionButton("Cancel", "cancel")];
    }
    const downloadButton = libraryActionButton("Download", "start", catalogRow.huggingfaceId);
    downloadButton.disabled = libraryCurrentDownload.state !== "idle";
    return [downloadButton];
}

function libraryActionButton(label, action, huggingfaceId = null) {
    const button = document.createElement("button");
    button.className = action === "cancel" ? "button button-danger" : "button button-primary";
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", () => performLibraryAction(action, huggingfaceId));
    return button;
}

async function performLibraryAction(action, huggingfaceId) {
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
        libraryCurrentDownload = actionDocument;
    } catch {
        libraryCurrentDownload = { state: "failed", error_code: "download_failed" };
    }
    renderLibraryCatalogDocument(libraryCatalogDocument);
}

function refreshLibraryDownloadState() {
    if (libraryRefreshPromise !== null) return libraryRefreshPromise;
    libraryRefreshPromise = refreshLibraryDownloadStateOnce()
        .finally(() => { libraryRefreshPromise = null; });
    return libraryRefreshPromise;
}

async function refreshLibraryDownloadStateOnce() {
    try {
        if (!libraryCatalogDocument) {
            const catalogContainer = document.getElementById("library-catalog");
            if (!catalogContainer) return;
            if (libraryCatalogLoadPromise !== null) {
                await libraryCatalogLoadPromise;
            }
            if (libraryCatalogDocument) {
                renderLibraryCatalogDocument(libraryCatalogDocument, catalogContainer);
            } else {
                await loadLibraryCatalog(catalogContainer);
            }
            if (!libraryCatalogDocument) return;
        }
        const downloadResponse = await fetch(LIBRARY_DOWNLOAD_URL);
        if (!downloadResponse.ok) return;
        libraryCurrentDownload = await downloadResponse.json();
        recordLibraryProgressMeasurement(libraryCurrentDownload);
        if (libraryCurrentDownload.state === "idle"
            || libraryCurrentDownload.state === "verifying"
            || libraryCurrentDownload.state === "publishing") {
            await refetchLibraryCatalogDocument();
        }
        renderLibraryCatalogDocument(libraryCatalogDocument);
    } catch {
        // Polling is self-healing and must never cancel daemon-owned work.
    }
}

// The ceiling is re-read while still unknown (the worker reports it after
// startup) and otherwise at most every LIBRARY_CEILING_REFRESH_MILLIS, so a
// memory limit changed on the Configuration page reaches the fit labels within
// seconds without doubling every request of the idle 1s catalog poll.
async function refetchLibraryCatalogDocument(nowMillis = Date.now()) {
    const catalogResponse = await fetch(LIBRARY_CATALOG_URL);
    if (!catalogResponse.ok) return;
    libraryCatalogDocument = await catalogResponse.json();
    const isCeilingStale = nowMillis - libraryServingCeilingReadAtMillis
        >= LIBRARY_CEILING_REFRESH_MILLIS;
    if (libraryServingCeilingBytes === null || isCeilingStale) {
        await refreshLibraryServingCeiling();
        libraryServingCeilingReadAtMillis = nowMillis;
    }
}

// The page names the model, never a path: the daemon resolves the folder from
// its own discovery and download state and asks Finder to select it. Returns
// whether Finder accepted the request.
async function revealLibraryModelInFinder(huggingfaceId) {
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

function renderLibraryEmptyState(catalogContainer) {
    return renderLibraryStatusState(
        catalogContainer,
        "empty",
        "No catalog entries are available in this release."
    );
}

function renderLibraryUnavailableState(catalogContainer) {
    return renderLibraryStatusState(
        catalogContainer,
        "unavailable",
        "The model catalog is unavailable."
    );
}

function renderLibraryStatusState(catalogContainer, libraryState, message) {
    const statusMessage = resolveLibraryStatusMessage();
    statusMessage.className = "library-catalog-state";
    catalogContainer.dataset.libraryState = libraryState;
    // Preserve one live-region node so assistive technology observes state
    // transitions reliably.
    catalogContainer.replaceChildren(statusMessage);
    statusMessage.textContent = message;
    return libraryState;
}

function resolveLibraryStatusMessage() {
    const existingStatusMessage = document.getElementById("library-catalog-status");
    if (existingStatusMessage) {
        return existingStatusMessage;
    }
    const statusMessage = document.createElement("p");
    statusMessage.id = "library-catalog-status";
    statusMessage.setAttribute("role", "status");
    statusMessage.setAttribute("aria-live", "polite");
    return statusMessage;
}
