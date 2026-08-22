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

let libraryCatalogLoadPromise = null;
let libraryCatalogDocument = null;
let libraryCurrentDownload = { state: "idle" };
let libraryPollingTimer = null;
let libraryRefreshPromise = null;
let libraryProgressSample = null;
let libraryMeasuredBytesPerSecond = null;
// Active filter state: search query, family filter, and readiness filter.
let librarySearchQuery = "";
let libraryFamilyFilter = "all";
let libraryReadinessFilter = "all";
let libraryExpandedModelId = null;

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
    const catalogRows = libraryCatalogRowsFromDocument(catalogDocument);
    if (catalogRows === null) {
        return renderLibraryUnavailableState(resolvedCatalogContainer);
    }
    if (catalogRows.length === 0) {
        return renderLibraryEmptyState(resolvedCatalogContainer);
    }

    const filteredRows = filterLibraryRows(catalogRows);
    const statusMessage = resolveLibraryStatusMessage();
    statusMessage.className = "library-catalog-state library-catalog-status-only";
    statusMessage.textContent = "Model catalog loaded.";
    resolvedCatalogContainer.dataset.libraryState = "ready";

    const heroSummary = createLibraryHeroSummary(catalogRows);
    const filterBar = createLibraryFilterBar(catalogRows);
    const catalogList = document.createElement("div");
    catalogList.className = "library-catalog-list";
    catalogList.replaceChildren(
        ...filteredRows.map((catalogRow) =>
            createLibraryCatalogRow(catalogRow, catalogRows)
        )
    );
    if (filteredRows.length === 0 && catalogRows.length > 0) {
        const noMatches = document.createElement("p");
        noMatches.className = "library-no-matches";
        noMatches.textContent = "No models match the current filter.";
        catalogList.replaceChildren(noMatches);
    }
    resolvedCatalogContainer.replaceChildren(
        statusMessage,
        heroSummary,
        filterBar,
        catalogList
    );
    return "ready";
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
                catalogRow.quantizationLabel || "",
                catalogRow.architectureSummary || ""
            ].join(" ").toLowerCase();
            if (!haystack.includes(librarySearchQuery)) return false;
        }
        return true;
    });
}

function libraryCatalogRowsFromDocument(catalogDocument) {
    if (!catalogDocument
        || catalogDocument.schema_version !== 1
        || !Array.isArray(catalogDocument.entries)) {
        return null;
    }
    const catalogRows = [];
    for (const catalogEntry of catalogDocument.entries) {
        if (!isRenderableLibraryCatalogEntry(catalogEntry)) {
            return null;
        }
        const capabilities = catalogEntry.capabilities || {};
        catalogRows.push({
            huggingfaceId: catalogEntry.huggingface_id,
            displayName: catalogEntry.display_name,
            family: catalogEntry.family,
            approximateSize: formatLibrarySizeGigabytes(catalogEntry.approximate_size_bytes),
            approximateSizeBytes: catalogEntry.approximate_size_bytes,
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
            architectureSummary: typeof catalogEntry.architecture_summary === "string"
                ? catalogEntry.architecture_summary
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
            contextWindow: typeof capabilities.context_window === "number"
                ? capabilities.context_window
                : null,
            maxOutputTokens: typeof capabilities.max_output_tokens === "number"
                ? capabilities.max_output_tokens
                : null
        });
    }
    return catalogRows;
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
        && catalogEntry.public === true;
}

function formatLibrarySizeGigabytes(approximateSizeBytes) {
    return (approximateSizeBytes / 1_000_000_000).toFixed(2) + " GB";
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

function libraryStateTitle(catalogRow) {
    if (catalogRow.readyOnThisMac) return "Ready to use";
    const activeDownload = libraryCurrentDownload.huggingface_id === catalogRow.huggingfaceId
        ? libraryCurrentDownload : null;
    const downloadState = activeDownload?.state || catalogRow.downloadState;
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
    if (bytesPerSecond >= 1_000_000_000) {
        return (bytesPerSecond / 1_000_000_000).toFixed(1) + " GB/s";
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
        model_already_present: "A model already exists at this destination"
    };
    return messages[errorCode] || "Download failed. You can resume it.";
}

function libraryActionButtons(catalogRow) {
    if (catalogRow.readyOnThisMac) return [];
    const isCurrentDownload = libraryCurrentDownload.huggingface_id === catalogRow.huggingfaceId;
    const state = isCurrentDownload ? libraryCurrentDownload.state : catalogRow.downloadState;
    if (state === "verifying" || state === "publishing") return [];
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
            const catalogResponse = await fetch(LIBRARY_CATALOG_URL);
            if (catalogResponse.ok) libraryCatalogDocument = await catalogResponse.json();
        }
        renderLibraryCatalogDocument(libraryCatalogDocument);
    } catch {
        // Polling is self-healing and must never cancel daemon-owned work.
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
