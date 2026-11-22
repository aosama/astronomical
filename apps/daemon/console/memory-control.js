// Astronomical Observatory Maximum model RAM control, at the top of the
// Settings page. It reuses pollStatus and formatGigabytes from console.js
// and tells the configuration editor about a changed config.json through the
// "observatory:config-file-written" event rather than calling into it.

const MAXIMUM_MLX_MEMORY_URL = "/v1/config/maximum-mlx-memory";
const CONFIG_FILE_WRITTEN_EVENT = "observatory:config-file-written";
const MARKETED_GIGABYTE_BYTES = 1073741824;
// Mirrors METAL_WORKING_SET_MARGIN_PARTS_PER_THOUSAND in the serving ceiling:
// the engine plans 2.5% below any wired claim, so the control must too or it
// offers a top value the engine quietly cuts.
const METAL_WORKING_SET_MARGIN_PARTS_PER_THOUSAND = 25;
let maximumMlxMemoryEditIsActive = false;

function mlxMemoryCeilingTheEngineWillInstall(requestedCeilingBytes, wiredResidencyBudgetBytes) {
    if (wiredResidencyBudgetBytes === null || wiredResidencyBudgetBytes === undefined) {
        return requestedCeilingBytes;
    }
    const metalWorkingSetMarginBytes =
        Math.floor(wiredResidencyBudgetBytes / 1000) * METAL_WORKING_SET_MARGIN_PARTS_PER_THOUSAND;
    const plannableWiredResidencyBytes = Math.max(0, wiredResidencyBudgetBytes - metalWorkingSetMarginBytes);
    return Math.min(requestedCeilingBytes, plannableWiredResidencyBytes);
}

function wholeGigabyteBounds(statusDocument) {
    const machineBytes = statusDocument.machine_mlx_memory_ceiling_bytes || 0;
    const reachableBytes = mlxMemoryCeilingTheEngineWillInstall(
        machineBytes, statusDocument.wired_residency_budget_bytes);
    return {
        minimumGigabytes: 1,
        maximumGigabytes: Math.floor(reachableBytes / MARKETED_GIGABYTE_BYTES)
    };
}

// The value the input rests on while nobody edits it; never above the
// reachable maximum, even when the configuration file asks for more.
function appliedMaximumGigabytes(statusDocument, bounds) {
    const configuredGigabytes = statusDocument.configured_maximum_mlx_memory_gb;
    if (configuredGigabytes === null || configuredGigabytes === undefined) {
        return bounds.maximumGigabytes;
    }
    return Math.min(configuredGigabytes, bounds.maximumGigabytes);
}

// A change waiting for the reply being written to finish. It names the limit
// the engine will really install, never the raw request: "Use Max" asks for
// the whole Mac's memory, which the engine lowers to what macOS lets it keep.
function queuedMemoryLimitNotice(statusDocument) {
    const pendingBytes = statusDocument.pending_mlx_memory_ceiling_bytes;
    if (!pendingBytes) { return ""; }
    const installedBytes = mlxMemoryCeilingTheEngineWillInstall(
        pendingBytes, statusDocument.wired_residency_budget_bytes);
    return "Changes to " + formatGigabytes(installedBytes) + " when the current reply finishes.";
}

// Something worth saying: an error, a queued change, or an explicit request
// the engine serves below. "Use Max" landing under the machine ceiling is the
// request being honoured, so it never counts as a cap.
function memoryLimitNotice(statusDocument) {
    if (statusDocument.mlx_memory_limit_error) { return statusDocument.mlx_memory_limit_error; }
    if (statusDocument.pending_mlx_memory_ceiling_bytes) {
        return queuedMemoryLimitNotice(statusDocument);
    }
    const configuredGigabytes = statusDocument.configured_maximum_mlx_memory_gb;
    const liveCeilingBytes = statusDocument.mlx_memory_ceiling_bytes || 0;
    if (configuredGigabytes === null || configuredGigabytes === undefined || liveCeilingBytes === 0) {
        return "";
    }
    if (configuredGigabytes * MARKETED_GIGABYTE_BYTES > liveCeilingBytes) {
        return "Capped at " + formatGigabytes(liveCeilingBytes) + " on this Mac";
    }
    return "";
}

function renderMemoryLimitButtons(statusDocument) {
    const input = document.getElementById("maximum-mlx-memory-gb");
    const applyButton = document.getElementById("maximum-mlx-memory-apply");
    const resetButton = document.getElementById("maximum-mlx-memory-reset");
    const bounds = wholeGigabyteBounds(statusDocument);
    const configuredGigabytes = statusDocument.configured_maximum_mlx_memory_gb;
    const configurationIsExplicit = configuredGigabytes !== null && configuredGigabytes !== undefined;
    // A configuration above the reachable maximum also earns Apply: the input
    // already rests on the reachable value, so one click replaces it.
    const configurationIsUnreachable = configurationIsExplicit && configuredGigabytes > bounds.maximumGigabytes;
    applyButton.hidden = Number(input.value) === appliedMaximumGigabytes(statusDocument, bounds)
        && !configurationIsUnreachable;
    resetButton.hidden = !configurationIsExplicit;
}

let lastRenderedMemoryStatusDocument = null;

function renderMemoryLimitControl(statusDocument) {
    const input = document.getElementById("maximum-mlx-memory-gb");
    if (!input) { return; }
    lastRenderedMemoryStatusDocument = statusDocument;
    const bounds = wholeGigabyteBounds(statusDocument);
    input.min = String(bounds.minimumGigabytes);
    input.max = String(bounds.maximumGigabytes);
    input.disabled = bounds.minimumGigabytes > bounds.maximumGigabytes;
    if (!maximumMlxMemoryEditIsActive) {
        input.value = String(appliedMaximumGigabytes(statusDocument, bounds));
    }
    renderMemoryLimitButtons(statusDocument);
    document.getElementById("maximum-mlx-memory-status").textContent =
        "Up to " + bounds.maximumGigabytes + " GB on this Mac";
    const noticeElement = document.getElementById("maximum-mlx-memory-notice");
    noticeElement.textContent = memoryLimitNotice(statusDocument);
    // A queued change is the control working as designed, so it drops the
    // warning colour an error or a cap keeps.
    const isOnlyQueued = !statusDocument.mlx_memory_limit_error
        && Boolean(statusDocument.pending_mlx_memory_ceiling_bytes);
    noticeElement.className = isOnlyQueued
        ? "mlx-memory-control-notice mlx-memory-control-notice-calm"
        : "mlx-memory-control-notice";
    noticeElement.hidden = noticeElement.textContent === "";
}

// Feedback stays inside the control's own panel, next to the button pressed.
// An empty message hides it, as for a change the notice already names.
function showMemoryControlFeedback(message, feedbackKind) {
    const feedbackElement = document.getElementById("maximum-mlx-memory-feedback");
    feedbackElement.textContent = message;
    feedbackElement.className = "control-feedback feedback-" + feedbackKind;
    feedbackElement.hidden = message === "";
}

function wireMemoryLimitControl() {
    const input = document.getElementById("maximum-mlx-memory-gb");
    const applyButton = document.getElementById("maximum-mlx-memory-apply");
    const resetButton = document.getElementById("maximum-mlx-memory-reset");
    if (!input || !applyButton || !resetButton) { return; }
    input.addEventListener("input", () => {
        maximumMlxMemoryEditIsActive = true;
        if (lastRenderedMemoryStatusDocument) { renderMemoryLimitButtons(lastRenderedMemoryStatusDocument); }
    });
    input.addEventListener("blur", () => { maximumMlxMemoryEditIsActive = false; });
    applyButton.addEventListener("click", async () => {
        const requestedGigabytes = Number(input.value);
        const bounds = { minimumGigabytes: Number(input.min), maximumGigabytes: Number(input.max) };
        if (!Number.isInteger(requestedGigabytes) || requestedGigabytes < bounds.minimumGigabytes || requestedGigabytes > bounds.maximumGigabytes) {
            showMemoryControlFeedback("Choose a whole GB value within the reported bounds.", "error");
            return;
        }
        await updateMaximumMlxMemory(requestedGigabytes);
    });
    resetButton.addEventListener("click", async () => { await updateMaximumMlxMemory(null); });
}

async function updateMaximumMlxMemory(maximumMlxMemoryGigabytes) {
    try {
        const response = await fetch(MAXIMUM_MLX_MEMORY_URL, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ maximum_mlx_memory_gb: maximumMlxMemoryGigabytes })
        });
        const responseDocument = await response.json();
        // The daemon can write config.json even when it answers with an error (a
        // refused ceiling keeps the persisted intent), so the editor re-reads the
        // file after every answered request; a request that never answered skips it.
        document.dispatchEvent(new CustomEvent(CONFIG_FILE_WRITTEN_EVENT));
        if (!response.ok) { showMemoryControlFeedback(responseDocument.message || "Memory limit update failed", "error"); return; }
        maximumMlxMemoryEditIsActive = false;
        // A change that waits for the current reply is already named, quietly,
        // by the notice under the input, so it gets no second, coloured message.
        if (response.status === 202) {
            showMemoryControlFeedback("", "success");
        } else {
            showMemoryControlFeedback(responseDocument.message || "Maximum model RAM updated.", "success");
        }
        await pollStatus();
    } catch (fetchError) {
        showMemoryControlFeedback("Memory limit update failed: " + fetchError.message, "error");
    }
}
