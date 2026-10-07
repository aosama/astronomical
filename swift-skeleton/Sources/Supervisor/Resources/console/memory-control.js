const MAXIMUM_MLX_MEMORY_URL = "/v1/config/maximum-mlx-memory";
const DECIMAL_GIGABYTE_BYTES = 1e9;
let maximumMlxMemoryEditIsActive = false;

function wholeDecimalGigabyteBounds(statusDocument) {
    const minimumBytes = statusDocument.minimum_mlx_memory_ceiling_bytes || 0;
    const machineBytes = statusDocument.machine_mlx_memory_ceiling_bytes || 0;
    return {
        minimumGigabytes: Math.ceil(minimumBytes / DECIMAL_GIGABYTE_BYTES),
        maximumGigabytes: Math.floor(machineBytes / DECIMAL_GIGABYTE_BYTES)
    };
}

function renderMemoryLimitControl(statusDocument) {
    const input = document.getElementById("maximum-mlx-memory-gb");
    if (!input) { return; }
    const bounds = wholeDecimalGigabyteBounds(statusDocument);
    input.min = String(bounds.minimumGigabytes);
    input.max = String(bounds.maximumGigabytes);
    input.disabled = bounds.minimumGigabytes > bounds.maximumGigabytes;
    if (!maximumMlxMemoryEditIsActive) {
        const configuredGigabytes = statusDocument.configured_maximum_mlx_memory_gb;
        input.value = configuredGigabytes === null || configuredGigabytes === undefined
            ? String(bounds.maximumGigabytes) : String(configuredGigabytes);
    }
    const effectiveGigabytes = formatGigabytes(statusDocument.mlx_memory_ceiling_bytes || 0);
    const machineGigabytes = formatGigabytes(statusDocument.machine_mlx_memory_ceiling_bytes || 0);
    let message = "Effective " + effectiveGigabytes + " of " + machineGigabytes;
    if (statusDocument.pending_mlx_memory_ceiling_bytes) {
        message += " · Pending " + formatGigabytes(statusDocument.pending_mlx_memory_ceiling_bytes);
    }
    if (statusDocument.mlx_memory_limit_error) { message = statusDocument.mlx_memory_limit_error; }
    document.getElementById("maximum-mlx-memory-status").textContent = message;
}

function wireMemoryLimitControl() {
    const input = document.getElementById("maximum-mlx-memory-gb");
    const applyButton = document.getElementById("maximum-mlx-memory-apply");
    const resetButton = document.getElementById("maximum-mlx-memory-reset");
    if (!input || !applyButton || !resetButton) { return; }
    input.addEventListener("input", () => { maximumMlxMemoryEditIsActive = true; });
    input.addEventListener("blur", () => { maximumMlxMemoryEditIsActive = false; });
    applyButton.addEventListener("click", async () => {
        const requestedGigabytes = Number(input.value);
        const bounds = { minimumGigabytes: Number(input.min), maximumGigabytes: Number(input.max) };
        if (!Number.isInteger(requestedGigabytes) || requestedGigabytes < bounds.minimumGigabytes || requestedGigabytes > bounds.maximumGigabytes) {
            showControlFeedback("Choose a whole decimal GB value within the reported bounds.", "error");
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
        if (!response.ok) { showControlFeedback(responseDocument.message || "Memory limit update failed", "error"); return; }
        maximumMlxMemoryEditIsActive = false;
        showControlFeedback(responseDocument.message || "Memory limit updated", response.status === 202 ? "progress" : "success");
        await pollStatus();
    } catch (fetchError) {
        showControlFeedback("Memory limit update failed: " + fetchError.message, "error");
    }
}
