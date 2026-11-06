// Owns what the Overview says about the Neural Engine, for someone who turned it on and
// wants to know whether it is helping, whether it is worth its memory, and why it is not
// running when it is not. Someone who left it off never sees any of this.

const BYTES_PER_GIGABYTE = 1073741824;
const MILLISECONDS_PER_SECOND = 1000;
const SECONDS_PER_MINUTE = 60;
const MINUTES_PER_HOUR = 60;

// Always "about", because a chunk cannot run both ways at once, so this is never measured.
function formatEstimatedSavedTime(savedMilliseconds) {
    if (savedMilliseconds < MILLISECONDS_PER_SECOND) { return "under 1 s"; }
    const totalSeconds = Math.round(savedMilliseconds / MILLISECONDS_PER_SECOND);
    if (totalSeconds < SECONDS_PER_MINUTE) { return `about ${totalSeconds} s`; }
    const totalMinutes = Math.floor(totalSeconds / SECONDS_PER_MINUTE);
    if (totalMinutes < MINUTES_PER_HOUR) {
        return `about ${totalMinutes} min ${totalSeconds % SECONDS_PER_MINUTE} s`;
    }
    return `about ${Math.floor(totalMinutes / MINUTES_PER_HOUR)} h ${totalMinutes % MINUTES_PER_HOUR} min`;
}

function neuralEngineSpeedTile(prefill) {
    if (!prefill.measured_speed_percent) {
        return { value: "Measuring", detail: "finishes after a few long prompts" };
    }
    return {
        value: `${(prefill.measured_speed_percent / 100).toFixed(2)}×`,
        detail: "vs GPU alone, this model",
    };
}

function neuralEngineWorkTile(totals) {
    const promptTokenCount = totals.prompt_token_count || 0;
    if (promptTokenCount === 0) { return { value: "—", detail: "only long prompts use it" }; }
    const neuralEngineTokenCount = totals.neural_engine_token_count || 0;
    return {
        value: `${Math.round((neuralEngineTokenCount / promptTokenCount) * 100)}%`,
        detail: `${neuralEngineTokenCount.toLocaleString()} of ${promptTokenCount.toLocaleString()} prompt tokens`,
    };
}

function neuralEngineSavedTile(prefill, totals) {
    const savedMilliseconds = totals.estimated_saved_ms || 0;
    if (!prefill.measured_speed_percent && savedMilliseconds === 0) {
        return { value: "—", detail: "known once speed is measured" };
    }
    return {
        value: savedMilliseconds === 0 ? "None yet" : formatEstimatedSavedTime(savedMilliseconds),
        detail: "estimated, warm prompts only",
    };
}

function neuralEngineActiveNote(statusDocument, prefill, totals) {
    const memoryGigabytes = Math.ceil((prefill.measured_memory_bytes || 0) / BYTES_PER_GIGABYTE);
    if (statusDocument.progress && totals.last_chunk_placement === "gpu_memory_tight") {
        const ceilingGigabytes = Math.round((statusDocument.mlx_memory_ceiling_bytes || 0) / BYTES_PER_GIGABYTE);
        return `This long prompt moved to the GPU because the programs' ${memoryGigabytes} GB wouldn't fit under `
            + `your ${ceilingGigabytes} GB memory ceiling beside it. A higher ceiling keeps it on both engines.`;
    }
    if (!prefill.measured_speed_percent) {
        return "Long prompts take turns between the GPU alone and both engines, "
            + "which costs them a few percent, once per model.";
    }
    const sharePercent = Math.round((prefill.neural_engine_tokens_per_chunk / prefill.chunk_token_count) * 100);
    return `Using about ${memoryGigabytes} GB of memory. It takes up to ${sharePercent}% of each long prompt, `
        + "and the GPU does the rest.";
}

function neuralEngineSettingUpNote(prefill) {
    if (prefill.state === "unavailable") {
        const reason = prefill.reason || "it can't run on this model";
        return `Not in use: ${reason}.`;
    }
    if (prefill.is_waiting_to_measure) {
        return `${prefill.total_program_count} programs built. Their memory is measured at the next pause `
            + "in requests, and your model serves on the GPU until then.";
    }
    return `Preparing programs, ${prefill.completed_program_count} of ${prefill.total_program_count}. `
        + "Your model is already serving on the GPU.";
}

// What the panel shows, apart from the page so every state can be tested without one.
function neuralEnginePanelContent(statusDocument) {
    const prefill = statusDocument.neural_engine_prefill;
    if (!prefill || prefill.state === "off") { return { isVisible: false }; }
    if (prefill.state !== "active") {
        return { isVisible: true, tiles: null, note: neuralEngineSettingUpNote(prefill) };
    }
    const totals = (statusDocument.serving_session && statusDocument.serving_session.neural_engine) || {};
    return {
        isVisible: true,
        tiles: {
            speed: neuralEngineSpeedTile(prefill),
            work: neuralEngineWorkTile(totals),
            saved: neuralEngineSavedTile(prefill, totals),
        },
        note: neuralEngineActiveNote(statusDocument, prefill, totals),
    };
}

// While the engines are being compared they alternate chunk by chunk, so this holds one
// label for the whole comparison instead of flickering between two.
function neuralEnginePhaseSuffix(statusDocument) {
    const prefill = statusDocument.neural_engine_prefill;
    if (!prefill || prefill.state === "off") { return ""; }
    if (prefill.state !== "active") { return " · GPU"; }
    if (!prefill.measured_speed_percent) { return " · comparing engines"; }
    const totals = (statusDocument.serving_session && statusDocument.serving_session.neural_engine) || {};
    switch (totals.last_chunk_placement) {
        case "both_engines": return " · both engines";
        case "gpu_memory_tight": return " · GPU, memory is tight";
        case undefined:
        case null: return "";
        default: return " · GPU";
    }
}

function renderNeuralEngineTile(tileName, tile) {
    document.getElementById(`neural-engine-${tileName}`).textContent = tile.value;
    document.getElementById(`neural-engine-${tileName}-detail`).textContent = tile.detail;
}

function renderNeuralEnginePanel(statusDocument) {
    const panel = document.getElementById("neural-engine-panel");
    if (!panel) { return; }
    const content = neuralEnginePanelContent(statusDocument);
    panel.hidden = !content.isVisible;
    if (!content.isVisible) { return; }
    document.getElementById("neural-engine-tiles").hidden = !content.tiles;
    document.getElementById("neural-engine-note").textContent = content.note;
    if (content.tiles) {
        renderNeuralEngineTile("speed", content.tiles.speed);
        renderNeuralEngineTile("work", content.tiles.work);
        renderNeuralEngineTile("saved", content.tiles.saved);
    }
}
