// Astronomical Observatory compact overview panels.
// Renders condensed MLX memory and prompt-cache panels on the Overview view.
// All model output is rendered via textContent only; never innerHTML.
// Reuses formatGigabytes and setMlxSegmentWidth from console.js.

// Pure reconciliation of MLX memory ownership segments against a ceiling.
// Segment order matches the full Memory panel so both renderers share one rule.
// A null snapshot is treated as an all-zero measurement.
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
    const expertPayloadBytes = mlxMemorySnapshot.expert_payload_bytes || 0;
    const modelCorePayloadBytes = mlxMemorySnapshot.model_core_payload_bytes || 0;
    const contextStatePayloadBytes = mlxMemorySnapshot.context_state_payload_bytes || 0;
    const reconciledExpertBytes = Math.min(expertPayloadBytes, activeMemoryBytes);
    const activeAfterExperts = Math.max(0, activeMemoryBytes - reconciledExpertBytes);
    const reconciledModelCoreBytes = Math.min(modelCorePayloadBytes, activeAfterExperts);
    const activeAfterModelCore = Math.max(0, activeAfterExperts - reconciledModelCoreBytes);
    const reconciledContextStateBytes = Math.min(contextStatePayloadBytes, activeAfterModelCore);
    const reconciledRuntimeWorkBytes = Math.max(0, activeAfterModelCore - reconciledContextStateBytes);
    const availableBytes = Math.max(0, mlxMemoryCeilingBytes - activeMemoryBytes);
    return {
        activeMemoryBytes,
        expertBytes: reconciledExpertBytes,
        modelCoreBytes: reconciledModelCoreBytes,
        contextStateBytes: reconciledContextStateBytes,
        runtimeWorkBytes: reconciledRuntimeWorkBytes,
        availableBytes
    };
}

function renderCompactMlxMemory(statusDocument) {
    const mlxMemorySnapshot = statusDocument.mlx_memory_snapshot;
    const mlxMemoryCeilingBytes = statusDocument.mlx_memory_ceiling_bytes || 0;
    const reconciledMemorySegments = reconciledMlxMemorySegmentBytes(
        mlxMemorySnapshot,
        mlxMemoryCeilingBytes
    );
    document.getElementById("compact-memory-total").textContent =
        formatGigabytes(reconciledMemorySegments.activeMemoryBytes);
    document.getElementById("compact-memory-ceiling").textContent =
        "/ " + formatGigabytes(mlxMemoryCeilingBytes);
    setCompactMlxSegmentWidths(
        reconciledMemorySegments.expertBytes,
        reconciledMemorySegments.modelCoreBytes,
        reconciledMemorySegments.contextStateBytes,
        reconciledMemorySegments.runtimeWorkBytes,
        reconciledMemorySegments.availableBytes,
        mlxMemoryCeilingBytes
    );
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

function renderCompactMemoryPressure(memoryPressureState) {
    const pressurePresentation = memoryPressurePresentationForState(memoryPressureState);
    const pressureStateElement = document.getElementById("compact-memory-pressure-state");
    pressureStateElement.textContent = pressurePresentation.title;
    pressureStateElement.dataset.pressureState = pressurePresentation.state;
}

function setCompactMlxSegmentWidths(expertBytes, modelCoreBytes, contextStateBytes, runtimeWorkBytes, availableBytes, ceilingBytes) {
    setMlxSegmentWidth("compact-mem-seg-experts", expertBytes, ceilingBytes);
    setMlxSegmentWidth("compact-mem-seg-model-core", modelCoreBytes, ceilingBytes);
    setMlxSegmentWidth("compact-mem-seg-context-state", contextStateBytes, ceilingBytes);
    setMlxSegmentWidth("compact-mem-seg-runtime-work", runtimeWorkBytes, ceilingBytes);
    setMlxSegmentWidth("compact-mem-seg-available", availableBytes, ceilingBytes);
}

function renderCompactCachePanel(cacheStatsDocument) {
    renderSpeculativePrefillCacheEfficacy(cacheStatsDocument);
    const cacheEfficacyDocument = cacheStatsDocument.speculative_prefill_cache_efficacy || {};
    const combinedCacheEfficacy = cacheEfficacyDocument.combined || {};
    const hitRate = combinedCacheEfficacy.reuse_rate || 0;
    document.getElementById("compact-cache-hit-rate").textContent =
        (hitRate * 100).toFixed(1) + "%";
    const savedPromptTokenCount = combinedCacheEfficacy.restored_token_count || 0;
    document.getElementById("compact-cache-tokens-saved").textContent =
        savedPromptTokenCount.toLocaleString() + " model rows reused";
    const persistentPromptCacheTotalSizeBytes =
        cacheStatsDocument.persistent_prompt_cache_total_size_bytes || 0;
    const persistentPromptCacheMaximumSizeBytes =
        cacheStatsDocument.persistent_prompt_cache_maximum_size_bytes || 0;
    const diskPercent = persistentPromptCacheMaximumSizeBytes > 0
        ? Math.min(100, (persistentPromptCacheTotalSizeBytes / persistentPromptCacheMaximumSizeBytes) * 100)
        : 0;
    document.getElementById("compact-cache-disk-fill").style.width =
        diskPercent.toFixed(1) + "%";
    document.getElementById("compact-cache-disk-label").textContent =
        formatGigabytes(persistentPromptCacheTotalSizeBytes) + " / " +
        formatGigabytes(persistentPromptCacheMaximumSizeBytes);
}

function renderSpeculativePrefillCacheEfficacy(cacheStatsDocument) {
    const cacheEfficacyDocument = cacheStatsDocument.speculative_prefill_cache_efficacy || {};
    renderModelCacheEfficacy("compact-cache-target-efficacy", cacheEfficacyDocument.target);
    renderModelCacheEfficacy("compact-cache-drafter-efficacy", cacheEfficacyDocument.drafter);
}

function renderModelCacheEfficacy(elementIdentifier, modelCacheEfficacy) {
    const boundedModelCacheEfficacy = modelCacheEfficacy || {};
    const reuseRate = boundedModelCacheEfficacy.reuse_rate || 0;
    const restoredTokenCount = boundedModelCacheEfficacy.restored_token_count || 0;
    const eligibleTokenCount = boundedModelCacheEfficacy.eligible_token_count || 0;
    document.getElementById(elementIdentifier).textContent =
        (reuseRate * 100).toFixed(1) + "% · " +
        restoredTokenCount.toLocaleString() + " / " + eligibleTokenCount.toLocaleString();
}
