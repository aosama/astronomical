// Pure reconciliation for the Observatory Experts view.
// No DOM access here: every function turns status and worker stats into
// plain values so node tests cover the sentences and the arithmetic.
// The DOM glue lives in experts.js.
//
// Gigabytes match the rest of Observatory: the gigabyte Apple prints on
// the box, 2^30 bytes. A decimal gigabyte would disagree with the ceiling.

const EXPERT_GIGABYTE_BYTES = 1073741824;

function formatExpertGigabytes(byteCount) {
    if (!Number.isFinite(byteCount) || byteCount <= 0) {
        return "0 GB";
    }
    return (byteCount / EXPERT_GIGABYTE_BYTES).toFixed(2) + " GB";
}

function formatExpertCount(valueCount) {
    if (!Number.isFinite(valueCount) || valueCount <= 0) {
        return "0";
    }
    return Math.round(valueCount).toLocaleString();
}

function formatExpertDuration(totalSeconds) {
    if (!Number.isFinite(totalSeconds) || totalSeconds < 0.05) {
        return "0s";
    }
    if (totalSeconds < 60) {
        return formatExpertSubMinuteDuration(totalSeconds);
    }
    const wholeSeconds = Math.floor(totalSeconds);
    const minutes = Math.floor(wholeSeconds / 60);
    const remainderSeconds = wholeSeconds % 60;
    if (minutes < 60) {
        return remainderSeconds === 0 ? minutes + "m" : minutes + "m " + remainderSeconds + "s";
    }
    const hours = Math.floor(minutes / 60);
    const remainderMinutes = minutes % 60;
    return remainderMinutes === 0 ? hours + "h" : hours + "h " + remainderMinutes + "m";
}

function formatExpertSubMinuteDuration(totalSeconds) {
    const roundedTenths = Math.round(totalSeconds * 10) / 10;
    if (roundedTenths >= 60) {
        return "1m";
    }
    if (Math.abs(roundedTenths - Math.round(roundedTenths)) < 0.001) {
        return Math.round(roundedTenths) + "s";
    }
    return roundedTenths.toFixed(1) + "s";
}

function expertMemoryModePhrase(expertMemoryMode) {
    if (expertMemoryMode === "resident") {
        return "Fully in memory";
    }
    if (expertMemoryMode === "paged") {
        return "RAM + SSD streaming";
    }
    if (expertMemoryMode === "hybrid") {
        return "Hot experts in RAM, the rest on the SSD";
    }
    return "";
}

// The daemon owns the stock and reply sentences. This reader only copies
// them so the menu, About, and this page cannot grow a second phrase table.
function expertPlacementCopy(statusDocument) {
    const placement = statusDocument && statusDocument.expert_placement;
    if (!placement) {
        return { stockSentence: "", replySentence: "" };
    }
    return {
        stockSentence: placement.stock_sentence || "",
        replySentence: placement.reply_sentence || ""
    };
}

function expertPayloadLeftOnSsdBytes(completeExpertPayloadByteCount, residentExpertPayloadBytes) {
    if (!Number.isFinite(completeExpertPayloadByteCount) || completeExpertPayloadByteCount <= 0) {
        return null;
    }
    const residentBytes = Number.isFinite(residentExpertPayloadBytes) && residentExpertPayloadBytes > 0
        ? residentExpertPayloadBytes
        : 0;
    return Math.max(0, completeExpertPayloadByteCount - residentBytes);
}

function expertHeadline(headlineFacts) {
    const facts = headlineFacts || {};
    const modelId = facts.modelId || "";
    if (!modelId) {
        return "Load a sparse model. This page shows which expert weights stay on the SSD.";
    }
    const completeBytes = facts.completeExpertPayloadByteCount || 0;
    const residentBytes = facts.residentExpertPayloadBytes || 0;
    const leftOnSsdBytes = expertPayloadLeftOnSsdBytes(completeBytes, residentBytes);
    if (leftOnSsdBytes !== null && facts.expertMemoryMode === "resident") {
        return modelId + " fits. Its expert library is in RAM. Streaming is idle, and it returns if context or the ceiling takes the room back.";
    }
    if (leftOnSsdBytes === 0) {
        return modelId
            + " has its expert library in RAM. "
            + formatExpertGigabytes(completeBytes)
            + " across "
            + formatExpertCount(facts.totalLayerCount || 0)
            + " layers.";
    }
    if (leftOnSsdBytes !== null) {
        return streamingExpertHeadline(modelId, facts, leftOnSsdBytes);
    }
    return unknownLibraryHeadline(modelId, facts);
}

function streamingExpertHeadline(modelId, headlineFacts, leftOnSsdBytes) {
    const layerCount = headlineFacts.totalLayerCount || 0;
    return modelId
        + " keeps "
        + formatExpertGigabytes(headlineFacts.completeExpertPayloadByteCount)
        + " of expert weights. "
        + formatExpertGigabytes(headlineFacts.residentExpertPayloadBytes || 0)
        + " are in RAM across "
        + formatExpertCount(layerCount)
        + " layers. "
        + formatExpertGigabytes(leftOnSsdBytes)
        + " stayed on the SSD. The layer map shows which layers. The ceiling is "
        + formatExpertGigabytes(headlineFacts.ceilingBytes || 0)
        + ".";
}

function unknownLibraryHeadline(modelId, headlineFacts) {
    const modePhrase = expertMemoryModePhrase(headlineFacts.expertMemoryMode);
    const modeSentence = modePhrase ? " " + modePhrase + "." : "";
    return modelId
        + " is loaded."
        + modeSentence
        + " "
        + formatExpertGigabytes(headlineFacts.residentExpertPayloadBytes || 0)
        + " of expert weights are in RAM. The ceiling is "
        + formatExpertGigabytes(headlineFacts.ceilingBytes || 0)
        + ".";
}

// Same ownership split as reconciledMlxMemorySegmentBytes in overview-compact.js.
// Both bars have to name the same bytes or the expert segment will disagree
// with Overview.
function reconciledExpertMemorySegments(mlxMemorySnapshot, mlxMemoryCeilingBytes) {
    const ceilingBytes = Number.isFinite(mlxMemoryCeilingBytes) && mlxMemoryCeilingBytes > 0
        ? mlxMemoryCeilingBytes
        : 0;
    if (!mlxMemorySnapshot) {
        return {
            expertBytes: 0,
            modelCoreBytes: 0,
            contextStateBytes: 0,
            runtimeWorkBytes: 0,
            availableBytes: ceilingBytes
        };
    }
    const activeMemoryBytes = mlxMemorySnapshot.active_memory_bytes || 0;
    const reconciledExpertBytes = Math.min(mlxMemorySnapshot.expert_payload_bytes || 0, activeMemoryBytes);
    const activeAfterExperts = Math.max(0, activeMemoryBytes - reconciledExpertBytes);
    const reconciledModelCoreBytes = Math.min(
        mlxMemorySnapshot.model_core_payload_bytes || 0,
        activeAfterExperts
    );
    const activeAfterModelCore = Math.max(0, activeAfterExperts - reconciledModelCoreBytes);
    const reconciledContextStateBytes = Math.min(
        mlxMemorySnapshot.context_state_payload_bytes || 0,
        activeAfterModelCore
    );
    const runtimeWorkBytes = Math.max(0, activeAfterModelCore - reconciledContextStateBytes);
    return {
        expertBytes: reconciledExpertBytes,
        modelCoreBytes: reconciledModelCoreBytes,
        contextStateBytes: reconciledContextStateBytes,
        runtimeWorkBytes,
        availableBytes: Math.max(0, ceilingBytes - activeMemoryBytes)
    };
}

function expertBarPercent(byteCount, totalBytes) {
    if (!Number.isFinite(totalBytes) || totalBytes <= 0) {
        return 0;
    }
    if (!Number.isFinite(byteCount) || byteCount <= 0) {
        return 0;
    }
    return Math.min(100, (byteCount / totalBytes) * 100);
}

function expertSessionPiles(valueSection) {
    const value = valueSection || {};
    return {
        promptBytes: value.mandatory_prefill_source_bytes || 0,
        generationBytes: value.mandatory_decode_source_bytes || 0,
        avoidedCompleteLayerBytes: value.avoided_complete_layer_source_bytes || 0
    };
}

function expertReplyIsOpen(activity) {
    return activity === "prompt_processing"
        || activity === "generation_preparation"
        || activity === "generating";
}

function expertSessionOpenSentence(activity) {
    if (expertReplyIsOpen(activity)) {
        return "These totals close when this reply finishes.";
    }
    return "";
}

function expertLayerRows(layerTopology, totalLayerCount, expertCapacity) {
    const topologyByLayerIndex = expertTopologyByLayerIndex(layerTopology);
    const layerCount = Number.isFinite(totalLayerCount) && totalLayerCount > 0
        ? Math.floor(totalLayerCount)
        : 0;
    const capacity = Number.isFinite(expertCapacity) && expertCapacity > 0
        ? Math.floor(expertCapacity)
        : 0;
    // Layer index order is the model, from the first block to the last.
    // Demand order buried the one layer still on the SSD among identical bars.
    const layerRows = [];
    for (let layerIndex = 0; layerIndex < layerCount; layerIndex += 1) {
        layerRows.push(expertLayerRow(
            layerIndex,
            topologyByLayerIndex.get(layerIndex),
            capacity
        ));
    }
    return layerRows;
}

function expertTopologyByLayerIndex(layerTopology) {
    const topologyByLayerIndex = new Map();
    for (const layerEntry of layerTopology || []) {
        if (layerEntry && Number.isFinite(layerEntry.layer_index)) {
            topologyByLayerIndex.set(layerEntry.layer_index, layerEntry);
        }
    }
    return topologyByLayerIndex;
}

function expertLayerRow(layerIndex, layerEntry, expertCapacity) {
    const retainedExpertIds = layerEntry && Array.isArray(layerEntry.retained_expert_ids)
        ? layerEntry.retained_expert_ids
        : [];
    return {
        layerIndex,
        retainedCount: retainedExpertIds.length,
        retainedExpertIds,
        retainedPayloadBytes: layerEntry ? layerEntry.retained_payload_bytes || 0 : 0,
        isCompleteLayer: Boolean(layerEntry && layerEntry.is_complete_layer),
        coveredDemand: layerEntry ? layerEntry.covered_demand || 0 : 0,
        expertCapacity,
        isStreaming: !layerEntry,
        identitiesDeferred: false
    };
}

function expertLayerRowsForObservatory(statusResidency, layerTopology, totalLayerCount, expertCapacity) {
    const retainedExpertCounts = statusResidency && Array.isArray(statusResidency.retained_expert_counts)
        ? statusResidency.retained_expert_counts
        : [];
    if (retainedExpertCounts.length > 0) {
        return expertLayerRowsFromLiveCounts(retainedExpertCounts, layerTopology, expertCapacity);
    }
    return expertLayerRows(layerTopology, totalLayerCount, expertCapacity);
}

// Live counts move the columns during a reply. Expert ids still arrive only
// when the reply finishes, and a mismatched id list would paint warm experts
// as cold, so the slot grid waits.
function expertLayerRowsFromLiveCounts(retainedExpertCounts, layerTopology, expertCapacity) {
    const topologyByLayerIndex = expertTopologyByLayerIndex(layerTopology);
    const capacity = Number.isFinite(expertCapacity) && expertCapacity > 0
        ? Math.floor(expertCapacity)
        : 0;
    const layerRows = [];
    const counts = retainedExpertCounts || [];
    for (let layerIndex = 0; layerIndex < counts.length; layerIndex += 1) {
        layerRows.push(expertLayerRowFromLiveCount(
            layerIndex,
            counts[layerIndex],
            topologyByLayerIndex.get(layerIndex),
            capacity
        ));
    }
    return layerRows;
}

function expertLayerRowFromLiveCount(layerIndex, retainedCount, layerEntry, expertCapacity) {
    const seatedCount = finiteRetainedCount(retainedCount);
    const topologyIds = layerEntry && Array.isArray(layerEntry.retained_expert_ids)
        ? layerEntry.retained_expert_ids
        : [];
    const identitiesMatch = seatedCount > 0 && topologyIds.length === seatedCount;
    return {
        layerIndex,
        retainedCount: seatedCount,
        retainedExpertIds: identitiesMatch ? topologyIds : [],
        retainedPayloadBytes: identitiesMatch && layerEntry ? layerEntry.retained_payload_bytes || 0 : 0,
        isCompleteLayer: expertCapacity > 0 && seatedCount >= expertCapacity,
        coveredDemand: identitiesMatch && layerEntry ? layerEntry.covered_demand || 0 : 0,
        expertCapacity,
        isStreaming: seatedCount <= 0,
        identitiesDeferred: seatedCount > 0 && !identitiesMatch
    };
}

function finiteRetainedCount(retainedCount) {
    if (!Number.isFinite(retainedCount) || retainedCount <= 0) {
        return 0;
    }
    return Math.floor(retainedCount);
}

function expertLayerIdentitiesNote(layerRow) {
    if (!layerRow || !layerRow.identitiesDeferred) {
        return "";
    }
    return " Individual experts for this layer appear when the reply finishes.";
}

function expertLayerCaption(layerRow) {
    return expertLayerDetailSentence(layerRow) + expertLayerIdentitiesNote(layerRow);
}

function expertLayerResidence(layerRow) {
    if (!layerRow || layerRow.expertCapacity <= 0) {
        return layerRow && layerRow.retainedCount > 0 ? "ram" : "ssd";
    }
    if (layerRow.isStreaming || layerRow.retainedCount <= 0) {
        return "ssd";
    }
    if (layerRow.retainedCount >= layerRow.expertCapacity) {
        return "ram";
    }
    return "split";
}

function expertLayerOnSsdCount(layerRow) {
    if (!layerRow || layerRow.expertCapacity <= 0) {
        return 0;
    }
    return Math.max(0, layerRow.expertCapacity - layerRow.retainedCount);
}

function expertLayerOverview(layerRows) {
    const rows = layerRows || [];
    if (rows.length === 0) {
        return "Layers appear here once a sparse model is loaded.";
    }
    const inRamCount = countLayerResidence(rows, "ram");
    const onSsdCount = countLayerResidence(rows, "ssd");
    const splitRows = layerRowsWithResidence(rows, "split");
    const ssdRows = layerRowsWithResidence(rows, "ssd");
    if (onSsdCount === 0 && splitRows.length === 0) {
        return "All " + formatExpertCount(rows.length) + " layers are in RAM.";
    }
    if (inRamCount === 0 && splitRows.length === 0) {
        return "All " + formatExpertCount(rows.length) + " layers are still on the SSD.";
    }
    return expertMixedLayerOverview(rows.length, inRamCount, splitRows, ssdRows);
}

function countLayerResidence(layerRows, residence) {
    let residenceCount = 0;
    for (const layerRow of layerRows) {
        if (expertLayerResidence(layerRow) === residence) {
            residenceCount += 1;
        }
    }
    return residenceCount;
}

function layerRowsWithResidence(layerRows, residence) {
    const matchingRows = [];
    for (const layerRow of layerRows) {
        if (expertLayerResidence(layerRow) === residence) {
            matchingRows.push(layerRow);
        }
    }
    return matchingRows;
}

function expertMixedLayerOverview(layerCount, inRamCount, splitRows, ssdRows) {
    const sentences = [];
    if (inRamCount > 0) {
        sentences.push(
            formatExpertCount(inRamCount)
                + " of "
                + formatExpertCount(layerCount)
                + " layers "
                + (inRamCount === 1 ? "is" : "are")
                + " entirely in RAM."
        );
    }
    appendSplitLayerSentence(sentences, splitRows);
    appendSsdLayerSentence(sentences, ssdRows);
    return sentences.join(" ");
}

function appendSplitLayerSentence(sentences, splitRows) {
    if (splitRows.length === 1) {
        const splitRow = splitRows[0];
        sentences.push(
            "Layer "
                + splitRow.layerIndex
                + " still has "
                + formatExpertCount(expertLayerOnSsdCount(splitRow))
                + " of "
                + formatExpertCount(splitRow.expertCapacity)
                + " experts on the SSD."
        );
        return;
    }
    if (splitRows.length > 1) {
        sentences.push(
            formatExpertCount(splitRows.length)
                + " layers are split between RAM and the SSD."
        );
    }
}

function appendSsdLayerSentence(sentences, ssdRows) {
    if (ssdRows.length === 1) {
        sentences.push("Layer " + ssdRows[0].layerIndex + " is still entirely on the SSD.");
        return;
    }
    if (ssdRows.length > 1 && ssdRows.length <= 4) {
        const layerNames = [];
        for (const ssdRow of ssdRows) {
            layerNames.push("layer " + ssdRow.layerIndex);
        }
        sentences.push("Still entirely on the SSD: " + layerNames.join(", ") + ".");
        return;
    }
    if (ssdRows.length > 4) {
        sentences.push(
            formatExpertCount(ssdRows.length) + " layers are still entirely on the SSD."
        );
    }
}

function expertLayerDetailSentence(layerRow) {
    if (!layerRow) {
        return "";
    }
    const residence = expertLayerResidence(layerRow);
    if (residence === "ram") {
        return "Layer "
            + layerRow.layerIndex
            + " · all "
            + formatExpertCount(layerRow.expertCapacity || layerRow.retainedCount)
            + " experts in RAM";
    }
    if (residence === "ssd") {
        if (layerRow.expertCapacity > 0) {
            return "Layer "
                + layerRow.layerIndex
                + " · all "
                + formatExpertCount(layerRow.expertCapacity)
                + " experts on the SSD";
        }
        return "Layer " + layerRow.layerIndex + " · on the SSD";
    }
    return "Layer "
        + layerRow.layerIndex
        + " · "
        + formatExpertCount(layerRow.retainedCount)
        + " in RAM · "
        + formatExpertCount(expertLayerOnSsdCount(layerRow))
        + " on the SSD";
}

function expertLayerFillPercent(retainedCount, expertCapacity) {
    return expertBarPercent(retainedCount, expertCapacity);
}

function expertSlotStates(retainedExpertIds, expertCapacity) {
    const warmExpertIds = new Set(retainedExpertIds || []);
    const capacity = Number.isFinite(expertCapacity) && expertCapacity > 0
        ? Math.floor(expertCapacity)
        : 0;
    const slotStates = [];
    for (let expertId = 0; expertId < capacity; expertId += 1) {
        slotStates.push({
            expertId,
            isWarm: warmExpertIds.has(expertId)
        });
    }
    return slotStates;
}

function expertGenerationSentence(valueSection) {
    const value = valueSection || {};
    const warmTokenCount = value.hot_route_fully_covered_token_count || 0;
    const mixedTokenCount = value.hot_route_partially_covered_token_count || 0;
    const missedTokenCount = value.hot_route_missed_token_count || 0;
    const classifiedTokenCount = warmTokenCount + mixedTokenCount + missedTokenCount;
    if (classifiedTokenCount <= 0) {
        return "No generated tokens have been classified yet.";
    }
    const touchedStorageCount = mixedTokenCount + missedTokenCount;
    return formatExpertCount(warmTokenCount)
        + " generated tokens had every routed expert already warm. "
        + formatExpertCount(touchedStorageCount)
        + " still touched the SSD.";
}

function expertMixedRouteSentence(valueSection) {
    const mixedTokenCount = valueSection
        ? valueSection.hot_route_partially_covered_token_count || 0
        : 0;
    if (mixedTokenCount <= 0) {
        return "";
    }
    return formatExpertCount(mixedTokenCount)
        + " of those tokens were mixed: a partial hit still reads the missing experts.";
}

function modeTransitionLine(modeTransition) {
    if (!modeTransition) {
        return "";
    }
    const observedAtMillis = modeTransition.observed_at_unix_millis || 0;
    const observedAt = new Date(observedAtMillis);
    const timeLabel = isNaN(observedAt.getTime())
        ? "unknown time"
        : observedAt.toLocaleTimeString();
    const previousPhrase = modeTransition.previous_mode
        ? expertMemoryModePhrase(modeTransition.previous_mode)
        : "Start";
    const nextPhrase = expertMemoryModePhrase(modeTransition.next_mode) || "Unknown";
    return timeLabel + " · " + previousPhrase + " → " + nextPhrase;
}

function expertMethodLiveLine(valueSection) {
    if (!valueSection) {
        return "No storage assignment has been timed yet, so no estimate is published.";
    }
    const storageAssignments = valueSection.storage_route_assignment_count || 0;
    const readElapsedNanoseconds = valueSection.expert_source_read_elapsed_nanoseconds || 0;
    if (storageAssignments === 0 || readElapsedNanoseconds === 0) {
        return "No storage assignment has been timed yet, so no estimate is published.";
    }
    const averageMillis = readElapsedNanoseconds / 1_000_000 / storageAssignments;
    const savedSeconds = (valueSection.storage_time_saved_estimate_nanoseconds || 0) / 1_000_000_000;
    return "Average measured storage assignment: "
        + averageMillis.toFixed(1)
        + " ms. Retained hits: "
        + formatExpertCount(valueSection.retained_route_assignment_count || 0)
        + ". Estimated time saved: "
        + formatExpertDuration(savedSeconds)
        + ".";
}
