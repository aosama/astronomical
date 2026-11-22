// Astronomical Observatory Models page: which models to suggest to this Mac,
// and how the builds of one model fold under their lead.
// Pure functions over catalog rows; nothing here touches the document.
//
// A suggestion is relative to what the person already has: once a model is
// here, the useful question is what would be better on this Mac, not which of
// twenty rows to read. The ladder of models the page climbs is the catalog's
// Get started picks for chat and coding, the ones the release authors chose to
// steer people to, ordered by the memory they need.

// What each kind of build changes, in words the row shows beside its summary.
const DOWNLOAD_MODELS_BUILD_DIFFERENCE_TEXT = {
    more_detail: "More detail kept",
    draft_model: "Draft model built in",
    streaming_layout: "Laid out for streaming from the SSD"
};
// A step up to a build with more detail is suggested only when it leaves a
// tenth of the model memory free: a build that just squeezes in leaves no room
// for a long conversation, and the page would be suggesting a slowdown.
const DOWNLOAD_MODELS_ROOMY_FIT_FRACTION = 0.9;
const DOWNLOAD_MODELS_MAXIMUM_SUGGESTION_COUNT = 3;

function isDownloadModelsChatSectionRow(catalogRow) {
    return catalogRow.role === "chat" || catalogRow.role === "coding";
}

// The ladder of models to suggest, smallest first.
function downloadModelsSuggestionLadder(catalogRows) {
    return catalogRows
        .filter((catalogRow) => catalogRow.getStarted && isDownloadModelsChatSectionRow(catalogRow)
            && !catalogRow.experimental && !catalogRow.retiring)
        .sort((smallerRow, largerRow) => smallerRow.approximateMemoryBytes - largerRow.approximateMemoryBytes);
}

function downloadModelsLeadModelId(catalogRow) {
    return catalogRow.buildOf || catalogRow.modelId;
}

// A model the person already has, or has asked for: on this Mac, downloading,
// or waiting in the download list. Suggesting it again would be noise.
function isDownloadModelsRowClaimed(catalogRow) {
    if (catalogRow.readyOnThisMac) return true;
    if (downloadModelsWaitingPosition(catalogRow.huggingfaceId) !== null) return true;
    const downloadState = downloadModelsRowDownloadState(catalogRow);
    return Boolean(downloadState) && downloadState !== "idle";
}

// The lead model IDs the person has any build of.
function downloadModelsClaimedLeadIds(catalogRows) {
    return new Set(catalogRows.filter(isDownloadModelsRowClaimed).map(downloadModelsLeadModelId));
}

function isDownloadModelsRowInMemory(catalogRow, servingCeilingBytes) {
    return catalogRow.approximateMemoryBytes <= servingCeilingBytes;
}

function isDownloadModelsRowStreamable(catalogRow, servingCeilingBytes) {
    return catalogRow.minimumMemoryBytes !== null && catalogRow.minimumMemoryBytes <= servingCeilingBytes;
}

// The suggestion cards for a Mac with this memory, at most three, most useful
// first. Each is { kind, tag, reason, catalogRow }. With the ceiling unknown the
// page suggests nothing rather than guess.
function downloadModelsSuggestions(catalogRows, servingCeilingBytes) {
    if (!(servingCeilingBytes > 0)) return [];
    const ladder = downloadModelsSuggestionLadder(catalogRows);
    const claimedLeadIds = downloadModelsClaimedLeadIds(catalogRows);
    const bestFitRow = downloadModelsBestFitRow(ladder, servingCeilingBytes);
    const suggestions = [
        downloadModelsBestFitSuggestion(ladder, bestFitRow, claimedLeadIds, servingCeilingBytes),
        downloadModelsMoreDetailSuggestion(catalogRows, bestFitRow, servingCeilingBytes),
        downloadModelsStreamingSuggestion(ladder, bestFitRow, claimedLeadIds, servingCeilingBytes),
        downloadModelsPicturesSuggestion(catalogRows)
    ];
    return suggestions.filter((suggestion) => suggestion !== null).slice(0, DOWNLOAD_MODELS_MAXIMUM_SUGGESTION_COUNT);
}

// The largest ladder model that runs entirely in memory; on a Mac where none
// does, the smallest, which streams the least.
function downloadModelsBestFitRow(ladder, servingCeilingBytes) {
    const inMemoryRows = ladder.filter((catalogRow) => isDownloadModelsRowInMemory(catalogRow, servingCeilingBytes));
    if (inMemoryRows.length > 0) return inMemoryRows[inMemoryRows.length - 1];
    return ladder.length > 0 ? ladder[0] : null;
}

function downloadModelsBestFitSuggestion(ladder, bestFitRow, claimedLeadIds, servingCeilingBytes) {
    if (bestFitRow === null || claimedLeadIds.has(bestFitRow.modelId)) return null;
    if (!isDownloadModelsRowInMemory(bestFitRow, servingCeilingBytes)) {
        const slowerReason = isDownloadModelsRowStreamable(bestFitRow, servingCeilingBytes)
            ? "It reads part of itself from the SSD while it answers, so replies are slower."
            : "It needs more memory than this Mac gives to models, so replies are very slow.";
        return {
            kind: "best-fit", tag: "Best fit for this Mac", catalogRow: bestFitRow,
            reason: "The lightest of the suggested models. " + slowerReason
        };
    }
    const smallerClaimedRow = ladder.find(
        (catalogRow) => claimedLeadIds.has(catalogRow.modelId)
            && catalogRow.approximateMemoryBytes < bestFitRow.approximateMemoryBytes
    );
    if (smallerClaimedRow) {
        return {
            kind: "stronger", tag: "A stronger model", catalogRow: bestFitRow,
            reason: "Better at hard questions and code than " + smallerClaimedRow.modelId + ", and it still runs entirely in this Mac's memory."
        };
    }
    return {
        kind: "best-fit", tag: "Best fit for this Mac", catalogRow: bestFitRow,
        reason: "The strongest suggested model that runs entirely in this Mac's memory."
    };
}

// A build of the best-fit model that keeps more of the original detail and
// still leaves room; the largest such build wins. It is offered whether or not
// the person has the best fit yet: a Mac with room for more detail should hear so.
// A build of that model the person already has that is at least as large
// (a 6-bit build with a draft model, say) already keeps that detail, so then
// nothing is offered.
function downloadModelsMoreDetailSuggestion(catalogRows, bestFitRow, servingCeilingBytes) {
    if (bestFitRow === null || !isDownloadModelsRowInMemory(bestFitRow, servingCeilingBytes)) return null;
    const roomyCeilingBytes = servingCeilingBytes * DOWNLOAD_MODELS_ROOMY_FIT_FRACTION;
    const candidateRows = catalogRows.filter(
        (catalogRow) => catalogRow.buildOf === bestFitRow.modelId && catalogRow.buildDifference === "more_detail"
            && catalogRow.approximateMemoryBytes <= roomyCeilingBytes
    );
    if (candidateRows.length === 0) return null;
    const largestCandidateRow = candidateRows.reduce(
        (largestRow, catalogRow) => (catalogRow.approximateMemoryBytes > largestRow.approximateMemoryBytes ? catalogRow : largestRow)
    );
    const hasAsDetailedBuild = catalogRows.some(
        (catalogRow) => downloadModelsLeadModelId(catalogRow) === bestFitRow.modelId
            && isDownloadModelsRowClaimed(catalogRow)
            && catalogRow.approximateMemoryBytes >= largestCandidateRow.approximateMemoryBytes
    );
    if (hasAsDetailedBuild) return null;
    const extraMemoryText = formatDownloadModelsCardSize(
        largestCandidateRow.approximateMemoryBytes - bestFitRow.approximateMemoryBytes
    );
    return {
        kind: "more-detail", tag: "Closer to the original", catalogRow: largestCandidateRow,
        reason: "The same model as " + bestFitRow.modelId + " with more of its original detail kept. It needs " + extraMemoryText + " more memory."
    };
}

// The largest ladder model, when it does not fit in memory but streams: much
// stronger, and the person decides whether slower replies are worth it.
function downloadModelsStreamingSuggestion(ladder, bestFitRow, claimedLeadIds, servingCeilingBytes) {
    if (ladder.length === 0) return null;
    const largestRow = ladder[ladder.length - 1];
    if (largestRow === bestFitRow || claimedLeadIds.has(largestRow.modelId)) return null;
    if (isDownloadModelsRowInMemory(largestRow, servingCeilingBytes)
        || !isDownloadModelsRowStreamable(largestRow, servingCeilingBytes)) {
        return null;
    }
    return {
        kind: "streams", tag: "The strongest that runs here", catalogRow: largestRow,
        reason: "Much stronger, but part of it is read from the SSD while it answers, so replies are slower."
    };
}

// Pictures are a different job, so a Mac with no picture model is told one
// exists; the Get started picture pick is the one offered.
function downloadModelsPicturesSuggestion(catalogRows) {
    const hasPictureModel = catalogRows.some(
        (catalogRow) => catalogRow.role === "image_generation" && isDownloadModelsRowClaimed(catalogRow)
    );
    if (hasPictureModel) return null;
    const picturePickRow = catalogRows.find(
        (catalogRow) => catalogRow.role === "image_generation" && catalogRow.getStarted
    );
    if (!picturePickRow) return null;
    return {
        kind: "pictures", tag: "Make pictures", catalogRow: picturePickRow,
        reason: "Turns a written description into a picture. You have no picture model yet."
    };
}

// The rows of one section in list order with each lead's builds folded under
// it: [{ catalogRow, buildRows }]. A build whose lead is not in the same rows
// (a search can match a build alone) is listed on its own.
function downloadModelsFoldedRows(sectionRows) {
    const sectionModelIds = new Set(sectionRows.map((catalogRow) => catalogRow.modelId));
    const foldedRows = [];
    for (const catalogRow of sectionRows) {
        if (catalogRow.buildOf && sectionModelIds.has(catalogRow.buildOf)) continue;
        foldedRows.push({
            catalogRow,
            buildRows: sectionRows.filter((buildRow) => buildRow.buildOf === catalogRow.modelId)
        });
    }
    return foldedRows;
}

// "More detail kept · 5.3 GB more memory" under a build's name: what sets it
// apart from the lead it is folded under, and what that costs.
function downloadModelsBuildDifferenceText(buildRow, leadRow) {
    const differenceText = DOWNLOAD_MODELS_BUILD_DIFFERENCE_TEXT[buildRow.buildDifference];
    if (!differenceText) return null;
    const extraMemoryBytes = buildRow.approximateMemoryBytes - leadRow.approximateMemoryBytes;
    if (extraMemoryBytes <= 0) return differenceText;
    return differenceText + " · " + formatDownloadModelsCardSize(extraMemoryBytes) + " more memory";
}
