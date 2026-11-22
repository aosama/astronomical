// Astronomical Observatory Models page: the engine's suggestions for this Mac,
// and how the builds of one model fold under their lead.
// Pure functions over catalog rows; nothing here touches the document.

// What each kind of build changes, in words the row shows beside its summary.
const DOWNLOAD_MODELS_BUILD_DIFFERENCE_TEXT = {
    more_detail: "More detail kept",
    draft_model: "Draft model built in",
    streaming_layout: "Laid out for streaming from the SSD"
};
const DOWNLOAD_MODELS_MAXIMUM_SUGGESTION_COUNT = 3;

// The engine's suggestions for this Mac, joined to their catalog rows:
// [{ kind, tag, reason, catalogRow }]. The engine decides and words them (the
// rules live in the download-catalog crate, shared with the chat app's first
// run); a suggestion naming a model this catalog does not list is dropped
// rather than shown without its row.
function downloadModelsSuggestionCards(catalogDocument, catalogRows) {
    const suggestions = Array.isArray(catalogDocument?.suggestions) ? catalogDocument.suggestions : [];
    const suggestionCards = [];
    for (const suggestion of suggestions) {
        const catalogRow = catalogRows.find((row) => row.huggingfaceId === suggestion?.huggingface_id);
        if (!catalogRow || typeof suggestion.tag !== "string" || typeof suggestion.reason !== "string") continue;
        suggestionCards.push({ kind: String(suggestion.kind || ""), tag: suggestion.tag, reason: suggestion.reason, catalogRow });
    }
    return suggestionCards.slice(0, DOWNLOAD_MODELS_MAXIMUM_SUGGESTION_COUNT);
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
