// Astronomical Observatory Models page grouping: orders catalog rows (one per
// model, each named by its model ID), places each in a section, and words how a
// model runs on this Mac.
// Pure functions over catalog rows; nothing here touches the document.

// Sections in page order. The first two are what most people come for; the
// rest stay behind the More filter so a newcomer meets a short list, not twenty.
// Coding models live in Chat: every one of them also chats, and a separate Code
// section split one model family across two lists. What sets them apart is
// shown as an ability, "Tuned for coding".
const DOWNLOAD_MODELS_SECTIONS = [
    { key: "chat", title: "Chat", note: "Talk, write, ask about photos and get help with code.", isEveryday: true },
    { key: "pictures", title: "Pictures", note: "Turn a description into an image.", isEveryday: true },
    { key: "helpers", title: "Helpers", note: "Background helpers other features use. You do not chat with these.", isEveryday: false },
    { key: "experimental", title: "Experimental", note: "Early builds that may be slower or less reliable.", isEveryday: false },
    { key: "leaving", title: "Leaving soon", note: "These leave the catalog in a future release. A copy already on this Mac keeps working.", isEveryday: false }
];
const DOWNLOAD_MODELS_FILTERS = [
    { key: "all", label: "All" },
    { key: "chat", label: "Chat" },
    { key: "pictures", label: "Pictures" },
    { key: "more", label: "More" }
];
const DOWNLOAD_MODELS_SECTION_KEY_BY_ROLE = {
    chat: "chat",
    coding: "chat",
    image_generation: "pictures",
    helper: "helpers",
    // A decision model answers typed questions for other programs; nobody
    // chats with it, so it sits with the other background helpers.
    decision: "helpers"
};
// How a model runs here, worded as speed because that is what the person
// feels. No outcome ever blocks a download: the Mac's size never stops a model
// (#1902), so the weakest label still says it runs.
const DOWNLOAD_MODELS_FIT_COPY = {
    fits: {
        label: "Runs smoothly",
        explanation: "It fits in the memory this Mac gives to models.",
        filledBarCount: 3
    },
    streams: {
        label: "Runs, but slower",
        explanation: "It is bigger than this Mac's memory for models, so part of it is read from the SSD while it answers.",
        filledBarCount: 2
    },
    slow: {
        label: "Very slow here",
        explanation: "It is much bigger than this Mac's memory for models. It still runs, with long waits between words.",
        filledBarCount: 1
    },
    "over-your-limit": {
        label: "Above your memory limit",
        explanation: "It needs more memory than the maximum you set under Configuration. Raise that maximum to run it.",
        filledBarCount: 1
    }
};
const DOWNLOAD_MODELS_ABILITIES = [
    { text: "Tuned for coding", isSupported: (catalogRow) => catalogRow.role === "coding" },
    { text: "Sees images", isSupported: (catalogRow) => catalogRow.supportsVision },
    { text: "Thinks before answering", isSupported: (catalogRow) => catalogRow.supportsReasoning },
    { text: "Uses tools", isSupported: (catalogRow) => catalogRow.supportsToolCalls },
    { text: "Makes pictures", isSupported: (catalogRow) => catalogRow.supportsImageGeneration },
    { text: "Text search", isSupported: (catalogRow) => catalogRow.supportsEmbeddings },
    { text: "Answers set questions", isSupported: (catalogRow) => catalogRow.supportsDecisions }
];
// English prose averages about three words for every four tokens; the figure
// is rounded hard because it is a feel for size, not a promise.
const DOWNLOAD_MODELS_WORDS_PER_TOKEN = 0.75;

// Retiring wins over everything so a model on its way out is never offered
// beside the ones replacing it; experimental wins over role so an early build
// never appears in an everyday section.
function downloadModelsSectionKey(catalogRow) {
    if (catalogRow.retiring) return "leaving";
    if (catalogRow.experimental) return "experimental";
    return DOWNLOAD_MODELS_SECTION_KEY_BY_ROLE[catalogRow.role];
}

// The one name the page shows for a model: the name a downloaded copy answers
// to in chat and the terminal, else the model ID the catalog declares. The two
// are the same for a copy this page downloaded; a copy the person placed under
// another folder name keeps that name, so the page never shows a second one.
function downloadModelsRowName(catalogRow) {
    return catalogRow.requestableModelId || catalogRow.modelId;
}

// Every catalog row is one model with one name, its model ID, so the list is
// the rows themselves in page order: smallest first, so each section reads as a
// ladder from quick to heavy thinker. Array sort is stable, so equal sizes keep
// catalog order.
function downloadModelsRowsInPageOrder(catalogRows) {
    return [...catalogRows].sort(
        (smallerRow, largerRow) => smallerRow.approximateSizeBytes - largerRow.approximateSizeBytes
    );
}

// A search looks through every section, so a model behind More is still found.
function downloadModelsVisibleSectionKeys(activeFilterKey, searchQuery) {
    if (searchQuery) return DOWNLOAD_MODELS_SECTIONS.map((section) => section.key);
    if (activeFilterKey === "more") {
        return DOWNLOAD_MODELS_SECTIONS.filter((section) => !section.isEveryday).map((section) => section.key);
    }
    if (activeFilterKey === "all") {
        return DOWNLOAD_MODELS_SECTIONS.filter((section) => section.isEveryday).map((section) => section.key);
    }
    return [activeFilterKey];
}

function downloadModelsRowMatchesQuery(catalogRow, searchQuery) {
    if (!searchQuery) return true;
    return [
        downloadModelsRowName(catalogRow), catalogRow.modelId, catalogRow.huggingfaceId, catalogRow.summary || "",
        catalogRow.description || "", catalogRow.quantizationLabel || ""
    ].join(" ").toLowerCase().includes(searchQuery);
}

function downloadModelsHiddenModelCount(catalogRows) {
    const everydaySectionKeys = downloadModelsVisibleSectionKeys("all", "");
    return catalogRows.filter(
        (catalogRow) => !everydaySectionKeys.includes(downloadModelsSectionKey(catalogRow))
    ).length;
}

// Fit compares the model's fully loaded memory with the serving ceiling, the
// limit the engine actually loads within. Equal counts as fitting because the
// ceiling is the runtime's own resident limit. Below that, a model with a
// streaming floor measured on a real engine streams experts or image components
// from the SSD; any other model still runs, with macOS paging the rest, so it is
// labelled as slow: the Mac's size never stops a model (#1902). Only a maximum
// the user set themself can hold a model back, and the label names it.
function downloadModelsFitOutcome(catalogRow, servingCeilingBytes, ceilingIsUserLimit = false) {
    if (!(servingCeilingBytes > 0)) return null;
    if (catalogRow.approximateMemoryBytes <= servingCeilingBytes) return "fits";
    const streamingFloorBytes = catalogRow.minimumMemoryBytes;
    if (streamingFloorBytes !== null && streamingFloorBytes <= servingCeilingBytes) {
        return "streams";
    }
    return ceilingIsUserLimit ? "over-your-limit" : "slow";
}

function downloadModelsFitLabel(catalogRow, servingCeilingBytes, ceilingIsUserLimit = false) {
    const fitOutcome = downloadModelsFitOutcome(catalogRow, servingCeilingBytes, ceilingIsUserLimit);
    return fitOutcome === null ? null : DOWNLOAD_MODELS_FIT_COPY[fitOutcome].label;
}

function downloadModelsAbilitiesText(catalogRow) {
    return DOWNLOAD_MODELS_ABILITIES
        .filter((ability) => ability.isSupported(catalogRow))
        .map((ability) => ability.text)
        .join(" · ");
}

// "About 200,000 words per conversation" for a 262,144-token window. Rounded
// to two significant figures; null when the model has no context window.
function downloadModelsRememberedWordsText(contextWindowTokens) {
    if (!(contextWindowTokens > 0)) return null;
    const approximateWords = contextWindowTokens * DOWNLOAD_MODELS_WORDS_PER_TOKEN;
    const roundingStep = 10 ** Math.max(0, Math.floor(Math.log10(approximateWords)) - 1);
    const roundedWords = Math.round(approximateWords / roundingStep) * roundingStep;
    return "About " + roundedWords.toLocaleString("en-US") + " words per conversation";
}

// One line under an installed model's name: what it is for, then its size.
function downloadModelsInstalledSummaryText(catalogRow) {
    const sectionTitle = DOWNLOAD_MODELS_SECTIONS
        .find((section) => section.key === downloadModelsSectionKey(catalogRow)).title;
    const summaryParts = [sectionTitle];
    if (catalogRow.role === "coding") summaryParts.push("tuned for coding");
    summaryParts.push(formatDownloadModelsCardSize(catalogRow.approximateSizeBytes));
    return summaryParts.join(" · ");
}

// Plain-language size for download decisions: one decimal in 2^30-byte
// gigabytes, the unit Apple prints on the box.
function formatDownloadModelsCardSize(approximateSizeBytes) {
    return formatDownloadModelsGigabyteCount(approximateSizeBytes) + " GB";
}

// The number alone, for a phrase that names the unit once ("9.2 of 21.9 GB").
function formatDownloadModelsGigabyteCount(byteCount) {
    return (byteCount / 1073741824).toFixed(1);
}
