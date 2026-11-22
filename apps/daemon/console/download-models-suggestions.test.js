// Models page suggestion tests: how the engine's suggestions are shown, and how
// the builds of one model fold under their lead. Which models are suggested is
// decided and tested in the download-catalog crate (model_suggestions_tests.rs).

const {
    assert,
    createDownloadModelsContext,
    createDownloadModelsDocument,
    findRenderedElementsByClass,
    renderedTexts,
    test,
    validCatalogEntry,
    vm
} = require("./download-models-test-support.js");

const GIGABYTE = 1073741824;

// Catalog byte counts are whole numbers; the fixtures speak in gigabytes.
function gigabytes(gigabyteCount) {
    return Math.round(gigabyteCount * GIGABYTE);
}

function entryNamed(modelId, overrides = {}) {
    return validCatalogEntry({
        huggingface_id: "astronomical-test/" + modelId,
        summary: modelId + " in one line.",
        ...overrides
    });
}

// The shape of the shipped catalog: a small and a large chat model on the
// ladder, builds of each, and a picture model.
function ladderCatalog(overridesByModelId = {}) {
    const entryFor = (modelId, fields) => entryNamed(modelId, { ...fields, ...(overridesByModelId[modelId] || {}) });
    return [
        entryFor("Small", { get_started: true, approximate_memory_bytes: gigabytes(6.6), minimum_memory_bytes: gigabytes(1.7) }),
        entryFor("Small-6bit", { approximate_memory_bytes: gigabytes(8.7), build_of: "Small", build_difference: "more_detail" }),
        entryFor("Large", { role: "coding", get_started: true, approximate_memory_bytes: gigabytes(23), minimum_memory_bytes: gigabytes(4.1) }),
        entryFor("Large-draft", { approximate_memory_bytes: gigabytes(24), build_of: "Large", build_difference: "draft_model" }),
        entryFor("Large-6bit", { approximate_memory_bytes: gigabytes(28.3), build_of: "Large", build_difference: "more_detail" }),
        entryFor("Large-8bit", { approximate_memory_bytes: gigabytes(36.4), build_of: "Large", build_difference: "more_detail" }),
        entryFor("Painter", { role: "image_generation", get_started: true, approximate_memory_bytes: gigabytes(6.2) })
    ];
}

// [tag, model] for each suggestion a Mac with this memory would see.
test("should_fold_builds_under_their_lead_and_list_a_build_alone_when_its_lead_is_not_listed", () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.catalogDocument = { schema_version: 5, entries: ladderCatalog() };
    const foldedModelIds = (modelIdsToKeep) => JSON.parse(vm.runInContext(
        `JSON.stringify(downloadModelsFoldedRows(
            downloadModelsCatalogRowsFromDocument(catalogDocument)
                .filter((catalogRow) => ${JSON.stringify(modelIdsToKeep)}.includes(catalogRow.modelId))
         ).map((foldedRow) => [foldedRow.catalogRow.modelId, foldedRow.buildRows.map((buildRow) => buildRow.modelId)]))`,
        scriptContext
    ));

    assert.deepEqual(foldedModelIds(["Small", "Small-6bit", "Large", "Large-draft", "Large-6bit"]), [
        ["Small", ["Small-6bit"]],
        ["Large", ["Large-draft", "Large-6bit"]]
    ]);
    assert.deepEqual(foldedModelIds(["Large-6bit"]), [["Large-6bit", []]], "a search can match a build alone");
});

function renderModelsPage(catalogEntries, pageStateScript = "", suggestions = []) {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    scriptContext.catalogDocument = { schema_version: 5, suggestions, entries: catalogEntries };
    vm.runInContext(
        `downloadModelsCatalogDocument = catalogDocument;
         ${pageStateScript}
         renderDownloadModelsCatalogDocument(catalogDocument)`,
        scriptContext
    );
    return downloadModelsDocument.catalogContainer;
}

function textsOfClass(rootElement, className) {
    return findRenderedElementsByClass(rootElement, className).map((element) => element.textContent);
}

test("should_keep_the_first_visit_question_instead_of_suggestions_when_nothing_is_here", () => {
    const catalogContainer = renderModelsPage(ladderCatalog(), `downloadModelsServingCeilingBytes = ${gigabytes(36.5)};`);

    assert.deepEqual(findRenderedElementsByClass(catalogContainer, "download-models-suggestions"), []);
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-first-visit-title"), ["What would you like to do first?"]);
});

test("should_fold_builds_behind_a_toggle_that_shows_them_with_what_sets_each_apart", () => {
    const catalogContainer = renderModelsPage(ladderCatalog({ Small: { ready_on_this_mac: true } }));

    assert.deepEqual(textsOfClass(catalogContainer, "download-models-row-name").slice(0, 2), ["Small", "Large"]);
    const largeToggle = findRenderedElementsByClass(catalogContainer, "download-models-fold-toggle")[1];
    assert.equal(largeToggle.textContent, "Show 3 other builds of this model");
    assert.equal(largeToggle.attributes["aria-expanded"], "false");

    for (const clickListener of largeToggle.listeners.click) clickListener();

    assert.deepEqual(
        textsOfClass(catalogContainer, "download-models-row-difference"),
        ["Draft model built in · 1.0 GB more memory", "More detail kept · 5.3 GB more memory", "More detail kept · 13.4 GB more memory"]
    );
    assert.equal(
        findRenderedElementsByClass(catalogContainer, "download-models-fold-toggle")[1].textContent,
        "Hide 3 other builds of this model"
    );
});

test("should_list_matching_builds_without_a_toggle_while_searching", () => {
    const catalogContainer = renderModelsPage(ladderCatalog(), 'downloadModelsSearchQuery = "large";');

    assert.deepEqual(findRenderedElementsByClass(catalogContainer, "download-models-fold-toggle"), []);
    assert.deepEqual(
        textsOfClass(catalogContainer, "download-models-row-name"),
        ["Large", "Large-draft", "Large-6bit", "Large-8bit"]
    );
    assert.equal(renderedTexts(catalogContainer).includes("Draft model built in · 1.0 GB more memory"), true);
});

function measuredSpeed(essayRate, codeRate) {
    return {
        measured_on: "Mac with an M5 Pro and 48 GB",
        astronomical_version: "0.2.129",
        tokens_per_second: {
            essay_short_prompt: essayRate, essay_long_prompt: essayRate,
            code_short_prompt: codeRate, code_long_prompt: codeRate
        }
    };
}

test("should_list_measured_speeds_with_where_they_were_measured_in_the_details", () => {
    const catalogContainer = renderModelsPage(
        [entryNamed("Measured", { measured_speed: measuredSpeed(96, 125) })],
        'downloadModelsExpandedModelId = "Measured";'
    );
    const detailTexts = renderedTexts(catalogContainer);

    for (const expectedText of [
        "96 tokens a second, 96 after a 10,000-token prompt",
        "125 tokens a second, 125 after a 10,000-token prompt",
        "Mac with an M5 Pro and 48 GB, Astronomical 0.2.129"
    ]) {
        assert.equal(detailTexts.includes(expectedText), true, expectedText);
    }
});


function engineSuggestion(kind, modelId, tag, reason) {
    return { kind, huggingface_id: "astronomical-test/" + modelId, model_id: modelId, tag, reason };
}

const ENGINE_SUGGESTIONS = [
    engineSuggestion("stronger", "Large", "A stronger model", "Better at hard questions and code than Small."),
    engineSuggestion("more_detail", "Large-6bit", "Closer to the original", "The same model as Large with more detail kept."),
    engineSuggestion("pictures", "Painter", "Make pictures", "You have no picture model yet.")
];

test("should_show_the_engines_suggestions_once_a_model_is_here_with_only_the_first_get_button_strong", () => {
    const catalogContainer = renderModelsPage(
        ladderCatalog({ Small: { ready_on_this_mac: true } }),
        `downloadModelsServingCeilingBytes = ${gigabytes(36.5)};`,
        ENGINE_SUGGESTIONS
    );
    const suggestionCards = findRenderedElementsByClass(catalogContainer, "download-models-suggestion-card");

    assert.deepEqual(textsOfClass(catalogContainer, "download-models-suggestions-title"), ["Suggested for this Mac"]);
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-suggestion-tag"), ["A stronger model", "Closer to the original", "Make pictures"]);
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-suggestion-model"), ["Large", "Large-6bit", "Painter"]);
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-suggestion-reason")[0], "Better at hard questions and code than Small.");
    assert.deepEqual(
        suggestionCards.map((suggestionCard) => findRenderedElementsByClass(suggestionCard, "button-primary").length),
        [1, 0, 0]
    );
    assert.equal(findRenderedElementsByClass(catalogContainer, "button-primary").length, 1, "the list's Get buttons stay quiet");
});

test("should_drop_a_suggestion_for_a_model_the_catalog_does_not_list_or_without_words", () => {
    const catalogContainer = renderModelsPage(
        ladderCatalog({ Small: { ready_on_this_mac: true } }),
        "",
        [
            engineSuggestion("best_fit", "Unlisted", "Best fit for this Mac", "Not in this catalog."),
            { kind: "pictures", huggingface_id: "astronomical-test/Painter", model_id: "Painter", tag: 7, reason: "No tag." },
            ENGINE_SUGGESTIONS[0]
        ]
    );

    assert.deepEqual(textsOfClass(catalogContainer, "download-models-suggestion-model"), ["Large"]);
});

test("should_show_no_suggestions_section_when_the_engine_suggests_nothing_or_sends_none", () => {
    const withEmptyList = renderModelsPage(ladderCatalog({ Small: { ready_on_this_mac: true } }), "", []);
    const withoutField = renderModelsPage(ladderCatalog({ Small: { ready_on_this_mac: true } }), "", undefined);

    assert.deepEqual(findRenderedElementsByClass(withEmptyList, "download-models-suggestions"), []);
    assert.deepEqual(findRenderedElementsByClass(withoutField, "download-models-suggestions"), []);
});
