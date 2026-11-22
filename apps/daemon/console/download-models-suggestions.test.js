// Models page suggestion tests: which models a Mac is offered, relative to what
// it already has and how much memory it gives to models, and how the builds of
// one model fold under their lead.

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
function suggestionsFor(catalogEntries, servingCeilingBytes) {
    const scriptContext = createDownloadModelsContext();
    scriptContext.catalogDocument = { schema_version: 5, entries: catalogEntries };
    return JSON.parse(vm.runInContext(
        `JSON.stringify(downloadModelsSuggestions(
            downloadModelsCatalogRowsFromDocument(catalogDocument), ${servingCeilingBytes}
         ).map((suggestion) => [suggestion.tag, suggestion.catalogRow.modelId, suggestion.reason]))`,
        scriptContext
    ));
}

function tagsAndModels(suggestions) {
    return suggestions.map(([tag, modelId]) => [tag, modelId]);
}

test("should_offer_a_stronger_model_its_detailed_build_and_pictures_to_a_large_mac_with_the_small_model", () => {
    const suggestions = suggestionsFor(ladderCatalog({ Small: { ready_on_this_mac: true } }), gigabytes(36.5));

    assert.deepEqual(tagsAndModels(suggestions), [
        ["A stronger model", "Large"],
        ["Closer to the original", "Large-6bit"],
        ["Make pictures", "Painter"]
    ]);
    assert.match(suggestions[0][2], /than Small/, "the reason names the model it improves on");
    assert.match(suggestions[1][2], /5\.3 GB more memory/, "the reason says what the detail costs");
});

test("should_offer_a_detailed_build_and_the_large_model_streaming_to_a_small_mac", () => {
    const suggestions = suggestionsFor(ladderCatalog({ Small: { ready_on_this_mac: true } }), gigabytes(11.3));

    assert.deepEqual(tagsAndModels(suggestions), [
        ["Closer to the original", "Small-6bit"],
        ["The strongest that runs here", "Large"],
        ["Make pictures", "Painter"]
    ]);
    assert.match(suggestions[1][2], /read from the SSD/, "a streaming model is never sold as fast");
});

test("should_offer_the_lightest_model_as_the_best_fit_on_a_mac_where_nothing_fits_in_memory", () => {
    const suggestions = suggestionsFor(ladderCatalog({ Painter: { ready_on_this_mac: true } }), gigabytes(5.3));

    assert.deepEqual(tagsAndModels(suggestions), [
        ["Best fit for this Mac", "Small"],
        ["The strongest that runs here", "Large"]
    ]);
    assert.match(suggestions[0][2], /from the SSD/);
});

test("should_say_very_slow_rather_than_streaming_for_a_best_fit_without_a_measured_floor", () => {
    const suggestions = suggestionsFor(
        ladderCatalog({ Small: { minimum_memory_bytes: undefined }, Painter: { ready_on_this_mac: true } }),
        gigabytes(5.3)
    );

    assert.match(suggestions[0][2], /very slow/);
});

test("should_not_offer_a_model_whose_build_is_already_here_or_on_its_way", () => {
    const withBuildHere = suggestionsFor(
        ladderCatalog({ "Large-draft": { ready_on_this_mac: true }, Painter: { ready_on_this_mac: true } }),
        gigabytes(36.5)
    );
    const withDownloadRunning = suggestionsFor(
        ladderCatalog({ Large: { download_state: "downloading" }, Painter: { ready_on_this_mac: true } }),
        gigabytes(36.5)
    );

    assert.deepEqual(tagsAndModels(withBuildHere), [["Closer to the original", "Large-6bit"]]);
    assert.deepEqual(tagsAndModels(withDownloadRunning), [["Closer to the original", "Large-6bit"]]);
});

test("should_not_offer_more_detail_when_a_build_here_is_at_least_as_large", () => {
    const withLargerBuildHere = suggestionsFor(
        ladderCatalog({
            "Large-draft": { approximate_memory_bytes: gigabytes(29), ready_on_this_mac: true },
            Painter: { ready_on_this_mac: true }
        }),
        gigabytes(36.5)
    );
    const withSmallerBuildHere = suggestionsFor(
        ladderCatalog({ "Large-draft": { ready_on_this_mac: true }, Painter: { ready_on_this_mac: true } }),
        gigabytes(36.5)
    );

    assert.deepEqual(withLargerBuildHere, [], "a 29 GB build already keeps the 28.3 GB build's detail");
    assert.deepEqual(tagsAndModels(withSmallerBuildHere), [["Closer to the original", "Large-6bit"]]);
});

test("should_offer_more_detail_only_when_it_leaves_a_tenth_of_the_memory_free", () => {
    const roomyCeilingBytes = Math.ceil(gigabytes(28.3) / 0.9);
    const atTheEdge = suggestionsFor(ladderCatalog({ Painter: { ready_on_this_mac: true } }), roomyCeilingBytes);
    const justShort = suggestionsFor(ladderCatalog({ Painter: { ready_on_this_mac: true } }), roomyCeilingBytes - GIGABYTE);

    assert.deepEqual(tagsAndModels(atTheEdge)[1], ["Closer to the original", "Large-6bit"]);
    assert.deepEqual(tagsAndModels(justShort), [["Best fit for this Mac", "Large"]]);
});

test("should_suggest_nothing_while_the_memory_ceiling_is_unknown_or_the_catalog_has_no_ladder", () => {
    assert.deepEqual(suggestionsFor(ladderCatalog(), null), []);
    assert.deepEqual(suggestionsFor(ladderCatalog(), 0), []);
    assert.deepEqual(suggestionsFor([entryNamed("Plain"), entryNamed("Painter", { role: "image_generation" })], gigabytes(36)), []);
});

test("should_keep_experimental_and_retiring_picks_off_the_ladder", () => {
    const suggestions = suggestionsFor(
        ladderCatalog({ Large: { experimental: true }, Painter: { ready_on_this_mac: true } }),
        gigabytes(36.5)
    );

    assert.deepEqual(tagsAndModels(suggestions)[0], ["Best fit for this Mac", "Small"]);
});

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

function renderModelsPage(catalogEntries, pageStateScript = "") {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    scriptContext.catalogDocument = { schema_version: 5, entries: catalogEntries };
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

test("should_show_suggestions_once_a_model_is_here_with_only_the_first_get_button_strong", () => {
    const catalogContainer = renderModelsPage(
        ladderCatalog({ Small: { ready_on_this_mac: true } }),
        `downloadModelsServingCeilingBytes = ${gigabytes(36.5)};`
    );
    const suggestionCards = findRenderedElementsByClass(catalogContainer, "download-models-suggestion-card");

    assert.deepEqual(textsOfClass(catalogContainer, "download-models-suggestions-title"), ["Suggested for this Mac"]);
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-suggestion-model"), ["Large", "Large-6bit", "Painter"]);
    assert.deepEqual(
        suggestionCards.map((suggestionCard) => findRenderedElementsByClass(suggestionCard, "button-primary").length),
        [1, 0, 0]
    );
    assert.equal(
        findRenderedElementsByClass(catalogContainer, "button-primary").length, 1,
        "the list's Get buttons stay quiet, so the page asks for one decision"
    );
});

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

test("should_suggest_a_draft_build_for_code_when_it_measured_clearly_quicker_and_say_what_it_costs", () => {
    const suggestions = suggestionsFor(
        ladderCatalog({
            Large: { measured_speed: measuredSpeed(96, 96) },
            "Large-draft": { measured_speed: measuredSpeed(89, 125) },
            Painter: { ready_on_this_mac: true }
        }),
        gigabytes(36.5)
    );

    assert.deepEqual(tagsAndModels(suggestions).slice(0, 2), [
        ["Best fit for this Mac", "Large"],
        ["Quicker for code", "Large-draft"]
    ]);
    assert.match(suggestions[1][2], /about 30% quicker than Large/);
    assert.match(suggestions[1][2], /essay is about 7% slower/, "a slower everyday reply is disclosed");
});

test("should_not_suggest_a_draft_build_inside_the_measurement_spread_or_without_figures", () => {
    const withinSpread = suggestionsFor(
        ladderCatalog({
            Large: { measured_speed: measuredSpeed(96, 100) },
            "Large-draft": { measured_speed: measuredSpeed(96, 109) },
            Painter: { ready_on_this_mac: true }
        }),
        gigabytes(36.5)
    );
    const unmeasured = suggestionsFor(
        ladderCatalog({ "Large-draft": { measured_speed: measuredSpeed(89, 125) }, Painter: { ready_on_this_mac: true } }),
        gigabytes(36.5)
    );

    assert.equal(tagsAndModels(withinSpread).some(([tag]) => tag === "Quicker for code"), false, "9% is within the spread");
    assert.equal(tagsAndModels(unmeasured).some(([tag]) => tag === "Quicker for code"), false, "no reference figure, no claim");
});

test("should_never_suggest_a_draft_build_for_speed_when_it_would_not_fit_whole_in_memory", () => {
    const suggestions = suggestionsFor(
        ladderCatalog({
            Large: { measured_speed: measuredSpeed(96, 96) },
            "Large-draft": { measured_speed: measuredSpeed(89, 125), approximate_memory_bytes: gigabytes(26) },
            Painter: { ready_on_this_mac: true }
        }),
        gigabytes(26)
    );

    assert.equal(tagsAndModels(suggestions).some(([tag]) => tag === "Quicker for code"), false, "a streaming draft build drafts nothing yet");
});

test("should_measure_a_draft_build_against_the_quickest_build_already_here", () => {
    const suggestions = suggestionsFor(
        ladderCatalog({
            "Large-6bit": { ready_on_this_mac: true, measured_speed: measuredSpeed(72, 118) },
            "Large-draft": { measured_speed: measuredSpeed(89, 125) },
            Painter: { ready_on_this_mac: true }
        }),
        gigabytes(36.5)
    );

    assert.equal(tagsAndModels(suggestions).some(([tag]) => tag === "Quicker for code"), false, "125 is not clearly quicker than 118");
});

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
