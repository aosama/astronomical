// Models page grouping tests: every catalog row is one model named by its
// model ID, each lands in one section, and its fit is worded against the
// memory ceiling.

const {
    assert,
    createDownloadModelsContext,
    test,
    validCatalogEntry,
    vm
} = require("./download-models-test-support.js");

const GIGABYTE = 1073741824;

function catalogRowsFrom(scriptContext, catalogEntries) {
    scriptContext.catalogEntries = catalogEntries;
    return vm.runInContext("catalogEntries.map(downloadModelsCatalogRowFromEntry)", scriptContext);
}

// The rows come back as arrays of the script's own realm; tests copy what they
// compare with Array.from so deepEqual sees ordinary arrays.
function orderedRowsFrom(catalogEntries) {
    const scriptContext = createDownloadModelsContext();
    scriptContext.catalogRows = catalogRowsFrom(scriptContext, catalogEntries);
    const orderedRows = vm.runInContext("downloadModelsRowsInPageOrder(catalogRows)", scriptContext);
    return { scriptContext, orderedRows };
}

function modelNamed(modelId, sizeGigabytes, overrides = {}) {
    return validCatalogEntry({
        huggingface_id: "astronomical-test/" + modelId,
        approximate_size_bytes: sizeGigabytes * GIGABYTE,
        ...overrides
    });
}

test("should_list_every_catalog_row_as_its_own_model_smallest_first", () => {
    const { orderedRows } = orderedRowsFrom([
        modelNamed("Example-9B-MLX-6bit", 8),
        modelNamed("Other-2B-4bit", 2),
        modelNamed("Example-9B-MLX-4bit", 6)
    ]);

    assert.deepEqual(
        Array.from(orderedRows, (catalogRow) => catalogRow.modelId),
        ["Other-2B-4bit", "Example-9B-MLX-4bit", "Example-9B-MLX-6bit"],
        "builds of one family are separate models now, each under its own name"
    );
});

// A copy the person placed under another folder name answers to that name in
// chat and the terminal, so the page shows it too; otherwise the model ID.
test("should_name_a_row_by_its_installed_copy_else_by_its_model_id", () => {
    const scriptContext = createDownloadModelsContext();
    const [placedCopyRow, notDownloadedRow] = catalogRowsFrom(scriptContext, [
        modelNamed("Example-9B-MLX-6bit", 8, { ready_on_this_mac: true, requestable_model_id: "my-renamed-folder" }),
        modelNamed("Example-2B-4bit", 2, { requestable_model_id: null })
    ]);
    scriptContext.placedCopyRow = placedCopyRow;
    scriptContext.notDownloadedRow = notDownloadedRow;

    assert.equal(vm.runInContext("downloadModelsRowName(placedCopyRow)", scriptContext), "my-renamed-folder");
    assert.equal(vm.runInContext("downloadModelsRowName(notDownloadedRow)", scriptContext), "Example-2B-4bit");
});

test("should_keep_catalog_order_for_models_of_equal_size_and_leave_the_input_untouched", () => {
    const catalogEntries = [modelNamed("Second-in-catalog", 4), modelNamed("First-in-catalog", 4)];
    const { scriptContext, orderedRows } = orderedRowsFrom(catalogEntries);

    assert.deepEqual(
        Array.from(orderedRows, (catalogRow) => catalogRow.modelId), ["Second-in-catalog", "First-in-catalog"]
    );
    assert.equal(
        vm.runInContext("catalogRows[0].modelId", scriptContext), "Second-in-catalog", "ordering copies the rows"
    );
});

test("should_order_no_rows_into_an_empty_list", () => {
    const { orderedRows } = orderedRowsFrom([]);

    assert.equal(orderedRows.length, 0);
});

test("should_place_retiring_before_experimental_before_role_when_choosing_a_section", () => {
    const scriptContext = createDownloadModelsContext();
    const sectionKeyOf = (catalogRow) => {
        scriptContext.catalogRow = catalogRow;
        return vm.runInContext("downloadModelsSectionKey(catalogRow)", scriptContext);
    };

    assert.equal(sectionKeyOf({ role: "coding", retiring: true, experimental: true }), "leaving");
    assert.equal(sectionKeyOf({ role: "coding", retiring: false, experimental: true }), "experimental");
    assert.equal(sectionKeyOf({ role: "coding" }), "chat", "coding models chat too, so they share the Chat section");
    assert.equal(sectionKeyOf({ role: "image_generation" }), "pictures");
    assert.equal(sectionKeyOf({ role: "helper" }), "helpers");
    assert.equal(sectionKeyOf({ role: "decision" }), "helpers", "nobody chats with a decision model");
    assert.equal(sectionKeyOf({ role: "chat" }), "chat");
});

test("should_show_everyday_sections_for_all_the_rest_for_more_and_every_section_while_searching", () => {
    const scriptContext = createDownloadModelsContext();
    const visibleSectionKeys = (filterKey, searchQuery) => JSON.parse(JSON.stringify(vm.runInContext(
        `downloadModelsVisibleSectionKeys(${JSON.stringify(filterKey)}, ${JSON.stringify(searchQuery)})`, scriptContext
    )));

    assert.deepEqual(visibleSectionKeys("all", ""), ["chat", "pictures"]);
    assert.deepEqual(visibleSectionKeys("more", ""), ["helpers", "experimental", "leaving"]);
    assert.deepEqual(visibleSectionKeys("pictures", ""), ["pictures"]);
    assert.equal(visibleSectionKeys("pictures", "ornith").length, 5, "a search spans every section");
});

test("should_match_a_search_against_the_model_id_source_and_texts_and_everything_when_empty", () => {
    const scriptContext = createDownloadModelsContext();
    [scriptContext.catalogRow] = catalogRowsFrom(scriptContext, [
        modelNamed("Example-9B-MLX-6bit", 8, { quantization_label: "6-bit affine", summary: "A quick everyday assistant." })
    ]);
    const matches = (searchQuery) => vm.runInContext(
        `downloadModelsRowMatchesQuery(catalogRow, ${JSON.stringify(searchQuery)})`, scriptContext
    );

    assert.equal(matches(""), true);
    assert.equal(matches("example-9b"), true, "the model ID, lower-cased as the page stores queries");
    assert.equal(matches("astronomical-test"), true, "the source it downloads from");
    assert.equal(matches("6-bit affine"), true);
    assert.equal(matches("everyday"), true);
    assert.equal(matches("flux"), false);
});

test("should_count_only_models_outside_the_everyday_sections_as_hidden", () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.catalogRows = catalogRowsFrom(scriptContext, [
        modelNamed("Example-9B", 6),
        modelNamed("Helper", 1, { role: "helper" }),
        modelNamed("Early", 1, { experimental: true }),
        modelNamed("Old", 1, { retiring: true })
    ]);

    assert.equal(vm.runInContext("downloadModelsHiddenModelCount(catalogRows)", scriptContext), 3);
    assert.equal(vm.runInContext("downloadModelsHiddenModelCount(catalogRows.slice(0, 1))", scriptContext), 0);
    assert.equal(vm.runInContext("downloadModelsHiddenModelCount([])", scriptContext), 0);
});

test("should_judge_fit_at_the_exact_memory_boundary_and_stream_only_with_a_measured_floor", () => {
    const scriptContext = createDownloadModelsContext();
    const fitOutcome = (catalogRow, ceilingBytes, ceilingIsUserLimit = false) => {
        scriptContext.catalogRow = catalogRow;
        return vm.runInContext(
            `downloadModelsFitOutcome(catalogRow, ${ceilingBytes}, ${ceilingIsUserLimit})`, scriptContext
        );
    };
    const wholeModel = { approximateMemoryBytes: 8 * GIGABYTE, minimumMemoryBytes: null };
    const streamingModel = { approximateMemoryBytes: 8 * GIGABYTE, minimumMemoryBytes: 4 * GIGABYTE };

    assert.equal(fitOutcome(wholeModel, 8 * GIGABYTE), "fits", "equal to the ceiling fits");
    assert.equal(fitOutcome(wholeModel, 8 * GIGABYTE - 1), "slow", "one byte over never blocks, it is slow");
    assert.equal(fitOutcome(streamingModel, 4 * GIGABYTE), "streams", "a ceiling at the floor streams");
    assert.equal(fitOutcome(streamingModel, 4 * GIGABYTE - 1), "slow", "below the floor it is slow");
    assert.equal(fitOutcome(wholeModel, 4 * GIGABYTE, true), "over-your-limit", "a limit the person set is named");
    assert.equal(fitOutcome(wholeModel, 0), null, "an unknown ceiling claims nothing");
});

test("should_word_every_fit_outcome_as_running_and_fill_fewer_bars_as_it_slows", () => {
    const scriptContext = createDownloadModelsContext();
    const fitCopy = JSON.parse(JSON.stringify(vm.runInContext("DOWNLOAD_MODELS_FIT_COPY", scriptContext)));

    assert.deepEqual(
        ["fits", "streams", "slow"].map((fitOutcome) => fitCopy[fitOutcome].filledBarCount), [3, 2, 1]
    );
    assert.equal(fitCopy.fits.label, "Runs smoothly");
    assert.match(fitCopy.slow.explanation, /still runs/, "the Mac's size never stops a model");
});

test("should_list_abilities_in_one_fixed_order_and_nothing_for_a_model_without_any", () => {
    const scriptContext = createDownloadModelsContext();
    const abilitiesText = (catalogRow) => {
        scriptContext.catalogRow = catalogRow;
        return vm.runInContext("downloadModelsAbilitiesText(catalogRow)", scriptContext);
    };

    assert.equal(
        abilitiesText({ supportsToolCalls: true, supportsVision: true, supportsReasoning: true }),
        "Sees images · Thinks before answering · Uses tools"
    );
    assert.equal(abilitiesText({ supportsImageGeneration: true }), "Makes pictures");
    assert.equal(abilitiesText({ supportsDecisions: true }), "Answers set questions");
    assert.equal(
        abilitiesText({ role: "coding", supportsToolCalls: true }),
        "Tuned for coding · Uses tools",
        "what sets a coding model apart comes first"
    );
    assert.equal(abilitiesText({}), "");
});

test("should_round_the_context_window_to_a_feel_for_size_in_words", () => {
    const scriptContext = createDownloadModelsContext();
    const rememberedWords = (contextWindowTokens) => vm.runInContext(
        `downloadModelsRememberedWordsText(${contextWindowTokens})`, scriptContext
    );

    assert.equal(rememberedWords(262144), "About 200,000 words per conversation");
    assert.equal(rememberedWords(32768), "About 25,000 words per conversation");
    assert.equal(rememberedWords(0), null);
    assert.equal(rememberedWords(null), null);
});

test("should_summarise_an_installed_model_by_section_and_size", () => {
    const scriptContext = createDownloadModelsContext();
    const [codingRow, pictureRow] = catalogRowsFrom(scriptContext, [
        modelNamed("Example-Coder-9B", 6, { role: "coding" }),
        validCatalogEntry({ approximate_size_bytes: 2 * GIGABYTE, role: "image_generation" })
    ]);
    scriptContext.codingRow = codingRow;
    scriptContext.pictureRow = pictureRow;

    assert.equal(
        vm.runInContext("downloadModelsInstalledSummaryText(codingRow)", scriptContext),
        "Chat · tuned for coding · 6.0 GB"
    );
    assert.equal(vm.runInContext("downloadModelsInstalledSummaryText(pictureRow)", scriptContext), "Pictures · 2.0 GB");
});

test("should_put_coding_and_chat_models_in_one_section_ordered_by_size", () => {
    const { scriptContext, orderedRows } = orderedRowsFrom([
        modelNamed("Big-coder", 22, { role: "coding" }),
        modelNamed("Talker", 6, { role: "chat" }),
        modelNamed("Painter", 4, { role: "image_generation" })
    ]);
    scriptContext.orderedRows = orderedRows;
    const chatModelIds = JSON.parse(JSON.stringify(vm.runInContext(
        'orderedRows.filter((catalogRow) => downloadModelsSectionKey(catalogRow) === "chat").map((catalogRow) => catalogRow.modelId)',
        scriptContext
    )));

    assert.deepEqual(chatModelIds, ["Talker", "Big-coder"]);
});

test("should_offer_no_code_filter_now_that_coding_models_live_in_chat", () => {
    const scriptContext = createDownloadModelsContext();
    const filterKeys = JSON.parse(JSON.stringify(
        vm.runInContext("DOWNLOAD_MODELS_FILTERS.map((filterDefinition) => filterDefinition.key)", scriptContext)
    ));
    const sectionKeys = JSON.parse(JSON.stringify(
        vm.runInContext("DOWNLOAD_MODELS_SECTIONS.map((catalogSection) => catalogSection.key)", scriptContext)
    ));

    assert.deepEqual(filterKeys, ["all", "chat", "pictures", "more"]);
    assert.equal(sectionKeys.includes("code"), false);
});
