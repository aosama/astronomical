// Models page grouping tests: builds become versions of one model, each model
// lands in one section, and a build's fit is worded against the memory ceiling.

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

// The models come back as arrays of the script's own realm; tests copy what
// they compare with Array.from so deepEqual sees ordinary arrays.
function modelsFrom(catalogEntries) {
    const scriptContext = createDownloadModelsContext();
    scriptContext.catalogRows = catalogRowsFrom(scriptContext, catalogEntries);
    const models = vm.runInContext("downloadModelsModelsFromRows(catalogRows)", scriptContext);
    return { scriptContext, models };
}

function buildOf(shortName, versionLabel, sizeGigabytes, overrides = {}) {
    return validCatalogEntry({
        huggingface_id: "astronomical-test/" + (shortName + "-" + versionLabel).toLowerCase().replaceAll(" ", "-"),
        display_name: shortName + " " + versionLabel + " build",
        short_name: shortName,
        version_label: versionLabel,
        approximate_size_bytes: sizeGigabytes * GIGABYTE,
        ...overrides
    });
}

test("should_group_builds_sharing_a_short_name_into_one_model_ordered_smallest_first", () => {
    const { models } = modelsFrom([
        buildOf("Example 9B", "Sharper", 8),
        buildOf("Other 2B", "Compact", 2),
        buildOf("Example 9B", "Compact", 6)
    ]);

    assert.deepEqual(Array.from(models, (model) => model.name), ["Other 2B", "Example 9B"], "smallest model first");
    assert.deepEqual(Array.from(models[1].versions, (version) => version.versionLabel), ["Compact", "Sharper"]);
});

test("should_keep_a_build_without_a_short_name_as_its_own_model_under_its_display_name", () => {
    const { models } = modelsFrom([
        validCatalogEntry({ huggingface_id: "astronomical-test/one", display_name: "Plain model one" }),
        validCatalogEntry({ huggingface_id: "astronomical-test/two", display_name: "Plain model two" })
    ]);

    assert.deepEqual(Array.from(models, (model) => model.name), ["Plain model one", "Plain model two"]);
    assert.deepEqual(Array.from(models, (model) => model.versions.length), [1, 1]);
});

test("should_not_merge_builds_that_share_a_name_across_sections", () => {
    const { models } = modelsFrom([
        buildOf("Example 9B", "Compact", 6, { role: "chat" }),
        buildOf("Example 9B", "Early", 7, { role: "chat", experimental: true })
    ]);

    assert.deepEqual(Array.from(models, (model) => model.sectionKey), ["chat", "experimental"]);
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

test("should_open_a_row_on_the_downloading_then_owned_then_starter_then_smallest_build", () => {
    const { scriptContext, models } = modelsFrom([
        buildOf("Example 9B", "Compact", 6),
        buildOf("Example 9B", "Sharper", 8, { get_started: true }),
        buildOf("Example 9B", "Sharpest", 10, { ready_on_this_mac: true })
    ]);
    scriptContext.model = models[0];
    const defaultLabel = (activeDownloadModelId) => vm.runInContext(
        `downloadModelsDefaultVersion(model, ${JSON.stringify(activeDownloadModelId)}).versionLabel`, scriptContext
    );

    assert.equal(defaultLabel("astronomical-test/example-9b-compact"), "Compact", "the build being downloaded wins");
    assert.equal(defaultLabel(undefined), "Sharpest", "then a build already on this Mac");
    models[0].versions[2].readyOnThisMac = false;
    assert.equal(defaultLabel(undefined), "Sharper", "then the Get started pick");
    models[0].versions[1].getStarted = false;
    assert.equal(defaultLabel(undefined), "Compact", "then the smallest");
});

test("should_prefer_the_picked_version_and_fall_back_when_the_pick_is_unknown", () => {
    const { scriptContext, models } = modelsFrom([
        buildOf("Example 9B", "Compact", 6),
        buildOf("Example 9B", "Sharper", 8)
    ]);
    scriptContext.model = models[0];
    const selectedLabel = (pickedVersionId) => vm.runInContext(
        `downloadModelsSelectedVersion(model, { [model.modelKey]: ${JSON.stringify(pickedVersionId)} }, undefined).versionLabel`,
        scriptContext
    );

    assert.equal(selectedLabel("astronomical-test/example-9b-sharper"), "Sharper");
    assert.equal(selectedLabel("astronomical-test/no-longer-in-the-catalog"), "Compact");
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

test("should_match_a_search_against_any_build_of_a_model_and_match_everything_when_empty", () => {
    const { scriptContext, models } = modelsFrom([
        buildOf("Example 9B", "Compact", 6, { quantization_label: "4-bit affine" }),
        buildOf("Example 9B", "Sharper", 8, { quantization_label: "6-bit affine" })
    ]);
    scriptContext.model = models[0];
    const matches = (searchQuery) => vm.runInContext(
        `downloadModelsModelMatchesQuery(model, ${JSON.stringify(searchQuery)})`, scriptContext
    );

    assert.equal(matches(""), true);
    assert.equal(matches("example 9b"), true);
    assert.equal(matches("6-bit"), true, "a term found only on the second build still finds the model");
    assert.equal(matches("flux"), false);
});

test("should_count_only_models_outside_the_everyday_sections_as_hidden", () => {
    const { scriptContext, models } = modelsFrom([
        buildOf("Example 9B", "Compact", 6),
        buildOf("Helper", "Compact", 1, { role: "helper" }),
        buildOf("Early", "Compact", 1, { experimental: true }),
        buildOf("Old", "Compact", 1, { retiring: true })
    ]);
    scriptContext.models = models;

    assert.equal(vm.runInContext("downloadModelsHiddenModelCount(models)", scriptContext), 3);
    assert.equal(
        vm.runInContext('downloadModelsHiddenModelCount(models.filter((model) => model.name === "Example 9B"))', scriptContext),
        0
    );
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

test("should_summarise_an_installed_build_by_section_version_and_size", () => {
    const scriptContext = createDownloadModelsContext();
    const [labelledRow, plainRow] = catalogRowsFrom(scriptContext, [
        buildOf("Example 9B", "Compact", 6, { role: "coding" }),
        validCatalogEntry({ approximate_size_bytes: 2 * GIGABYTE, role: "image_generation" })
    ]);
    scriptContext.labelledRow = labelledRow;
    scriptContext.plainRow = plainRow;

    assert.equal(
        vm.runInContext("downloadModelsInstalledSummaryText(labelledRow)", scriptContext),
        "Chat · tuned for coding · Compact · 6.0 GB"
    );
    assert.equal(vm.runInContext("downloadModelsInstalledSummaryText(plainRow)", scriptContext), "Pictures · 2.0 GB");
});

test("should_put_coding_and_chat_models_in_one_section_ordered_by_size", () => {
    const { models } = modelsFrom([
        buildOf("Big coder", "Compact", 22, { role: "coding" }),
        buildOf("Talker", "Compact", 6, { role: "chat" }),
        buildOf("Painter", "Compact", 4, { role: "image_generation" })
    ]);
    const chatModelNames = Array.from(models)
        .filter((model) => model.sectionKey === "chat")
        .map((model) => model.name);

    assert.deepEqual(chatModelNames, ["Talker", "Big coder"]);
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
