// Models page tests: what a first visit shows, the Add a model list with its
// filter chips and search, and one model row, named by its model ID, with its details.

const {
    assert,
    createDownloadModelsContext,
    createDownloadModelsDocument,
    findRenderedElement,
    findRenderedElementsByClass,
    renderedTexts,
    test,
    validCatalogEntry,
    vm
} = require("./download-models-test-support.js");

const GIGABYTE = 1073741824;

// Renders a catalog with explicit page state, so each test states the exact
// inputs it depends on instead of inheriting defaults.
function renderModelsPage(catalogEntries, pageStateScript = "") {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    scriptContext.catalogDocument = { schema_version: 5, entries: catalogEntries };
    const renderedState = vm.runInContext(
        `downloadModelsCatalogDocument = catalogDocument;
         ${pageStateScript}
         renderDownloadModelsCatalogDocument(catalogDocument)`,
        scriptContext
    );
    return { scriptContext, downloadModelsDocument, renderedState };
}

function textsOfClass(rootElement, className) {
    return findRenderedElementsByClass(rootElement, className).map((element) => element.textContent);
}

function clickRendered(renderedElement) {
    for (const clickListener of renderedElement.listeners.click) clickListener();
}

// A fixture model named by its model ID, which is the leaf of its Hugging
// Face identity exactly as the daemon publishes it.
function entryNamed(modelId, overrides = {}) {
    return validCatalogEntry({
        huggingface_id: "astronomical-test/" + modelId,
        summary: modelId + " in one line.",
        ...overrides
    });
}

const STARTER_ENTRIES = [
    entryNamed("Coder", { role: "coding", get_started: true, approximate_size_bytes: 22 * GIGABYTE }),
    entryNamed("Painter", { role: "image_generation", get_started: true, approximate_size_bytes: 4 * GIGABYTE }),
    entryNamed("Talker", { role: "chat", get_started: true, approximate_size_bytes: 6 * GIGABYTE }),
    entryNamed("Other-talker", { role: "chat" })
];

test("should_ask_what_to_do_first_and_offer_the_three_starters_when_nothing_is_on_this_mac", () => {
    const { downloadModelsDocument, renderedState } = renderModelsPage(STARTER_ENTRIES);
    const catalogContainer = downloadModelsDocument.catalogContainer;

    assert.equal(renderedState, "ready");
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-first-visit-title"), ["What would you like to do first?"]);
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-starter-title"), ["Chat", "Make pictures", "Write code"]);
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-starter-model"), ["Talker", "Painter", "Coder"]);
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-installed"), [], "no On this Mac section yet");
});

test("should_mark_only_the_first_starter_as_the_place_to_start_and_put_the_size_on_each_button", () => {
    const { downloadModelsDocument } = renderModelsPage(STARTER_ENTRIES);
    const starterCards = findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-starter-card");

    assert.deepEqual(starterCards.map((starterCard) => textsOfClass(starterCard, "download-models-starter-tag")), [["Start here"], [], []]);
    assert.deepEqual(
        starterCards.map((starterCard) => textsOfClass(starterCard, "download-models-download-button")[0]),
        ["Get · 6.0 GB", "Get · 4.0 GB", "Get · 22.0 GB"]
    );
});

test("should_skip_a_starter_whose_role_has_no_get_started_pick", () => {
    const { downloadModelsDocument } = renderModelsPage([STARTER_ENTRIES[2], STARTER_ENTRIES[3]]);

    assert.deepEqual(textsOfClass(downloadModelsDocument.catalogContainer, "download-models-starter-title"), ["Chat"]);
});

test("should_replace_the_first_visit_question_with_on_this_mac_once_a_model_is_here", () => {
    const { downloadModelsDocument } = renderModelsPage([
        entryNamed("Talker", { get_started: true, ready_on_this_mac: true }),
        entryNamed("Other-talker")
    ]);
    const catalogContainer = downloadModelsDocument.catalogContainer;

    assert.deepEqual(findRenderedElementsByClass(catalogContainer, "download-models-first-visit"), []);
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-installed-name"), ["Talker"]);
});

test("should_show_everyday_sections_by_default_and_offer_the_rest_behind_show_more", () => {
    const { downloadModelsDocument } = renderModelsPage([
        entryNamed("Talker"),
        entryNamed("Coder", { role: "coding" }),
        entryNamed("Painter", { role: "image_generation" }),
        entryNamed("Helper", { role: "helper" }),
        entryNamed("Early", { experimental: true }),
        entryNamed("Old", { retiring: true })
    ]);
    const catalogContainer = downloadModelsDocument.catalogContainer;
    const sectionTitles = findRenderedElementsByClass(catalogContainer, "download-models-group-heading")
        .map((groupHeading) => groupHeading.children[0].textContent);

    assert.deepEqual(sectionTitles, ["Chat", "Pictures"]);
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-row-name").slice(0, 2), ["Talker", "Coder"]);
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-chip"), ["All", "Chat", "Pictures", "More"]);
    assert.deepEqual(
        textsOfClass(catalogContainer, "download-models-show-more"),
        ["Show 3 more: helpers, experimental builds and models leaving soon"]
    );
});

test("should_show_only_the_hidden_sections_after_show_more_is_clicked", () => {
    const { downloadModelsDocument } = renderModelsPage([
        entryNamed("Talker"),
        entryNamed("Early-coder", { role: "coding", experimental: true }),
        entryNamed("Old", { retiring: true, experimental: true })
    ]);
    const catalogContainer = downloadModelsDocument.catalogContainer;

    clickRendered(findRenderedElementsByClass(catalogContainer, "download-models-show-more")[0]);

    const sectionTitles = findRenderedElementsByClass(catalogContainer, "download-models-group-heading")
        .map((groupHeading) => groupHeading.children[0].textContent);
    assert.deepEqual(sectionTitles, ["Experimental", "Leaving soon"]);
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-chip-pressed"), ["More"]);
    assert.deepEqual(findRenderedElementsByClass(catalogContainer, "download-models-show-more"), []);
});

test("should_press_exactly_one_chip_and_none_while_a_search_spans_every_section", () => {
    const filtered = renderModelsPage([entryNamed("Talker")], `downloadModelsActiveFilter = "chat";`);
    const searching = renderModelsPage(
        [entryNamed("Talker"), entryNamed("Old", { retiring: true })],
        `downloadModelsActiveFilter = "chat"; downloadModelsSearchQuery = "old";`
    );

    assert.deepEqual(textsOfClass(filtered.downloadModelsDocument.catalogContainer, "download-models-chip-pressed"), ["Chat"]);
    assert.deepEqual(textsOfClass(searching.downloadModelsDocument.catalogContainer, "download-models-chip-pressed"), []);
    assert.deepEqual(
        textsOfClass(searching.downloadModelsDocument.catalogContainer, "download-models-row-name"), ["Old"],
        "the search reaches a model behind More"
    );
});

test("should_say_nothing_matches_instead_of_showing_an_empty_list", () => {
    const { downloadModelsDocument } = renderModelsPage([entryNamed("Talker")], `downloadModelsSearchQuery = "zebra";`);
    const catalogContainer = downloadModelsDocument.catalogContainer;

    assert.deepEqual(findRenderedElementsByClass(catalogContainer, "download-models-row"), []);
    assert.equal(textsOfClass(catalogContainer, "download-models-no-matches").length, 1);
});

test("should_show_one_row_per_model_named_by_its_model_id_with_no_version_picker", () => {
    const { downloadModelsDocument } = renderModelsPage([
        entryNamed("Talker-9B-6bit", { approximate_size_bytes: 8 * GIGABYTE }),
        entryNamed("Talker-9B-4bit", { approximate_size_bytes: 6 * GIGABYTE }),
        entryNamed("Solo")
    ]);
    const catalogContainer = downloadModelsDocument.catalogContainer;
    const modelRows = findRenderedElementsByClass(catalogContainer, "download-models-row");

    assert.deepEqual(
        modelRows.map((modelRow) => textsOfClass(modelRow, "download-models-row-name")[0]),
        ["Solo", "Talker-9B-4bit", "Talker-9B-6bit"],
        "two builds of one family are two rows, smallest first, each under the name the terminal accepts"
    );
    assert.deepEqual(
        modelRows.map((modelRow) => textsOfClass(modelRow, "download-models-download-button")),
        [["Get · 3.7 GB"], ["Get · 6.0 GB"], ["Get · 8.0 GB"]]
    );
    assert.deepEqual(findRenderedElementsByClass(catalogContainer, "download-models-version-picker"), []);
});

test("should_show_a_model_on_this_mac_as_here_instead_of_offering_get", () => {
    const { downloadModelsDocument } = renderModelsPage([entryNamed("Talker", { ready_on_this_mac: true })]);
    const modelRow = findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-row")[0];

    assert.deepEqual(textsOfClass(modelRow, "download-models-row-status"), ["On this Mac"]);
    assert.deepEqual(textsOfClass(modelRow, "download-models-download-button"), []);
});

test("should_disable_get_and_say_why_while_another_download_holds_the_slot", () => {
    const { downloadModelsDocument } = renderModelsPage(
        [entryNamed("Talker"), entryNamed("Painter", { role: "image_generation" })],
        `downloadModelsCurrentDownload = { huggingface_id: "astronomical-test/Talker", state: "downloading", bytes_total: 10, bytes_completed: 4 };`
    );
    const catalogContainer = downloadModelsDocument.catalogContainer;
    const getButtons = findRenderedElementsByClass(catalogContainer, "download-models-download-button");

    assert.deepEqual(getButtons.map((getButton) => getButton.disabled), [true]);
    assert.equal(textsOfClass(catalogContainer, "download-models-busy-hint").length, 1);
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-row-status"), ["Downloading · 40%"]);
});

test("should_keep_get_enabled_and_show_no_hint_when_nothing_is_downloading", () => {
    const { downloadModelsDocument } = renderModelsPage([entryNamed("Talker")]);
    const catalogContainer = downloadModelsDocument.catalogContainer;

    assert.deepEqual(
        findRenderedElementsByClass(catalogContainer, "download-models-download-button").map((getButton) => getButton.disabled),
        [false]
    );
    assert.deepEqual(findRenderedElementsByClass(catalogContainer, "download-models-busy-hint"), []);
});

test("should_draw_the_speed_meter_with_fewer_filled_bars_for_a_model_above_the_ceiling", () => {
    const { downloadModelsDocument } = renderModelsPage(
        [
            entryNamed("Small", { approximate_memory_bytes: 4 * GIGABYTE }),
            entryNamed("Streaming", { approximate_memory_bytes: 20 * GIGABYTE, minimum_memory_bytes: 4 * GIGABYTE }),
            entryNamed("Huge", { approximate_memory_bytes: 20 * GIGABYTE })
        ],
        `downloadModelsServingCeilingBytes = ${8 * GIGABYTE};`
    );
    const catalogContainer = downloadModelsDocument.catalogContainer;
    const filledBarCounts = findRenderedElementsByClass(catalogContainer, "download-models-fit")
        .map((fitMeter) => findRenderedElementsByClass(fitMeter, "download-models-fit-bar-filled").length);

    assert.deepEqual(filledBarCounts, [3, 2, 1]);
    assert.deepEqual(
        textsOfClass(catalogContainer, "download-models-fit-label"),
        ["Runs smoothly", "Runs, but slower", "Very slow here"]
    );
});

test("should_show_no_speed_meter_and_no_memory_figure_while_the_ceiling_is_unknown", () => {
    const { downloadModelsDocument } = renderModelsPage([entryNamed("Talker")]);
    const catalogContainer = downloadModelsDocument.catalogContainer;

    assert.deepEqual(findRenderedElementsByClass(catalogContainer, "download-models-fit"), []);
    assert.deepEqual(findRenderedElementsByClass(catalogContainer, "download-models-section-meta"), []);
});

test("should_open_the_details_from_the_name_and_keep_the_technical_facts_inside_them", () => {
    const { downloadModelsDocument } = renderModelsPage([entryNamed("Talker")]);
    const catalogContainer = downloadModelsDocument.catalogContainer;
    const collapsedTexts = renderedTexts(catalogContainer);
    assert.equal(collapsedTexts.includes("astronomical-test/Talker"), false, "the source stays out of the plain row");
    assert.equal(collapsedTexts.includes("Talker"), true, "the row shows the model ID");

    const nameButton = findRenderedElementsByClass(catalogContainer, "download-models-row-name")[0];
    assert.equal(nameButton.attributes["aria-expanded"], "false");
    clickRendered(nameButton);

    const expandedTexts = renderedTexts(catalogContainer);
    for (const technicalFact of ["Downloaded from", "astronomical-test/Talker", "6-bit affine (group 64)", "MIT", "About 25,000 words per conversation"]) {
        assert.equal(expandedTexts.includes(technicalFact), true, technicalFact);
    }
    for (const retiredLabel of ["Full name", "Published as"]) {
        assert.equal(expandedTexts.includes(retiredLabel), false, retiredLabel + " named a second name");
    }
    assert.equal(
        findRenderedElementsByClass(catalogContainer, "download-models-row-name")[0].attributes["aria-expanded"], "true"
    );
});

test("should_treat_a_missing_or_malformed_streaming_floor_as_no_streaming", () => {
    const scriptContext = createDownloadModelsContext();
    for (const malformedFloor of [undefined, null, 0, -1, "4000000000", 1.5]) {
        scriptContext.catalogEntry = validCatalogEntry({ minimum_memory_bytes: malformedFloor });
        assert.equal(
            vm.runInContext("downloadModelsCatalogRowFromEntry(catalogEntry).minimumMemoryBytes", scriptContext),
            null,
            `floor ${String(malformedFloor)} must not become a streaming promise`
        );
    }
    scriptContext.catalogEntry = validCatalogEntry({ minimum_memory_bytes: 4_451_287_808 });
    assert.equal(
        vm.runInContext("downloadModelsCatalogRowFromEntry(catalogEntry).minimumMemoryBytes", scriptContext),
        4_451_287_808
    );
});

test("should_return_no_fit_label_when_the_serving_ceiling_is_unknown", () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.catalogRow = { approximateMemoryBytes: 8_000_000_000, minimumMemoryBytes: 4_000_000_000 };
    for (const unknownCeiling of ["null", "0", "undefined"]) {
        assert.equal(
            vm.runInContext(`downloadModelsFitLabel(catalogRow, ${unknownCeiling})`, scriptContext),
            null,
            `a ${unknownCeiling} ceiling must show no fit label`
        );
    }
});

test("should_render_the_unavailable_state_for_a_catalog_of_an_older_schema", () => {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    scriptContext.catalogDocument = { schema_version: 3, entries: [validCatalogEntry()] };

    assert.equal(
        vm.runInContext("renderDownloadModelsCatalogDocument(catalogDocument)", scriptContext),
        "unavailable"
    );
    assert.equal(downloadModelsDocument.catalogContainer.dataset.downloadModelsState, "unavailable");
});

test("should_treat_only_a_configured_maximum_below_the_machine_as_a_user_limit", () => {
    const scriptContext = createDownloadModelsContext();
    const machineCeilingBytes = 48 * GIGABYTE;
    const isUserLimit = (configuredGigabytes) => {
        scriptContext.statusDocument = {
            configured_maximum_mlx_memory_gb: configuredGigabytes,
            machine_mlx_memory_ceiling_bytes: machineCeilingBytes
        };
        return vm.runInContext("readDownloadModelsCeilingIsUserLimit(statusDocument)", scriptContext);
    };
    assert.equal(isUserLimit(32), true);
    assert.equal(isUserLimit(47), true, "one gigabyte under the machine is a user limit");
    assert.equal(isUserLimit(48), false, "a maximum equal to the machine limits nothing");
    assert.equal(isUserLimit(null), false, "no configured maximum is no user limit");
    scriptContext.statusDocument = { configured_maximum_mlx_memory_gb: 32 };
    assert.equal(
        vm.runInContext("readDownloadModelsCeilingIsUserLimit(statusDocument)", scriptContext),
        false,
        "without the machine ceiling nothing can be claimed"
    );
});
