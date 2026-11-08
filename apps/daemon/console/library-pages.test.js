// Library page tests: Get started role cards, More models groups, fit labels
// against the serving memory ceiling, and keyboard access to the page tabs.

const {
    assert,
    createLibraryContext,
    createLibraryDocument,
    findRenderedElement,
    findRenderedElementsByClass,
    renderedTexts,
    test,
    validCatalogEntry,
    vm
} = require("./library-test-support.js");

// Renders a catalog with an explicit page and serving memory ceiling, so each test
// states the exact inputs it depends on instead of inheriting defaults.
function renderLibraryWithEntries(catalogEntries, { activeTab = "get-started", servingCeilingBytes = null } = {}) {
    const scriptContext = createLibraryContext();
    const libraryDocument = createLibraryDocument();
    scriptContext.document = libraryDocument;
    scriptContext.catalogDocument = { schema_version: 3, entries: catalogEntries };
    vm.runInContext(
        `libraryActiveTab = ${JSON.stringify(activeTab)};
         libraryServingCeilingBytes = ${JSON.stringify(servingCeilingBytes)};
         libraryCatalogDocument = catalogDocument;`,
        scriptContext
    );
    const renderedState = vm.runInContext(
        "renderLibraryCatalogDocument(catalogDocument)",
        scriptContext
    );
    return { scriptContext, libraryDocument, renderedState };
}

function renderedTextsOfClass(rootElement, className) {
    return findRenderedElementsByClass(rootElement, className).map((element) => element.textContent);
}

test("should_label_fit_at_the_exact_memory_boundary_and_stream_only_when_allowed", () => {
    const scriptContext = createLibraryContext();
    scriptContext.catalogRow = { approximateMemoryBytes: 8_000_000_000, minimumMemoryBytes: null };
    assert.equal(
        vm.runInContext("libraryFitLabel(catalogRow, 8_000_000_000)", scriptContext),
        "Runs in memory on this Mac",
        "memory equal to the ceiling still fits"
    );
    assert.equal(
        vm.runInContext("libraryFitLabel(catalogRow, 7_999_999_999)", scriptContext),
        "Needs more memory than this Mac has",
        "one byte over the ceiling does not fit"
    );
    scriptContext.catalogRow.minimumMemoryBytes = 4_000_000_000;
    assert.equal(
        vm.runInContext("libraryFitLabel(catalogRow, 7_999_999_999)", scriptContext),
        "Streams from your SSD on this Mac (slower)",
        "a model with a measured streaming floor is offered the SSD path"
    );
    assert.equal(
        vm.runInContext("libraryFitLabel(catalogRow, 4_000_000_000)", scriptContext),
        "Streams from your SSD on this Mac (slower)",
        "a ceiling exactly at the streaming floor still streams"
    );
    assert.equal(
        vm.runInContext("libraryFitLabel(catalogRow, 3_999_999_999)", scriptContext),
        "Needs more memory than this Mac has",
        "one byte under the streaming floor cannot run"
    );
});

test("should_treat_a_missing_or_malformed_streaming_floor_as_no_streaming", () => {
    const scriptContext = createLibraryContext();
    for (const malformedFloor of [undefined, null, 0, -1, "4000000000", 1.5]) {
        scriptContext.catalogEntry = validCatalogEntry({ minimum_memory_bytes: malformedFloor });
        assert.equal(
            vm.runInContext("libraryCatalogRowFromEntry(catalogEntry).minimumMemoryBytes", scriptContext),
            null,
            `floor ${String(malformedFloor)} must not become a streaming promise`
        );
    }
    scriptContext.catalogEntry = validCatalogEntry({ minimum_memory_bytes: 4_451_287_808 });
    assert.equal(
        vm.runInContext("libraryCatalogRowFromEntry(catalogEntry).minimumMemoryBytes", scriptContext),
        4_451_287_808
    );
});

test("should_return_no_fit_label_when_the_serving_ceiling_is_unknown", () => {
    const scriptContext = createLibraryContext();
    scriptContext.catalogRow = { approximateMemoryBytes: 8_000_000_000, minimumMemoryBytes: 4_000_000_000 };
    for (const unknownCeiling of ["null", "0", "undefined"]) {
        assert.equal(
            vm.runInContext(`libraryFitLabel(catalogRow, ${unknownCeiling})`, scriptContext),
            null,
            `a ${unknownCeiling} ceiling must show no fit label`
        );
    }
});

test("should_judge_fit_against_the_lower_serving_ceiling_not_the_machine_maximum", () => {
    // The engine serves within the configured 32 GB, so a 40 GB model does not
    // run in memory even though the 40 GB machine could hold it.
    const scriptContext = createLibraryContext();
    scriptContext.catalogRow = { approximateMemoryBytes: 40_000_000_000, minimumMemoryBytes: null };
    scriptContext.statusDocument = {
        mlx_memory_ceiling_bytes: 34_359_738_368,
        machine_mlx_memory_ceiling_bytes: 42_949_672_960
    };
    vm.runInContext("libraryServingCeilingBytes = readLibraryServingCeilingBytes(statusDocument)", scriptContext);
    assert.equal(
        vm.runInContext("libraryFitLabel(catalogRow, libraryServingCeilingBytes)", scriptContext),
        "Needs more memory than this Mac has",
        "a model between the serving and machine ceilings must not be labelled as fitting"
    );
});

test("should_mark_each_fit_outcome_with_its_own_css_modifier_class", () => {
    const ceilingBytes = 4_000_000_000;
    const { libraryDocument, renderedState } = renderLibraryWithEntries([
        validCatalogEntry({ huggingface_id: "astronomical-test/fits", approximate_memory_bytes: 3_000_000_000 }),
        validCatalogEntry({ huggingface_id: "astronomical-test/streams", approximate_memory_bytes: 9_000_000_000, minimum_memory_bytes: 3_500_000_000 }),
        validCatalogEntry({ huggingface_id: "astronomical-test/too-big", approximate_memory_bytes: 9_000_000_000 })
    ], { activeTab: "more-models", servingCeilingBytes: ceilingBytes });

    assert.equal(renderedState, "ready");
    const fitClasses = findRenderedElementsByClass(libraryDocument.catalogContainer, "library-fit-label")
        .map((fitLabel) => fitLabel.className);
    assert.deepEqual(fitClasses, [
        "library-fit-label library-fit-fits",
        "library-fit-label library-fit-streams",
        "library-fit-label library-fit-too-big"
    ]);
});

test("should_render_get_started_picks_in_role_order_with_size_lines", () => {
    const { libraryDocument } = renderLibraryWithEntries([
        validCatalogEntry({ huggingface_id: "astronomical-test/image-pick", display_name: "Image pick", role: "image_generation", get_started: true, approximate_size_bytes: 3_221_225_472 }),
        validCatalogEntry({ huggingface_id: "astronomical-test/chat-pick", display_name: "Chat pick", role: "chat", get_started: true, approximate_size_bytes: 5_368_709_120 }),
        validCatalogEntry({ huggingface_id: "astronomical-test/chat-other", display_name: "Other chat", role: "chat" }),
        validCatalogEntry({ huggingface_id: "astronomical-test/coding-pick", display_name: "Coding pick", role: "coding", get_started: true, approximate_size_bytes: 2_147_483_648 })
    ]);

    const container = libraryDocument.catalogContainer;
    assert.deepEqual(renderedTextsOfClass(container, "library-role-title"), ["Chat", "Code", "Pictures"]);
    assert.deepEqual(renderedTextsOfClass(container, "library-role-size"), [
        "5.0 GB download",
        "2.0 GB download",
        "3.0 GB download"
    ]);
    assert.deepEqual(renderedTextsOfClass(container, "library-role-model"), ["Chat pick", "Coding pick", "Image pick"]);
    assert.equal(
        renderedTexts(container).some((text) => text.includes("astronomical-test/")),
        false,
        "Get started must not show Hugging Face ids"
    );
});

test("should_skip_a_role_that_has_no_get_started_pick", () => {
    const { libraryDocument } = renderLibraryWithEntries([
        validCatalogEntry({ huggingface_id: "astronomical-test/chat-pick", display_name: "Chat pick", role: "chat", get_started: true }),
        validCatalogEntry({ huggingface_id: "astronomical-test/coding-other", display_name: "Coding other", role: "coding" }),
        validCatalogEntry({ huggingface_id: "astronomical-test/image-pick", display_name: "Image pick", role: "image_generation", get_started: true })
    ]);

    assert.deepEqual(
        renderedTextsOfClass(libraryDocument.catalogContainer, "library-role-title"),
        ["Chat", "Pictures"]
    );
});

test("should_group_more_models_and_keep_experimental_entries_only_in_experimental", () => {
    const { libraryDocument } = renderLibraryWithEntries([
        validCatalogEntry({ huggingface_id: "astronomical-test/chat-a", display_name: "Chat A", role: "chat" }),
        validCatalogEntry({ huggingface_id: "astronomical-test/helper-h", display_name: "Helper H", role: "helper" }),
        validCatalogEntry({ huggingface_id: "astronomical-test/coder-x", display_name: "Experimental coder", role: "coding", experimental: true }),
        validCatalogEntry({ huggingface_id: "astronomical-test/helper-x", display_name: "Experimental helper", role: "helper", experimental: true })
    ], { activeTab: "more-models" });

    const groupSections = findRenderedElementsByClass(libraryDocument.catalogContainer, "library-group");
    const groupSummaries = groupSections.map((groupSection) => ({
        heading: groupSection.children[0].textContent,
        models: findRenderedElementsByClass(groupSection, "library-model-card").map(
            (modelCard) => findRenderedElement(modelCard, (element) => element.tagName === "H4").textContent
        )
    }));
    assert.deepEqual(groupSummaries.map((groupSummary) => groupSummary.heading), [
        "Chat",
        "Helpers",
        "Experimental"
    ], "empty Code and Pictures groups must be hidden");
    assert.equal(groupSummaries[1].models.length, 1, "Helpers holds only the non-experimental helper");
    assert.equal(groupSummaries[2].models.length, 2, "Experimental holds both experimental entries");
    assert.equal(
        findRenderedElementsByClass(libraryDocument.catalogContainer, "library-model-size").length,
        4,
        "every More models card shows its size line"
    );
});

test("should_keep_the_selected_page_across_a_poll_rerender", () => {
    const { scriptContext, libraryDocument } = renderLibraryWithEntries([validCatalogEntry()]);
    vm.runInContext('selectLibraryTab("more-models")', scriptContext);

    const selectedTab = () => findRenderedElement(
        libraryDocument.catalogContainer,
        (element) => element.id === "library-tab-more-models"
    );
    assert.equal(selectedTab().attributes["aria-selected"], "true");
    assert.equal(selectedTab().tabIndex, 0, "the selected tab stays in the Tab order");

    // The one-second poll re-renders with unchanged catalog input; the page must not revert.
    const pollOutcome = vm.runInContext("renderLibraryCatalogDocument(catalogDocument)", scriptContext);
    assert.equal(pollOutcome, "ready");
    assert.equal(selectedTab().attributes["aria-selected"], "true");
    assert.notEqual(
        findRenderedElement(libraryDocument.catalogContainer, (element) => element.className === "library-hero-summary"),
        undefined,
        "the More models hero must still be showing"
    );
});

test("should_render_the_unavailable_state_for_a_schema_version_2_catalog", () => {
    const scriptContext = createLibraryContext();
    const libraryDocument = createLibraryDocument();
    scriptContext.document = libraryDocument;
    scriptContext.catalogDocument = { schema_version: 2, entries: [validCatalogEntry()] };

    assert.equal(
        vm.runInContext("renderLibraryCatalogDocument(catalogDocument)", scriptContext),
        "unavailable"
    );
    assert.equal(libraryDocument.catalogContainer.dataset.libraryState, "unavailable");
});

test("should_map_arrow_home_and_end_keys_to_the_wrapping_tab_order", () => {
    const scriptContext = createLibraryContext();
    const tabNameForKey = (currentTabName, pressedKey) => vm.runInContext(
        `libraryTabNameForKey(${JSON.stringify(currentTabName)}, ${JSON.stringify(pressedKey)})`,
        scriptContext
    );
    assert.equal(tabNameForKey("get-started", "ArrowRight"), "more-models");
    assert.equal(tabNameForKey("more-models", "ArrowRight"), "get-started", "ArrowRight wraps past the last tab");
    assert.equal(tabNameForKey("get-started", "ArrowLeft"), "more-models", "ArrowLeft wraps before the first tab");
    assert.equal(tabNameForKey("more-models", "Home"), "get-started");
    assert.equal(tabNameForKey("get-started", "End"), "more-models");
});

test("should_leave_unhandled_keys_to_the_browser_and_select_on_handled_keys", () => {
    const scriptContext = createLibraryContext();
    const libraryDocument = createLibraryDocument();
    scriptContext.document = libraryDocument;
    scriptContext.catalogDocument = { schema_version: 3, entries: [validCatalogEntry()] };
    vm.runInContext("libraryCatalogDocument = catalogDocument; renderLibraryCatalogDocument(catalogDocument)", scriptContext);

    let preventedDefaultCount = 0;
    scriptContext.keyboardEvent = { key: "Tab", preventDefault() { preventedDefaultCount += 1; } };
    vm.runInContext("handleLibraryTabKeydown(keyboardEvent, 'get-started')", scriptContext);
    assert.equal(preventedDefaultCount, 0, "Tab keeps moving focus out of the tablist");
    assert.equal(vm.runInContext("libraryActiveTab", scriptContext), "get-started");

    scriptContext.keyboardEvent = { key: "ArrowRight", preventDefault() { preventedDefaultCount += 1; } };
    vm.runInContext("handleLibraryTabKeydown(keyboardEvent, 'get-started')", scriptContext);
    assert.equal(preventedDefaultCount, 1);
    assert.equal(vm.runInContext("libraryActiveTab", scriptContext), "more-models");
});
