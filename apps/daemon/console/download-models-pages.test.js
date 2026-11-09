// Download Models page tests: Get started role cards, More models groups, fit labels
// against the serving memory ceiling, and keyboard access to the page tabs.

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

// Renders a catalog with an explicit page and serving memory ceiling, so each test
// states the exact inputs it depends on instead of inheriting defaults.
function renderDownloadModelsWithEntries(catalogEntries, { activeTab = "get-started", servingCeilingBytes = null } = {}) {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    scriptContext.catalogDocument = { schema_version: 3, entries: catalogEntries };
    vm.runInContext(
        `downloadModelsActiveTab = ${JSON.stringify(activeTab)};
         downloadModelsServingCeilingBytes = ${JSON.stringify(servingCeilingBytes)};
         downloadModelsCatalogDocument = catalogDocument;`,
        scriptContext
    );
    const renderedState = vm.runInContext(
        "renderDownloadModelsCatalogDocument(catalogDocument)",
        scriptContext
    );
    return { scriptContext, downloadModelsDocument, renderedState };
}

function renderedTextsOfClass(rootElement, className) {
    return findRenderedElementsByClass(rootElement, className).map((element) => element.textContent);
}

test("should_label_fit_at_the_exact_memory_boundary_and_stream_only_when_allowed", () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.catalogRow = { approximateMemoryBytes: 8_000_000_000, minimumMemoryBytes: null };
    assert.equal(
        vm.runInContext("downloadModelsFitLabel(catalogRow, 8_000_000_000)", scriptContext),
        "Runs in memory on this Mac",
        "memory equal to the ceiling still fits"
    );
    assert.equal(
        vm.runInContext("downloadModelsFitLabel(catalogRow, 7_999_999_999)", scriptContext),
        "May be slow on this Mac",
        "one byte over the ceiling no longer runs fully in memory"
    );
    scriptContext.catalogRow.minimumMemoryBytes = 4_000_000_000;
    assert.equal(
        vm.runInContext("downloadModelsFitLabel(catalogRow, 7_999_999_999)", scriptContext),
        "Streams from your SSD on this Mac (slower)",
        "a model with a measured streaming floor is offered the SSD path"
    );
    assert.equal(
        vm.runInContext("downloadModelsFitLabel(catalogRow, 4_000_000_000)", scriptContext),
        "Streams from your SSD on this Mac (slower)",
        "a ceiling exactly at the streaming floor still streams"
    );
    assert.equal(
        vm.runInContext("downloadModelsFitLabel(catalogRow, 3_999_999_999)", scriptContext),
        "May be slow on this Mac",
        "one byte under the streaming floor is labelled as possibly slow, never as refused"
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

test("should_judge_fit_against_the_lower_serving_ceiling_not_the_machine_maximum", () => {
    // The engine serves within the configured 32 GB, so a 40 GB model does not
    // run in memory even though the 40 GB machine could hold it.
    const scriptContext = createDownloadModelsContext();
    scriptContext.catalogRow = { approximateMemoryBytes: 40_000_000_000, minimumMemoryBytes: null };
    scriptContext.statusDocument = {
        mlx_memory_ceiling_bytes: 34_359_738_368,
        machine_mlx_memory_ceiling_bytes: 42_949_672_960
    };
    vm.runInContext("downloadModelsServingCeilingBytes = readDownloadModelsServingCeilingBytes(statusDocument)", scriptContext);
    assert.equal(
        vm.runInContext("downloadModelsFitLabel(catalogRow, downloadModelsServingCeilingBytes)", scriptContext),
        "May be slow on this Mac",
        "a model between the serving and machine ceilings must not be labelled as fitting"
    );
});

test("should_mark_each_fit_outcome_with_its_own_css_modifier_class", () => {
    const ceilingBytes = 4_000_000_000;
    const { downloadModelsDocument, renderedState } = renderDownloadModelsWithEntries([
        validCatalogEntry({ huggingface_id: "astronomical-test/fits", approximate_memory_bytes: 3_000_000_000 }),
        validCatalogEntry({ huggingface_id: "astronomical-test/streams", approximate_memory_bytes: 9_000_000_000, minimum_memory_bytes: 3_500_000_000 }),
        validCatalogEntry({ huggingface_id: "astronomical-test/slow", approximate_memory_bytes: 9_000_000_000 })
    ], { activeTab: "more-models", servingCeilingBytes: ceilingBytes });

    assert.equal(renderedState, "ready");
    const fitClasses = findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-fit-label")
        .map((fitLabel) => fitLabel.className);
    assert.deepEqual(fitClasses, [
        "download-models-fit-label download-models-fit-fits",
        "download-models-fit-label download-models-fit-streams",
        "download-models-fit-label download-models-fit-slow"
    ]);
});

test("should_render_get_started_picks_in_role_order_with_size_lines", () => {
    const { downloadModelsDocument } = renderDownloadModelsWithEntries([
        validCatalogEntry({ huggingface_id: "astronomical-test/image-pick", display_name: "Image pick", role: "image_generation", get_started: true, approximate_size_bytes: 3_221_225_472 }),
        validCatalogEntry({ huggingface_id: "astronomical-test/chat-pick", display_name: "Chat pick", role: "chat", get_started: true, approximate_size_bytes: 5_368_709_120 }),
        validCatalogEntry({ huggingface_id: "astronomical-test/chat-other", display_name: "Other chat", role: "chat" }),
        validCatalogEntry({ huggingface_id: "astronomical-test/coding-pick", display_name: "Coding pick", role: "coding", get_started: true, approximate_size_bytes: 2_147_483_648 })
    ]);

    const container = downloadModelsDocument.catalogContainer;
    assert.deepEqual(renderedTextsOfClass(container, "download-models-role-title"), ["Chat", "Code", "Image Generation"]);
    assert.deepEqual(renderedTextsOfClass(container, "download-models-role-size"), [
        "5.0 GB",
        "2.0 GB",
        "3.0 GB"
    ]);
    assert.deepEqual(renderedTextsOfClass(container, "download-models-role-model"), ["Chat pick", "Coding pick", "Image pick"]);
    assert.equal(
        renderedTexts(container).some((text) => text.includes("astronomical-test/")),
        false,
        "Get started must not show Hugging Face ids"
    );
});

test("should_skip_a_role_that_has_no_get_started_pick", () => {
    const { downloadModelsDocument } = renderDownloadModelsWithEntries([
        validCatalogEntry({ huggingface_id: "astronomical-test/chat-pick", display_name: "Chat pick", role: "chat", get_started: true }),
        validCatalogEntry({ huggingface_id: "astronomical-test/coding-other", display_name: "Coding other", role: "coding" }),
        validCatalogEntry({ huggingface_id: "astronomical-test/image-pick", display_name: "Image pick", role: "image_generation", get_started: true })
    ]);

    assert.deepEqual(
        renderedTextsOfClass(downloadModelsDocument.catalogContainer, "download-models-role-title"),
        ["Chat", "Image Generation"]
    );
});

test("should_group_more_models_and_keep_experimental_entries_only_in_experimental", () => {
    const { downloadModelsDocument } = renderDownloadModelsWithEntries([
        validCatalogEntry({ huggingface_id: "astronomical-test/chat-a", display_name: "Chat A", role: "chat" }),
        validCatalogEntry({ huggingface_id: "astronomical-test/helper-h", display_name: "Helper H", role: "helper" }),
        validCatalogEntry({ huggingface_id: "astronomical-test/coder-x", display_name: "Experimental coder", role: "coding", experimental: true }),
        validCatalogEntry({ huggingface_id: "astronomical-test/helper-x", display_name: "Experimental helper", role: "helper", experimental: true })
    ], { activeTab: "more-models" });

    const groupSections = findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-group");
    const groupSummaries = groupSections.map((groupSection) => ({
        heading: groupSection.children[0].textContent,
        models: findRenderedElementsByClass(groupSection, "download-models-model-card").map(
            (modelCard) => findRenderedElement(modelCard, (element) => element.tagName === "H4").textContent
        )
    }));
    assert.deepEqual(groupSummaries.map((groupSummary) => groupSummary.heading), [
        "Chat",
        "Helpers",
        "Experimental"
    ], "empty Code and Image Generation groups must be hidden");
    assert.equal(groupSummaries[1].models.length, 1, "Helpers holds only the non-experimental helper");
    assert.equal(groupSummaries[2].models.length, 2, "Experimental holds both experimental entries");
    assert.equal(
        findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-model-size").length,
        4,
        "every More models card shows its size line"
    );
});

test("should_keep_the_selected_page_across_a_poll_rerender", () => {
    const { scriptContext, downloadModelsDocument } = renderDownloadModelsWithEntries([validCatalogEntry()]);
    vm.runInContext('selectDownloadModelsTab("more-models")', scriptContext);

    const selectedTab = () => findRenderedElement(
        downloadModelsDocument.catalogContainer,
        (element) => element.id === "download-models-tab-more-models"
    );
    assert.equal(selectedTab().attributes["aria-selected"], "true");
    assert.equal(selectedTab().tabIndex, 0, "the selected tab stays in the Tab order");

    // The one-second poll re-renders with unchanged catalog input; the page must not revert.
    const pollOutcome = vm.runInContext("renderDownloadModelsCatalogDocument(catalogDocument)", scriptContext);
    assert.equal(pollOutcome, "ready");
    assert.equal(selectedTab().attributes["aria-selected"], "true");
    assert.notEqual(
        findRenderedElement(downloadModelsDocument.catalogContainer, (element) => element.className === "download-models-hero-summary"),
        undefined,
        "the More models hero must still be showing"
    );
});

test("should_render_the_unavailable_state_for_a_schema_version_2_catalog", () => {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    scriptContext.catalogDocument = { schema_version: 2, entries: [validCatalogEntry()] };

    assert.equal(
        vm.runInContext("renderDownloadModelsCatalogDocument(catalogDocument)", scriptContext),
        "unavailable"
    );
    assert.equal(downloadModelsDocument.catalogContainer.dataset.downloadModelsState, "unavailable");
});

test("should_map_arrow_home_and_end_keys_to_the_wrapping_tab_order", () => {
    const scriptContext = createDownloadModelsContext();
    const tabNameForKey = (currentTabName, pressedKey) => vm.runInContext(
        `downloadModelsTabNameForKey(${JSON.stringify(currentTabName)}, ${JSON.stringify(pressedKey)})`,
        scriptContext
    );
    assert.equal(tabNameForKey("get-started", "ArrowRight"), "more-models");
    assert.equal(tabNameForKey("more-models", "ArrowRight"), "get-started", "ArrowRight wraps past the last tab");
    assert.equal(tabNameForKey("get-started", "ArrowLeft"), "more-models", "ArrowLeft wraps before the first tab");
    assert.equal(tabNameForKey("more-models", "Home"), "get-started");
    assert.equal(tabNameForKey("get-started", "End"), "more-models");
});

test("should_leave_unhandled_keys_to_the_browser_and_select_on_handled_keys", () => {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    scriptContext.catalogDocument = { schema_version: 3, entries: [validCatalogEntry()] };
    vm.runInContext("downloadModelsCatalogDocument = catalogDocument; renderDownloadModelsCatalogDocument(catalogDocument)", scriptContext);

    let preventedDefaultCount = 0;
    scriptContext.keyboardEvent = { key: "Tab", preventDefault() { preventedDefaultCount += 1; } };
    vm.runInContext("handleDownloadModelsTabKeydown(keyboardEvent, 'get-started')", scriptContext);
    assert.equal(preventedDefaultCount, 0, "Tab keeps moving focus out of the tablist");
    assert.equal(vm.runInContext("downloadModelsActiveTab", scriptContext), "get-started");

    scriptContext.keyboardEvent = { key: "ArrowRight", preventDefault() { preventedDefaultCount += 1; } };
    vm.runInContext("handleDownloadModelsTabKeydown(keyboardEvent, 'get-started')", scriptContext);
    assert.equal(preventedDefaultCount, 1);
    assert.equal(vm.runInContext("downloadModelsActiveTab", scriptContext), "more-models");
});

test("should_name_a_user_set_limit_instead_of_promising_a_slow_run", () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.catalogRow = { approximateMemoryBytes: 8_000_000_000, minimumMemoryBytes: null };
    assert.equal(
        vm.runInContext("downloadModelsFitLabel(catalogRow, 4_000_000_000, true)", scriptContext),
        "Above the memory limit you set"
    );
    assert.equal(
        vm.runInContext("downloadModelsFitLabel(catalogRow, 4_000_000_000, false)", scriptContext),
        "May be slow on this Mac",
        "the Mac's own size never blocks a model"
    );
    scriptContext.catalogRow.minimumMemoryBytes = 3_000_000_000;
    assert.equal(
        vm.runInContext("downloadModelsFitLabel(catalogRow, 4_000_000_000, true)", scriptContext),
        "Streams from your SSD on this Mac (slower)",
        "a model that streams within the user's limit still streams"
    );
    assert.equal(
        vm.runInContext("downloadModelsFitLabel(catalogRow, 8_000_000_000, true)", scriptContext),
        "Runs in memory on this Mac"
    );
});

test("should_treat_only_a_configured_maximum_below_the_machine_as_a_user_limit", () => {
    const scriptContext = createDownloadModelsContext();
    const machineCeilingBytes = 48 * 1073741824;
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

test("should_put_the_download_size_on_the_title_line_beside_the_model_name", () => {
    const { downloadModelsDocument } = renderDownloadModelsWithEntries([
        validCatalogEntry({ huggingface_id: "astronomical-test/chat-a", display_name: "Chat A", approximate_size_bytes: 28_252_643_533 })
    ], { activeTab: "more-models" });

    const [titleLine] = findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-model-title-line");
    assert.ok(titleLine, "every More models card has a title line");
    assert.deepEqual(
        titleLine.children.map((child) => child.textContent),
        ["Chat A", "26.3 GB"],
        "the name comes first and the size sits right beside it"
    );
    assert.equal(titleLine.children[1].className, "download-models-model-size");
});

test("should_show_capabilities_in_one_fixed_order_each_with_its_own_colour_key", () => {
    const { downloadModelsDocument } = renderDownloadModelsWithEntries([
        validCatalogEntry({
            huggingface_id: "astronomical-test/everything",
            display_name: "Everything",
            capabilities: { supports_reasoning: true, supports_vision: true, supports_tool_calls: true, context_window: 262144, max_output_tokens: 4096 }
        }),
        validCatalogEntry({
            huggingface_id: "astronomical-test/no-vision",
            display_name: "No vision",
            capabilities: { supports_reasoning: true, supports_vision: false, supports_tool_calls: true, context_window: 32768, max_output_tokens: 4096 }
        })
    ], { activeTab: "more-models" });

    const badgeRows = findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-capability-badges").map(
        (badgeRow) => badgeRow.children.map((badge) => badge.dataset.capability)
    );
    assert.deepEqual(badgeRows.sort((first, second) => second.length - first.length), [
        ["reasoning", "vision", "tools", "context"],
        ["reasoning", "tools", "context"]
    ], "a missing capability is left out without reordering the rest, and context always comes last");
});

test("should_order_image_generation_before_embeddings_and_omit_context_a_model_does_not_report", () => {
    const { downloadModelsDocument } = renderDownloadModelsWithEntries([
        validCatalogEntry({
            huggingface_id: "astronomical-test/drawing-embedder",
            display_name: "Drawing embedder",
            capabilities: { supports_embeddings: true, supports_image_generation: true }
        })
    ], { activeTab: "more-models" });

    const badges = findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-capability-badges")[0].children;

    assert.deepEqual(badges.map((badge) => badge.dataset.capability), ["image-generation", "embeddings"]);
    assert.deepEqual(badges.map((badge) => badge.textContent), ["Image Generation", "Embeddings"]);
});

test("should_put_the_tags_and_the_buttons_on_one_footer_row_tags_first", () => {
    const { downloadModelsDocument } = renderDownloadModelsWithEntries([
        validCatalogEntry({
            huggingface_id: "astronomical-test/not-downloaded",
            display_name: "Not downloaded",
            capabilities: { supports_reasoning: true, context_window: 32768, max_output_tokens: 4096 }
        })
    ], { activeTab: "more-models" });

    const footer = findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-card-footer")[0];

    assert.deepEqual(footer.children.map((footerChild) => footerChild.className), [
        "download-models-capability-badges",
        "download-models-primary-actions"
    ]);
});

test("should_mark_only_the_download_button_for_the_download_icon", () => {
    const { downloadModelsDocument } = renderDownloadModelsWithEntries([
        validCatalogEntry({ huggingface_id: "astronomical-test/not-downloaded", display_name: "Not downloaded" })
    ], { activeTab: "more-models" });

    const actionButtons = findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-primary-actions")[0].children;
    const iconButtonLabels = actionButtons
        .filter((actionButton) => actionButton.className.includes("download-models-download-button"))
        .map((actionButton) => actionButton.textContent);

    assert.deepEqual(iconButtonLabels, ["Download"], "Details and other actions carry no download icon");
});

test("should_group_retiring_models_last_whatever_their_role_or_experimental_flag", () => {
    const { downloadModelsDocument } = renderDownloadModelsWithEntries([
        validCatalogEntry({ huggingface_id: "astronomical-test/staying-chat", display_name: "Staying chat", role: "chat" }),
        validCatalogEntry({ huggingface_id: "astronomical-test/retiring-chat", display_name: "Retiring chat", role: "chat", retiring: true }),
        validCatalogEntry({ huggingface_id: "astronomical-test/retiring-early", display_name: "Retiring early build", role: "coding", experimental: true, retiring: true })
    ], { activeTab: "more-models" });

    const groupTitles = findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-group-heading")
        .map((groupHeading) => groupHeading.textContent);
    const cardNamesByGroup = findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-group")
        .map((group) => findRenderedElementsByClass(group, "download-models-model-card")
            .map((card) => findRenderedElementsByClass(card, "download-models-model-title-line")[0].children[0].textContent));

    assert.equal(groupTitles[groupTitles.length - 1], "Retiring soon");
    assert.deepEqual(cardNamesByGroup[cardNamesByGroup.length - 1], ["Retiring chat", "Retiring early build"]);
    assert.deepEqual(cardNamesByGroup[0], ["Staying chat"], "the role group keeps only the model that stays");
});

test("should_not_create_a_retiring_group_when_no_model_is_retiring", () => {
    const { downloadModelsDocument } = renderDownloadModelsWithEntries([
        validCatalogEntry({ huggingface_id: "astronomical-test/staying-chat", display_name: "Staying chat", role: "chat" })
    ], { activeTab: "more-models" });

    const groupTitles = findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-group-heading")
        .map((groupHeading) => groupHeading.textContent);

    assert.equal(groupTitles.includes("Retiring soon"), false);
});

function availabilityBadgeTexts(downloadModelsDocument) {
    return findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-model-heading")
        .map((cardHeading) => findRenderedElementsByClass(cardHeading, "download-models-availability")
            .map((availabilityBadge) => availabilityBadge.textContent));
}

test("should_badge_a_downloaded_model_as_on_this_mac", () => {
    const { downloadModelsDocument } = renderDownloadModelsWithEntries([
        validCatalogEntry({ huggingface_id: "astronomical-test/downloaded", display_name: "Downloaded", ready_on_this_mac: true })
    ], { activeTab: "more-models" });

    assert.deepEqual(availabilityBadgeTexts(downloadModelsDocument), [["On this Mac"]]);
});

test("should_badge_a_model_whose_download_is_running_as_downloading", () => {
    const { downloadModelsDocument } = renderDownloadModelsWithEntries([
        validCatalogEntry({ huggingface_id: "astronomical-test/arriving", display_name: "Arriving", download_state: "downloading" })
    ], { activeTab: "more-models" });

    assert.deepEqual(availabilityBadgeTexts(downloadModelsDocument), [["Downloading"]]);
});

test("should_show_no_badge_for_a_model_that_is_only_downloadable", () => {
    const { downloadModelsDocument } = renderDownloadModelsWithEntries([
        validCatalogEntry({ huggingface_id: "astronomical-test/not-downloaded", display_name: "Not downloaded" })
    ], { activeTab: "more-models" });

    assert.deepEqual(availabilityBadgeTexts(downloadModelsDocument), [[]], "the Download button already says it can be downloaded");
});
