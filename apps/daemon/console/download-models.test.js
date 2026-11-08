// Focused Download Models rendering behavior tests keep the Observatory suite bounded.

const {
    ASYNC_TEST_OPTIONS,
    assert,
    createDownloadModelsContext,
    createDownloadModelsDocument,
    createDownloadModelsElement,
    findRenderedElement,
    findRenderedElementsByClass,
    renderedTexts,
    test,
    validCatalogEntry,
    vm
} = require("./download-models-test-support.js");

// The hero, filter bar, and model cards live on More models, which is no longer
// the default page; tests that inspect them select it first, as a person would.
function showMoreModelsTab(scriptContext) {
    vm.runInContext('downloadModelsActiveTab = "more-models"', scriptContext);
}

// Renders the catalog the way loadDownloadModelsCatalog does and wires the
// controls, so a test can pretend a browser is ready for the user to type.
function createTypingSession() {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    showMoreModelsTab(scriptContext);
    scriptContext.catalogDocument = {
        schema_version: 3,
        entries: [
            validCatalogEntry(),
            validCatalogEntry({
                huggingface_id: "astronomical-test/example-flux",
                display_name: "Example flux",
                family: "flux2_klein"
            })
        ]
    };

    assert.equal(
        vm.runInContext(
            "renderDownloadModelsCatalogDocument(catalogDocument)",
            scriptContext
        ),
        "ready"
    );
    // Simulate what loadDownloadModelsCatalog does — it is the only code path that
    // assigns downloadModelsCatalogDocument in production.
    vm.runInContext("downloadModelsCatalogDocument = catalogDocument", scriptContext);
    // A real browser wires the handlers one tick after the first render,
    // well before the user types the first character.
    vm.runInContext("wireDownloadModelsControls()", scriptContext);

    const currentSearchInput = () => findRenderedElement(
        downloadModelsDocument.catalogContainer,
        (element) => element.id === "download-models-search"
    );
    return { downloadModelsDocument, currentSearchInput };
}

test("shows where new downloads are saved in the hero summary", () => {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    showMoreModelsTab(scriptContext);
    scriptContext.catalogDocument = {
        schema_version: 3,
        models_download_directory: "/shared/astronomical-models",
        entries: [validCatalogEntry()]
    };

    assert.equal(
        vm.runInContext(
            "renderDownloadModelsCatalogDocument(catalogDocument)",
            scriptContext
        ),
        "ready"
    );

    const heroSummary = findRenderedElement(
        downloadModelsDocument.catalogContainer,
        (renderedElement) => renderedElement.className === "download-models-hero-summary"
    );
    const locationNote = heroSummary.children.find(
        (element) => element.className === "download-models-hero-location"
    );
    assert.ok(locationNote, "the hero should surface the download directory");
    assert.equal(
        locationNote.textContent,
        "New downloads are saved to /shared/astronomical-models"
    );
});

test("omits the hero download location when the catalog ships none", () => {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    showMoreModelsTab(scriptContext);
    scriptContext.catalogDocument = {
        schema_version: 3,
        entries: [validCatalogEntry()]
    };

    assert.equal(
        vm.runInContext(
            "renderDownloadModelsCatalogDocument(catalogDocument)",
            scriptContext
        ),
        "ready"
    );

    const heroSummary = findRenderedElement(
        downloadModelsDocument.catalogContainer,
        (renderedElement) => renderedElement.className === "download-models-hero-summary"
    );
    assert.equal(
        heroSummary.children.some(
            (element) => element.className === "download-models-hero-location"
        ),
        false,
        "no location line should render without a catalog directory"
    );
});

test("keeps the search input focused, valued, and caret-set while typing", () => {
    const { downloadModelsDocument, currentSearchInput } = createTypingSession();
    // The user clicks into the field and types — the exact flow the issue
    // report describes losing.
    const originalSearchInput = currentSearchInput();
    downloadModelsDocument.activeElement = originalSearchInput;
    originalSearchInput.value = "qwen";
    originalSearchInput.dispatch("input");

    const rebuiltSearchInput = currentSearchInput();
    assert.equal(rebuiltSearchInput.id, "download-models-search");
    assert.equal(downloadModelsDocument.activeElement, rebuiltSearchInput);
    assert.equal(downloadModelsDocument.lastFocusOptions.preventScroll, true);
    assert.equal(rebuiltSearchInput.value, "qwen");
    assert.deepEqual(rebuiltSearchInput.selectionRange, [4, 4]);
    // The catalog list now shows only the matching rows.
    const catalogList = findRenderedElement(
        downloadModelsDocument.catalogContainer,
        (renderedElement) => renderedElement.className === "download-models-catalog-list"
    );
    assert.equal(catalogList.children.length, 1);
});

test("restores the caret where the user placed it when typing mid-query", () => {
    const { downloadModelsDocument, currentSearchInput } = createTypingSession();
    const originalSearchInput = currentSearchInput();
    downloadModelsDocument.activeElement = originalSearchInput;
    // A click between characters leaves the caret there, and the browser
    // reports that position on the input event after a key press.
    originalSearchInput.value = "qwen3-max";
    originalSearchInput.setSelectionRange(2, 2);
    originalSearchInput.dispatch("input");

    const rebuiltSearchInput = currentSearchInput();
    assert.equal(rebuiltSearchInput.value, "qwen3-max");
    assert.deepEqual(rebuiltSearchInput.selectionRange, [2, 2]);
});

test("renders an empty catalog through the bounded empty state", () => {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    scriptContext.catalogDocument = { schema_version: 3, entries: [] };

    const renderedState = vm.runInContext(
        "renderDownloadModelsCatalogDocument(catalogDocument)",
        scriptContext
    );

    assert.equal(renderedState, "empty");
    assert.equal(downloadModelsDocument.catalogContainer.dataset.downloadModelsState, "empty");
    assert.equal(downloadModelsDocument.catalogContainer.children.length, 1);
    assert.equal(downloadModelsDocument.catalogContainer.children[0].attributes.role, "status");
});

test("retains catalog order and formats sizes as decimal SI gigabytes", () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.catalogDocument = {
        schema_version: 3,
        entries: [
            validCatalogEntry({ display_name: "First", approximate_size_bytes: 1_500_000_000 }),
            validCatalogEntry({
                huggingface_id: "astronomical-test/example-prism",
                display_name: "Second",
                family: "prism_hadamard_qwen35",
                approximate_size_bytes: 2_000_000_000
            })
        ]
    };

    const catalogRows = vm.runInContext(
        "downloadModelsCatalogRowsFromDocument(catalogDocument)",
        scriptContext
    );

    assert.deepEqual(JSON.parse(JSON.stringify(catalogRows)), [
        {
            huggingfaceId: "astronomical-test/example-qwen",
            displayName: "First",
            family: "qwen3_5",
            approximateSize: "1.40 GB",
            approximateSizeBytes: 1_500_000_000,
            readyOnThisMac: false,
            destinationDirectory: null,
            downloadState: null,
            description: "A test model for exercising Download Models rendering.",
            quantizationLabel: "6-bit affine (group 64)",
            upstreamLicense: "MIT",
            requestableModelId: null,
            approximateMemoryBytes: 6_000_000_000,
            role: "chat",
            getStarted: false,
            experimental: false,
            retiring: false,
            minimumMemoryBytes: null,
            supportsReasoning: true,
            supportsVision: false,
            supportsToolCalls: true,
            supportsImageGeneration: false,
            supportsEmbeddings: false,
            contextWindow: 32768,
            maxOutputTokens: 4096
        },
        {
            huggingfaceId: "astronomical-test/example-prism",
            displayName: "Second",
            family: "prism_hadamard_qwen35",
            approximateSize: "1.86 GB",
            approximateSizeBytes: 2_000_000_000,
            readyOnThisMac: false,
            destinationDirectory: null,
            downloadState: null,
            description: "A test model for exercising Download Models rendering.",
            quantizationLabel: "6-bit affine (group 64)",
            upstreamLicense: "MIT",
            requestableModelId: null,
            approximateMemoryBytes: 6_000_000_000,
            role: "chat",
            getStarted: false,
            experimental: false,
            retiring: false,
            minimumMemoryBytes: null,
            supportsReasoning: true,
            supportsVision: false,
            supportsToolCalls: true,
            supportsImageGeneration: false,
            supportsEmbeddings: false,
            contextWindow: 32768,
            maxOutputTokens: 4096
        }
    ]);
});

test("presents progress and only the controls valid for each durable download state", () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.document = createDownloadModelsDocument();
    scriptContext.catalogRow = {
        huggingfaceId: "astronomical-test/example-qwen",
        displayName: "Example model",
        family: "qwen3_5",
        approximateSize: "4.00 GB",
        readyOnThisMac: false,
        downloadState: "downloading"
    };
    vm.runInContext(
        `downloadModelsCurrentDownload = {
            state: "downloading",
            huggingface_id: "astronomical-test/example-qwen",
            bytes_completed: 7_500_000_000,
            bytes_total: 30_000_000_000
        }`,
        scriptContext
    );

    assert.equal(
        vm.runInContext("downloadModelsStateTitle(catalogRow)", scriptContext),
        "Downloading 25% · 6.98 GB of 27.94 GB"
    );
    assert.deepEqual(
        JSON.parse(JSON.stringify(vm.runInContext(
            "downloadModelsActionButtons(catalogRow).map(button => button.textContent)",
            scriptContext
        ))),
        ["Pause", "Cancel"]
    );
    assert.equal(vm.runInContext("formatDownloadModelsRate(27_500_000)", scriptContext), "27.5 MB/s");
    assert.equal(
        vm.runInContext("smoothDownloadModelsTransferRate(20_000_000, 4_000_000)", scriptContext),
        16_000_000
    );
    assert.equal(
        vm.runInContext("formatDownloadModelsRemainingTime(16 * 60)", scriptContext),
        "About 16 minutes left"
    );

    vm.runInContext(
        `downloadModelsCurrentDownload = {
            state: "publishing",
            huggingface_id: "astronomical-test/example-qwen",
            bytes_completed: 30_000_000_000,
            bytes_total: 30_000_000_000
        }`,
        scriptContext
    );
    assert.equal(
        vm.runInContext("downloadModelsStateTitle(catalogRow)", scriptContext),
        "Download complete. Adding it to your models…"
    );
    assert.deepEqual(
        JSON.parse(JSON.stringify(vm.runInContext(
            "downloadModelsActionButtons(catalogRow).map(button => button.textContent)",
            scriptContext
        ))),
        []
    );

    vm.runInContext(
        `downloadModelsCurrentDownload = {
            state: "failed",
            huggingface_id: "astronomical-test/example-qwen",
            error_code: "checksum_mismatch"
        }`,
        scriptContext
    );
    assert.equal(
        vm.runInContext("downloadModelsStateTitle(catalogRow)", scriptContext),
        "Downloaded files did not pass verification"
    );
    assert.deepEqual(
        JSON.parse(JSON.stringify(vm.runInContext(
            "downloadModelsActionButtons(catalogRow).map(button => button.textContent)",
            scriptContext
        ))),
        ["Resume", "Cancel"]
    );

    vm.runInContext(
        `downloadModelsCurrentDownload = {
            state: "failed",
            huggingface_id: "astronomical-test/example-qwen",
            error_code: "model_already_present"
        }`,
        scriptContext
    );
    assert.deepEqual(
        JSON.parse(JSON.stringify(vm.runInContext(
            "downloadModelsActionButtons(catalogRow).map(button => button.textContent)",
            scriptContext
        ))),
        ["Cancel"]
    );
    assert.match(
        vm.runInContext("downloadModelsStateTitle(catalogRow)", scriptContext),
        /move it elsewhere/i
    );
});

test("should not show another model's download state or controls on a row", () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.document = createDownloadModelsDocument();
    scriptContext.catalogRow = {
        huggingfaceId: "astronomical-test/example-qwen",
        displayName: "Example model",
        family: "qwen3_5",
        approximateSize: "4.00 GB",
        readyOnThisMac: false,
        downloadState: "failed"
    };
    const rowButtonLabels = () => JSON.parse(JSON.stringify(vm.runInContext(
        "downloadModelsActionButtons(catalogRow).map(button => button.textContent)",
        scriptContext
    )));

    // Before the first status poll, the catalog's own copy is all there is.
    assert.equal(vm.runInContext("downloadModelsRowDownloadState(catalogRow)", scriptContext), "failed");
    assert.deepEqual(rowButtonLabels(), ["Resume", "Cancel"]);

    // The failed download was replaced by another model's: this row's copy is stale.
    vm.runInContext(
        `downloadModelsCurrentDownload = {
            state: "downloading",
            huggingface_id: "astronomical-test/other-qwen",
            bytes_completed: 1_000,
            bytes_total: 4_000
        }`,
        scriptContext
    );
    assert.equal(
        vm.runInContext("downloadModelsStateTitle(catalogRow)", scriptContext),
        "Available to download"
    );
    assert.deepEqual(rowButtonLabels(), ["Download"]);
    assert.equal(
        vm.runInContext("downloadModelsActionButtons(catalogRow)[0].disabled", scriptContext),
        true
    );

    // Once the other model's download has failed, it no longer holds the download slot.
    vm.runInContext(
        `downloadModelsCurrentDownload = {
            state: "failed",
            huggingface_id: "astronomical-test/other-qwen",
            error_code: "checksum_mismatch"
        }`,
        scriptContext
    );
    assert.deepEqual(rowButtonLabels(), ["Download"]);
    assert.equal(
        vm.runInContext("downloadModelsActionButtons(catalogRow)[0].disabled", scriptContext),
        false
    );
    vm.runInContext(`downloadModelsCurrentDownload = { state: "paused", huggingface_id: "astronomical-test/other-qwen" }`, scriptContext);
    assert.equal(
        vm.runInContext("downloadModelsActionButtons(catalogRow)[0].disabled", scriptContext),
        true
    );
});

test("accepts the largest catalog size that JavaScript can represent exactly", () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.catalogEntry = validCatalogEntry({
        display_name: "Largest exact size",
        approximate_size_bytes: Number.MAX_SAFE_INTEGER
    });

    assert.equal(
        vm.runInContext("isRenderableDownloadModelsCatalogEntry(catalogEntry)", scriptContext),
        true
    );
    scriptContext.catalogEntry.approximate_size_bytes = Number.MAX_SAFE_INTEGER + 1;
    assert.equal(
        vm.runInContext("isRenderableDownloadModelsCatalogEntry(catalogEntry)", scriptContext),
        false
    );
});

test("renders catalog entries with safe document operations and an explicit download action", () => {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    showMoreModelsTab(scriptContext);
    scriptContext.catalogDocument = {
        schema_version: 3,
        entries: [
            validCatalogEntry({
                display_name: "<img src=x onerror=alert(1)>",
                approximate_size_bytes: 3_000_000_000
            })
        ]
    };

    const renderedState = vm.runInContext(
        "renderDownloadModelsCatalogDocument(catalogDocument)",
        scriptContext
    );
    const renderedTags = [];
    let renderedHeading = null;
    const pendingElements = [...downloadModelsDocument.catalogContainer.children];
    while (pendingElements.length > 0) {
        const renderedElement = pendingElements.shift();
        renderedTags.push(renderedElement.tagName);
        // Model names are H4 now; H3 is reserved for the More models group headings.
        if (renderedElement.tagName === "H4") renderedHeading = renderedElement;
        pendingElements.push(...renderedElement.children);
    }

    assert.equal(renderedState, "ready");
    assert.equal(downloadModelsDocument.catalogStatus.textContent, "Model catalog loaded.");
    assert.equal(renderedTags.includes("BUTTON"), true);
    // The filter bar uses a search input; no raw text inputs or anchors appear.
    assert.equal(renderedTags.includes("A"), false);
    assert.equal(
        renderedHeading.textContent,
        "<img src=x onerror=alert(1)>"
    );
});

test("renders daemon-authored catalog families without a client allowlist", () => {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    showMoreModelsTab(scriptContext);
    scriptContext.catalogDocument = {
        schema_version: 3,
        entries: [
            validCatalogEntry(),
            validCatalogEntry({
                huggingface_id: "astronomical-test/example-prism",
                family: "prism_hadamard_qwen35"
            }),
            validCatalogEntry({
                huggingface_id: "astronomical-test/example-flux",
                family: "flux2_klein"
            })
        ]
    };

    const renderedState = vm.runInContext(
        "renderDownloadModelsCatalogDocument(catalogDocument)",
        scriptContext
    );
    const catalogRows = vm.runInContext(
        "downloadModelsCatalogRowsFromDocument(catalogDocument)",
        scriptContext
    );

    assert.equal(renderedState, "ready");
    assert.deepEqual(
        JSON.parse(JSON.stringify(catalogRows.map((catalogRow) => catalogRow.family))),
        ["qwen3_5", "prism_hadamard_qwen35", "flux2_klein"]
    );
    const filterBar = findRenderedElement(
        downloadModelsDocument.catalogContainer,
        (renderedElement) => renderedElement.className === "download-models-filter-bar"
    );
    const familyFilter = filterBar.children[2];
    assert.deepEqual(
        familyFilter.children.map((option) => option.value),
        ["all", "qwen3_5", "prism_hadamard_qwen35", "flux2_klein"]
    );
});

test("surfaces ModernBERT embedding entries with their family and capability badge", () => {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    showMoreModelsTab(scriptContext);
    scriptContext.catalogDocument = {
        schema_version: 3,
        entries: [
            validCatalogEntry({
                huggingface_id: "astronomical-test/example-embedder",
                family: "modernbert",
                capabilities: { supports_embeddings: true }
            })
        ]
    };

    const renderedState = vm.runInContext(
        "renderDownloadModelsCatalogDocument(catalogDocument)",
        scriptContext
    );
    const catalogRows = vm.runInContext(
        "downloadModelsCatalogRowsFromDocument(catalogDocument)",
        scriptContext
    );

    assert.equal(renderedState, "ready");
    assert.equal(catalogRows[0].supportsEmbeddings, true);
    const visibleTexts = renderedTexts(downloadModelsDocument.catalogContainer);
    assert.equal(
        visibleTexts.some((text) => text.includes("Embeddings")),
        true,
        "the embedding capability badge should be rendered"
    );
    assert.equal(
        visibleTexts.some((text) => text.includes("ModernBERT")),
        true,
        "the ModernBERT family label should be rendered"
    );
});

test("does not offer the Thin Talk handoff for a ready image-generation-only model", () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.document = createDownloadModelsDocument();
    scriptContext.catalogRow = {
        huggingfaceId: "astronomical-test/example-flux",
        readyOnThisMac: true,
        requestableModelId: "example-flux",
        supportsImageGeneration: true
    };

    const actionLabels = vm.runInContext(
        "createDownloadModelsPrimaryActions(catalogRow, false).children.map(button => button.textContent)",
        scriptContext
    );

    assert.deepEqual(JSON.parse(JSON.stringify(actionLabels)), ["Details"]);
});

function primaryActionLabelsFor(catalogRow) {
    const scriptContext = createDownloadModelsContext();
    scriptContext.document = createDownloadModelsDocument();
    scriptContext.catalogRow = catalogRow;
    const actionLabels = vm.runInContext(
        "createDownloadModelsPrimaryActions(catalogRow, false).children.map(button => button.textContent)",
        scriptContext
    );
    return JSON.parse(JSON.stringify(actionLabels));
}

test("should_not_offer_the_thin_talk_handoff_for_a_ready_embedding_model", () => {
    const actionLabels = primaryActionLabelsFor({
        huggingfaceId: "astronomical-test/example-embedder",
        readyOnThisMac: true,
        requestableModelId: "example-embedder",
        supportsEmbeddings: true
    });

    assert.deepEqual(actionLabels, ["Details"]);
});

test("should_offer_the_thin_talk_handoff_for_a_ready_chat_model", () => {
    const actionLabels = primaryActionLabelsFor({
        huggingfaceId: "astronomical-test/example-chat",
        readyOnThisMac: true,
        requestableModelId: "example-chat",
        supportsReasoning: true
    });

    assert.equal(actionLabels[0], "Open in Thin Talk");
});

test("should_not_offer_the_thin_talk_handoff_for_a_chat_model_not_yet_on_this_mac", () => {
    const actionLabels = primaryActionLabelsFor({
        huggingfaceId: "astronomical-test/example-chat",
        readyOnThisMac: false,
        requestableModelId: "example-chat",
        supportsReasoning: true
    });

    assert.equal(actionLabels.includes("Open in Thin Talk"), false);
});

function thinTalkHandoffsFromDownloadModels(applicationChannel) {
    const thinTalkHandoffs = [];
    const scriptContext = createDownloadModelsContext();
    scriptContext.document = createDownloadModelsDocument();
    scriptContext.document.body = { dataset: { applicationChannel } };
    scriptContext.window = {
        location: { assign: (handoffTarget) => thinTalkHandoffs.push(handoffTarget) }
    };
    // console.js owns the channel table and its own tests pin the values; this
    // stands in for it so these tests prove Download Models asks for this instance's
    // channel rather than naming a link of its own.
    scriptContext.thinTalkChatLink = (requestedChannel) =>
        requestedChannel === "development" ? "development-chat-link" : null;
    scriptContext.catalogRow = {
        huggingfaceId: "astronomical-test/example-qwen",
        readyOnThisMac: true,
        supportsImageGeneration: false
    };

    const primaryActions = vm.runInContext(
        "createDownloadModelsPrimaryActions(catalogRow, false)",
        scriptContext
    );
    const openInThinTalkButton = primaryActions.children.find(
        (actionButton) => actionButton.textContent === "Open in Thin Talk"
    );

    assert.notEqual(
        openInThinTalkButton,
        undefined,
        "a ready model should offer the Thin Talk handoff"
    );
    for (const clickListener of openInThinTalkButton.listeners.click) {
        clickListener();
    }
    return thinTalkHandoffs;
}

test("should_hand_a_ready_model_to_this_channels_thin_talk", () => {
    assert.deepEqual(thinTalkHandoffsFromDownloadModels("development"), ["development-chat-link"]);
});

test("should_not_hand_off_when_this_channel_has_no_thin_talk_link", () => {
    assert.deepEqual(thinTalkHandoffsFromDownloadModels("unknown"), []);
});

test("rejects an incomplete entry even when its visible fields are renderable", () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.catalogEntry = {
        display_name: "Incomplete model",
        family: "qwen3_5",
        approximate_size_bytes: 3_000_000_000
    };

    assert.equal(
        vm.runInContext("isRenderableDownloadModelsCatalogEntry(catalogEntry)", scriptContext),
        false
    );
});
