// Models page coordinator tests: catalog validity, the search box across re-renders,
// and the controls each download state offers.

const {
    assert,
    createDownloadModelsContext,
    createDownloadModelsDocument,
    findRenderedElement,
    test,
    validCatalogEntry,
    vm
} = require("./download-models-test-support.js");

// Renders the catalog the way loadDownloadModelsCatalog does and wires the
// controls, so a test can pretend a browser is ready for the user to type.
function createTypingSession() {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    scriptContext.catalogDocument = {
        schema_version: 5,
        entries: [
            validCatalogEntry(),
            validCatalogEntry({
                huggingface_id: "astronomical-test/example-flux",
                model_id: "Example flux",
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
    scriptContext.catalogDocument = { schema_version: 5, entries: [] };

    const renderedState = vm.runInContext(
        "renderDownloadModelsCatalogDocument(catalogDocument)",
        scriptContext
    );

    assert.equal(renderedState, "empty");
    assert.equal(downloadModelsDocument.catalogContainer.dataset.downloadModelsState, "empty");
    assert.equal(downloadModelsDocument.catalogContainer.children.length, 1);
    assert.equal(downloadModelsDocument.catalogContainer.children[0].attributes.role, "status");
});

test("accepts the largest catalog size that JavaScript can represent exactly", () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.catalogEntry = validCatalogEntry({
        model_id: "Largest exact size",
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

test("rejects an incomplete entry even when its visible fields are renderable", () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.catalogEntry = {
        model_id: "Incomplete model",
        family: "qwen3_5",
        approximate_size_bytes: 3_000_000_000
    };

    assert.equal(
        vm.runInContext("isRenderableDownloadModelsCatalogEntry(catalogEntry)", scriptContext),
        false
    );
});

function downloadControlLabelsFor(downloadState, errorCode) {
    const scriptContext = createDownloadModelsContext();
    scriptContext.document = createDownloadModelsDocument();
    scriptContext.currentDownload = {
        huggingface_id: "astronomical-test/example-qwen", state: downloadState, error_code: errorCode
    };
    return JSON.parse(JSON.stringify(vm.runInContext(
        `downloadModelsCurrentDownload = currentDownload;
         downloadModelsActionButtons({ huggingfaceId: "astronomical-test/example-qwen", readyOnThisMac: false })
             .map((controlButton) => controlButton.textContent)`,
        scriptContext
    )));
}

test("should_offer_only_the_controls_valid_for_each_download_state", () => {
    assert.deepEqual(downloadControlLabelsFor("downloading"), ["Pause", "Cancel"]);
    assert.deepEqual(downloadControlLabelsFor("checking_disk"), ["Pause", "Cancel"]);
    assert.deepEqual(downloadControlLabelsFor("paused"), ["Resume", "Cancel"]);
    assert.deepEqual(downloadControlLabelsFor("failed", "checksum_mismatch"), ["Try again", "Cancel"]);
});

test("should_offer_no_control_while_a_finished_transfer_is_checked_or_when_idle", () => {
    assert.deepEqual(downloadControlLabelsFor("verifying"), []);
    assert.deepEqual(downloadControlLabelsFor("publishing"), []);
    assert.deepEqual(downloadControlLabelsFor("idle"), []);
});

test("should_offer_only_cancel_when_another_model_occupies_the_download_location", () => {
    assert.deepEqual(downloadControlLabelsFor("failed", "model_already_present"), ["Cancel"]);
});

// Trying again would be refused the same way, so no Try again is offered.
test("should_offer_only_cancel_when_this_version_cannot_run_the_model", () => {
    assert.deepEqual(downloadControlLabelsFor("failed", "model_not_executable"), ["Cancel"]);
});

test("should_explain_each_failure_that_trying_again_cannot_fix_and_fall_back_for_the_rest", () => {
    const scriptContext = createDownloadModelsContext();
    const explanationFor = (errorCode) => vm.runInContext(
        `downloadModelsFailureExplanation(${JSON.stringify(errorCode)})`, scriptContext
    );

    assert.equal(
        explanationFor("model_not_executable"),
        "This version of Astronomical cannot run this model. Cancel it to download another model."
    );
    assert.match(explanationFor("model_already_present"), /cancel this download/);
    assert.match(explanationFor("some_unlisted_code"), /picks up where it left off/);
    assert.match(explanationFor(undefined), /picks up where it left off/);
});

test("should_keep_get_usable_after_a_failure_and_busy_during_any_other_download_state", () => {
    const scriptContext = createDownloadModelsContext();
    const isSlotBusyIn = (downloadState) => vm.runInContext(
        `downloadModelsCurrentDownload = { state: ${JSON.stringify(downloadState)} }; isDownloadModelsDownloadSlotBusy()`,
        scriptContext
    );

    assert.equal(isSlotBusyIn("idle"), false);
    assert.equal(isSlotBusyIn("failed"), false, "the daemon discards a failed download when another starts");
    assert.equal(isSlotBusyIn("downloading"), true);
    assert.equal(isSlotBusyIn("paused"), true);
});

test("should_word_transfer_progress_with_time_left_only_once_the_rate_is_measured", () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.activeDownload = {
        state: "downloading", bytes_completed: 2 * 1073741824, bytes_total: 10 * 1073741824
    };

    const unmeasuredText = vm.runInContext("downloadModelsTransferProgressText(activeDownload)", scriptContext);
    const measuredText = vm.runInContext(
        `downloadModelsMeasuredBytesPerSecond = 8 * 1073741824 / 600;
         downloadModelsTransferProgressText(activeDownload)`,
        scriptContext
    );

    assert.equal(unmeasuredText, "2.0 of 10.0 GB");
    assert.equal(measuredText, "2.0 of 10.0 GB · about 10 minutes left");
});

test("should_read_the_model_id_and_summary_and_ignore_a_blank_summary", () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.catalogEntry = validCatalogEntry({
        huggingface_id: "astronomical-test/Example-9B-MLX-6bit", summary: "A quick everyday assistant."
    });
    const presentedRow = vm.runInContext("downloadModelsCatalogRowFromEntry(catalogEntry)", scriptContext);
    scriptContext.catalogEntry = validCatalogEntry({ summary: "  " });
    const plainRow = vm.runInContext("downloadModelsCatalogRowFromEntry(catalogEntry)", scriptContext);

    assert.deepEqual(
        [presentedRow.modelId, presentedRow.summary], ["Example-9B-MLX-6bit", "A quick everyday assistant."]
    );
    assert.equal(plainRow.summary, null);
    for (const retiredField of ["displayName", "shortName", "versionLabel"]) {
        assert.equal(retiredField in presentedRow, false, retiredField + " must not come back");
    }
});

test("should_refuse_a_catalog_entry_without_a_usable_model_id", () => {
    const scriptContext = createDownloadModelsContext();
    for (const unusableModelId of [undefined, null, "", "   ", 7]) {
        scriptContext.catalogEntry = validCatalogEntry({ model_id: unusableModelId });
        assert.equal(
            vm.runInContext("isRenderableDownloadModelsCatalogEntry(catalogEntry)", scriptContext),
            false,
            `model_id ${JSON.stringify(unusableModelId)} must not render`
        );
    }
    scriptContext.catalogEntry = validCatalogEntry();
    assert.equal(vm.runInContext("isRenderableDownloadModelsCatalogEntry(catalogEntry)", scriptContext), true);
});
