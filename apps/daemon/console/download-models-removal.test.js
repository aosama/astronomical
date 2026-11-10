// Download Models "Remove" tests: when the button is offered, what the
// confirmation says, how a refusal reaches the card, and that the request only
// names the model.

const {
    assert,
    createDownloadModelsContext,
    createDownloadModelsDocument,
    findRenderedElementsByClass,
    test,
    validCatalogEntry,
    vm
} = require("./download-models-test-support.js");

function renderMoreModels(catalogEntries, removalNotice = null, confirmingModelId = null, inFlightModelId = null) {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    scriptContext.catalogDocument = { schema_version: 4, entries: catalogEntries };
    scriptContext.removalNoticeForTest = removalNotice;
    scriptContext.confirmingModelIdForTest = confirmingModelId;
    scriptContext.inFlightModelIdForTest = inFlightModelId;
    vm.runInContext(
        `
         downloadModelsCatalogDocument = catalogDocument;
         downloadModelsRemovalNotice = removalNoticeForTest;
         downloadModelsRemovalConfirmingId = confirmingModelIdForTest;
         downloadModelsRemovalInFlightId = inFlightModelIdForTest;
         renderDownloadModelsCatalogDocument(catalogDocument);`,
        scriptContext
    );
    return downloadModelsDocument;
}

function actionLabels(downloadModelsDocument) {
    // A model that is not on this Mac has no line under On this Mac, so no actions.
    const installedActions = findRenderedElementsByClass(
        downloadModelsDocument.catalogContainer, "download-models-installed-actions"
    )[0];
    return installedActions ? installedActions.children.map((actionButton) => actionButton.textContent) : [];
}

const DOWNLOADED_BY_DOWNLOAD_MODELS = {
    huggingface_id: "astronomical-test/downloaded",
    display_name: "Downloaded",
    approximate_size_bytes: 1717986918,
    ready_on_this_mac: true,
    removable_from_disk: true
};

test("should_offer_remove_from_disk_for_a_model_download_models_installed", () => {
    const downloadModelsDocument = renderMoreModels([validCatalogEntry(DOWNLOADED_BY_DOWNLOAD_MODELS)]);

    assert.equal(actionLabels(downloadModelsDocument).includes("Remove"), true);
});

test("should_not_offer_remove_from_disk_for_a_copy_the_person_placed", () => {
    const downloadModelsDocument = renderMoreModels([
        validCatalogEntry({ ...DOWNLOADED_BY_DOWNLOAD_MODELS, removable_from_disk: false })
    ]);

    assert.equal(actionLabels(downloadModelsDocument).includes("Remove"), false);
});

test("should_not_offer_remove_from_disk_for_a_model_that_is_not_on_this_mac", () => {
    const downloadModelsDocument = renderMoreModels([
        validCatalogEntry({ ...DOWNLOADED_BY_DOWNLOAD_MODELS, ready_on_this_mac: false })
    ]);

    assert.equal(actionLabels(downloadModelsDocument).includes("Remove"), false);
});

test("should_name_the_model_its_size_and_the_trash_in_the_confirmation", () => {
    const scriptContext = createDownloadModelsContext();

    const confirmationText = vm.runInContext(
        `downloadModelsRemovalConfirmationText({ displayName: "Qwen 3.5 2B 4-bit", approximateSizeBytes: 1717986918 })`,
        scriptContext
    );

    assert.equal(confirmationText, "Move Qwen 3.5 2B 4-bit to the Trash? This frees 1.6 GB. You can download it again any time.");
});

test("should_ask_inside_the_card_once_remove_from_disk_is_clicked", () => {
    const downloadModelsDocument = renderMoreModels(
        [validCatalogEntry(DOWNLOADED_BY_DOWNLOAD_MODELS)],
        null,
        "astronomical-test/downloaded"
    );

    const labels = actionLabels(downloadModelsDocument);

    assert.equal(labels.includes("Remove"), false, "the question replaces the button");
    assert.equal(labels.includes("Move to Trash"), true);
    assert.equal(labels.includes("Keep"), true);
});

test("should_ask_only_on_the_card_whose_button_was_clicked", () => {
    const downloadModelsDocument = renderMoreModels([
        validCatalogEntry(DOWNLOADED_BY_DOWNLOAD_MODELS),
        validCatalogEntry({ ...DOWNLOADED_BY_DOWNLOAD_MODELS, huggingface_id: "astronomical-test/other", display_name: "Other" })
    ], null, "astronomical-test/other");

    const labelsByCard = findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-installed-actions")
        .map((actions) => actions.children.map((action) => action.textContent));

    assert.equal(labelsByCard[0].includes("Remove"), true);
    assert.equal(labelsByCard[1].includes("Move to Trash"), true);
});

test("should_never_use_the_browser_confirm_dialog_that_embedded_views_suppress", () => {
    const removalScript = require("node:fs").readFileSync(require("node:path").join(__dirname, "download-models-removal.js"), "utf8");

    assert.equal(/window\.confirm\(|\bconfirm\(/.test(removalScript.replace(/\/\/.*$/gm, "")), false);
});

test("should_send_only_the_model_identity_and_confirm_the_move_on_success", async () => {
    const scriptContext = createDownloadModelsContext();
    const sentRequests = [];
    scriptContext.fetch = async (requestUrl, requestOptions) => {
        sentRequests.push({ requestUrl, requestBody: JSON.parse(requestOptions.body) });
        return { ok: true, json: async () => ({ status: "removed" }) };
    };

    const removalNotice = await vm.runInContext(
        `removeDownloadModelsModelFromDisk({ huggingfaceId: "astronomical-test/downloaded", displayName: "Downloaded" })`,
        scriptContext
    );

    assert.deepEqual(JSON.parse(JSON.stringify(removalNotice)), {
        huggingfaceId: "astronomical-test/downloaded",
        message: "Downloaded was moved to the Trash.",
        movedToTrash: true
    });
    assert.deepEqual(JSON.parse(JSON.stringify(sentRequests)), [{
        requestUrl: "/v1/library/remove",
        requestBody: { huggingface_id: "astronomical-test/downloaded" }
    }]);
});

test("should_turn_a_refusal_into_the_daemon_message_for_that_model", async () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.fetch = async () => ({
        ok: false,
        json: async () => ({ error: { code: "model_loaded", message: "This model is loaded right now." } })
    });

    const removalNotice = await vm.runInContext(
        `removeDownloadModelsModelFromDisk({ huggingfaceId: "astronomical-test/downloaded", displayName: "Downloaded" })`,
        scriptContext
    );

    assert.deepEqual(JSON.parse(JSON.stringify(removalNotice)), {
        huggingfaceId: "astronomical-test/downloaded",
        message: "This model is loaded right now.",
        movedToTrash: false
    });
});

test("should_say_the_server_did_not_answer_when_the_request_fails", async () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.fetch = async () => { throw new Error("offline"); };

    const removalNotice = await vm.runInContext(
        `removeDownloadModelsModelFromDisk({ huggingfaceId: "astronomical-test/downloaded", displayName: "Downloaded" })`,
        scriptContext
    );

    assert.match(removalNotice.message, /did not answer/);
});

test("should_show_a_refusal_only_on_the_card_of_the_model_it_is_about", () => {
    const downloadModelsDocument = renderMoreModels([
        validCatalogEntry(DOWNLOADED_BY_DOWNLOAD_MODELS),
        validCatalogEntry({ ...DOWNLOADED_BY_DOWNLOAD_MODELS, huggingface_id: "astronomical-test/other", display_name: "Other" })
    ], { huggingfaceId: "astronomical-test/downloaded", message: "This model is loaded right now." });

    const cardNotices = findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-installed-line")
        .map((card) => findRenderedElementsByClass(card, "download-models-notice").map((detail) => detail.textContent));

    assert.deepEqual(cardNotices, [["This model is loaded right now."], []]);
});

test("should_redraw_the_card_when_only_the_removal_question_opens_in_a_page_already_drawn", () => {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    scriptContext.catalogDocument = { schema_version: 4, entries: [validCatalogEntry(DOWNLOADED_BY_DOWNLOAD_MODELS)] };
    vm.runInContext(
        `
         downloadModelsCatalogDocument = catalogDocument;
         renderDownloadModelsCatalogDocument(catalogDocument);
         downloadModelsRemovalConfirmingId = "astronomical-test/downloaded";
         renderDownloadModelsCatalogDocument(catalogDocument);`,
        scriptContext
    );

    assert.equal(actionLabels(downloadModelsDocument).includes("Move to Trash"), true);
});

test("should_say_the_removal_failed_when_the_refusal_is_not_json", async () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.fetch = async () => ({ ok: false, json: async () => { throw new SyntaxError("not json"); } });

    const removalNotice = await vm.runInContext(
        `removeDownloadModelsModelFromDisk({ huggingfaceId: "astronomical-test/downloaded", displayName: "Downloaded" })`,
        scriptContext
    );

    assert.equal(removalNotice.message, "The model could not be removed.");
});

test("should_show_a_disabled_moving_button_while_the_removal_runs", () => {
    const downloadModelsDocument = renderMoreModels(
        [validCatalogEntry(DOWNLOADED_BY_DOWNLOAD_MODELS)], null, null, "astronomical-test/downloaded"
    );

    const removalControls = findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-confirm-removal-button");

    assert.deepEqual(actionLabels(downloadModelsDocument).filter((label) => /Trash|Remove/.test(label)), ["Moving to Trash…"]);
    assert.equal(removalControls[0].disabled, true);
});

test("should_send_one_request_when_move_to_trash_is_clicked_twice", async () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.document = createDownloadModelsDocument();
    scriptContext.catalogDocument = { schema_version: 4, entries: [validCatalogEntry(DOWNLOADED_BY_DOWNLOAD_MODELS)] };
    let removalRequestCount = 0;
    let answerRemoval;
    scriptContext.fetch = async (requestUrl) => {
        if (requestUrl !== "/v1/library/remove") return { ok: true, json: async () => scriptContext.catalogDocument };
        removalRequestCount += 1;
        await new Promise((resolve) => { answerRemoval = resolve; });
        return { ok: true, json: async () => ({ status: "removed" }) };
    };
    vm.runInContext(`downloadModelsCatalogDocument = catalogDocument;`, scriptContext);
    const catalogRow = { huggingfaceId: "astronomical-test/downloaded", displayName: "Downloaded" };
    scriptContext.catalogRowForTest = catalogRow;

    const firstClick = vm.runInContext(`confirmDownloadModelsRemoval(catalogRowForTest)`, scriptContext);
    const secondClick = vm.runInContext(`confirmDownloadModelsRemoval(catalogRowForTest)`, scriptContext);
    await secondClick;
    answerRemoval();
    await firstClick;

    assert.equal(removalRequestCount, 1);
    assert.equal(vm.runInContext(`downloadModelsRemovalInFlightId`, scriptContext), null);
});

test("should_describe_both_answers_by_the_question_and_focus_keep_when_asked", async () => {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    scriptContext.catalogDocument = { schema_version: 4, entries: [validCatalogEntry(DOWNLOADED_BY_DOWNLOAD_MODELS)] };
    vm.runInContext(
        `
         downloadModelsCatalogDocument = catalogDocument;
         renderDownloadModelsCatalogDocument(catalogDocument);
         askToRemoveDownloadModelsModel({ huggingfaceId: "astronomical-test/downloaded" });`,
        scriptContext
    );
    await Promise.resolve();

    const answers = findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-keep-button")
        .concat(findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-confirm-removal-button"));

    assert.equal(downloadModelsDocument.activeElement?.textContent, "Keep");
    assert.deepEqual(answers.map((answer) => answer.attributes["aria-describedby"]), [
        "download-models-removal-question", "download-models-removal-question"
    ]);
});

test("should_show_download_progress_rather_than_an_older_removal_notice", () => {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    scriptContext.catalogDocument = {
        schema_version: 4,
        entries: [validCatalogEntry({ ...DOWNLOADED_BY_DOWNLOAD_MODELS, ready_on_this_mac: false, removable_from_disk: false })]
    };
    vm.runInContext(
        `
         downloadModelsCatalogDocument = catalogDocument;
         downloadModelsCurrentDownload = { huggingface_id: "astronomical-test/downloaded", state: "downloading", bytes_total: 2, bytes_completed: 1 };
         downloadModelsRemovalNotice = { huggingfaceId: "astronomical-test/downloaded", message: "Downloaded was moved to the Trash." };
         renderDownloadModelsCatalogDocument(catalogDocument);`,
        scriptContext
    );

    const notices = findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-notice");
    const bannerTitles = findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-banner-title")
        .map((bannerTitle) => bannerTitle.textContent);

    assert.deepEqual(notices, []);
    assert.deepEqual(bannerTitles, ["Downloading Downloaded"]);
});

function renderWithState(catalogEntries, stateScript) {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    scriptContext.catalogDocument = { schema_version: 4, entries: catalogEntries };
    vm.runInContext(
        `
         downloadModelsCatalogDocument = catalogDocument;
         ${stateScript}
         renderDownloadModelsCatalogDocument(catalogDocument);`,
        scriptContext
    );
    return { scriptContext, downloadModelsDocument };
}

function cardDetailElements(downloadModelsDocument) {
    return findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-notice");
}

const MOVED_NOTICE = `{ huggingfaceId: "astronomical-test/downloaded", message: "Downloaded was moved to the Trash.", movedToTrash: true }`;

test("should_announce_a_removal_outcome_as_a_status", () => {
    const { downloadModelsDocument } = renderWithState(
        [validCatalogEntry({ ...DOWNLOADED_BY_DOWNLOAD_MODELS, ready_on_this_mac: false, removable_from_disk: false })],
        `downloadModelsRemovalNotice = ${MOVED_NOTICE};`
    );

    const details = cardDetailElements(downloadModelsDocument);

    assert.deepEqual(details.map((detail) => detail.textContent), ["Downloaded was moved to the Trash."]);
    assert.equal(details[0].attributes.role, "status");
});

test("should_hide_the_removal_notice_while_a_paused_download_of_the_model_holds_the_card", () => {
    const { downloadModelsDocument } = renderWithState(
        [validCatalogEntry({ ...DOWNLOADED_BY_DOWNLOAD_MODELS, ready_on_this_mac: false, removable_from_disk: false })],
        `downloadModelsRemovalNotice = ${MOVED_NOTICE};
         downloadModelsCurrentDownload = { huggingface_id: "astronomical-test/downloaded", state: "paused", bytes_total: 2, bytes_completed: 1 };`
    );

    assert.equal(cardDetailElements(downloadModelsDocument).some((detail) => detail.textContent.includes("Trash")), false);
});

test("should_hide_moved_to_the_trash_once_the_model_is_on_this_mac_again", () => {
    const { downloadModelsDocument } = renderWithState(
        [validCatalogEntry(DOWNLOADED_BY_DOWNLOAD_MODELS)],
        `downloadModelsRemovalNotice = ${MOVED_NOTICE};`
    );

    assert.deepEqual(cardDetailElements(downloadModelsDocument), []);
});

test("should_keep_showing_a_refusal_while_the_model_is_still_on_this_mac", () => {
    const { downloadModelsDocument } = renderWithState(
        [validCatalogEntry(DOWNLOADED_BY_DOWNLOAD_MODELS)],
        `downloadModelsRemovalNotice = { huggingfaceId: "astronomical-test/downloaded", message: "This model is loaded right now.", movedToTrash: false };`
    );

    assert.deepEqual(cardDetailElements(downloadModelsDocument).map((detail) => detail.textContent), ["This model is loaded right now."]);
});

test("should_disable_other_cards_remove_buttons_while_one_removal_runs", () => {
    const { downloadModelsDocument } = renderWithState(
        [
            validCatalogEntry(DOWNLOADED_BY_DOWNLOAD_MODELS),
            validCatalogEntry({ ...DOWNLOADED_BY_DOWNLOAD_MODELS, huggingface_id: "astronomical-test/other", display_name: "Other" })
        ],
        `downloadModelsRemovalInFlightId = "astronomical-test/downloaded";`
    );

    const otherRemoveButtons = findRenderedElementsByClass(downloadModelsDocument.catalogContainer, "download-models-remove-button");

    assert.equal(otherRemoveButtons.length, 1);
    assert.equal(otherRemoveButtons[0].disabled, true);
});

test("should_return_focus_to_remove_from_disk_when_keep_is_chosen", async () => {
    const { scriptContext, downloadModelsDocument } = renderWithState(
        [validCatalogEntry(DOWNLOADED_BY_DOWNLOAD_MODELS)],
        `downloadModelsRemovalConfirmingId = "astronomical-test/downloaded";`
    );

    vm.runInContext(`cancelDownloadModelsRemoval();`, scriptContext);
    await Promise.resolve();

    assert.equal(downloadModelsDocument.activeElement?.textContent, "Remove");
});

test("should_focus_the_download_button_after_the_model_moves_to_the_trash", async () => {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    const removedEntry = validCatalogEntry({ ...DOWNLOADED_BY_DOWNLOAD_MODELS, ready_on_this_mac: false, removable_from_disk: false });
    scriptContext.catalogDocument = { schema_version: 4, entries: [validCatalogEntry(DOWNLOADED_BY_DOWNLOAD_MODELS)] };
    scriptContext.fetch = async (requestUrl) => requestUrl === "/v1/library/remove"
        ? { ok: true, json: async () => ({ status: "removed" }) }
        : { ok: true, json: async () => ({ schema_version: 4, entries: [removedEntry] }) };
    vm.runInContext(
        `
         downloadModelsCatalogDocument = catalogDocument;
         renderDownloadModelsCatalogDocument(catalogDocument);`,
        scriptContext
    );

    await vm.runInContext(
        `confirmDownloadModelsRemoval({ huggingfaceId: "astronomical-test/downloaded", displayName: "Downloaded" })`,
        scriptContext
    );
    await Promise.resolve();

    assert.equal(downloadModelsDocument.activeElement?.className.includes("download-models-download-button"), true);
});

test("should_finish_the_removal_when_the_catalog_refetch_fails", async () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.document = createDownloadModelsDocument();
    scriptContext.catalogDocument = { schema_version: 4, entries: [validCatalogEntry(DOWNLOADED_BY_DOWNLOAD_MODELS)] };
    scriptContext.fetch = async (requestUrl) => {
        if (requestUrl === "/v1/library/remove") return { ok: true, json: async () => ({ status: "removed" }) };
        throw new Error("offline");
    };
    vm.runInContext(`downloadModelsCatalogDocument = catalogDocument;`, scriptContext);

    await vm.runInContext(
        `confirmDownloadModelsRemoval({ huggingfaceId: "astronomical-test/downloaded", displayName: "Downloaded" })`,
        scriptContext
    );

    assert.equal(vm.runInContext(`downloadModelsRemovalInFlightId`, scriptContext), null);
    assert.equal(vm.runInContext(`downloadModelsRemovalNotice.movedToTrash`, scriptContext), true);
});
