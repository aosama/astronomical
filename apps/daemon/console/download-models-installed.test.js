// Models page On this Mac tests: the banner of the download in flight, the line
// of each model already here, and the Thin Talk handoff.

const {
    assert,
    createDownloadModelsContext,
    createDownloadModelsDocument,
    findRenderedElementsByClass,
    test,
    validCatalogEntry,
    vm
} = require("./download-models-test-support.js");

const GIGABYTE = 1073741824;
const DOWNLOADING_MODEL_ID = "astronomical-test/example-qwen";

function renderWithDownload(currentDownload, catalogEntries = [validCatalogEntry({ model_id: "Talker" })]) {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    scriptContext.catalogDocument = { schema_version: 5, entries: catalogEntries };
    scriptContext.currentDownload = currentDownload;
    vm.runInContext(
        `downloadModelsCatalogDocument = catalogDocument;
         downloadModelsCurrentDownload = currentDownload;
         renderDownloadModelsCatalogDocument(catalogDocument)`,
        scriptContext
    );
    return downloadModelsDocument.catalogContainer;
}

function textsOfClass(rootElement, className) {
    return findRenderedElementsByClass(rootElement, className).map((element) => element.textContent);
}

function bannerOf(catalogContainer) {
    const banner = findRenderedElementsByClass(catalogContainer, "download-models-banner")[0];
    return {
        className: banner.className,
        title: textsOfClass(banner, "download-models-banner-title")[0],
        detail: textsOfClass(banner, "download-models-banner-detail")[0],
        controls: findRenderedElementsByClass(banner, "download-models-installed-actions")[0]
            .children.map((controlButton) => controlButton.textContent),
        progressBars: findRenderedElementsByClass(banner, "download-models-download-progress")
    };
}

test("should_name_the_model_show_progress_and_offer_pause_and_cancel_while_downloading", () => {
    const banner = bannerOf(renderWithDownload({
        huggingface_id: DOWNLOADING_MODEL_ID, state: "downloading",
        bytes_completed: 4 * GIGABYTE, bytes_total: 10 * GIGABYTE
    }));

    assert.equal(banner.title, "Downloading Talker");
    assert.equal(banner.detail, "4.0 of 10.0 GB");
    assert.deepEqual(banner.controls, ["Pause", "Cancel"]);
    assert.equal(banner.progressBars[0].attributes["aria-valuenow"], "40");
    assert.equal(banner.progressBars[0].attributes["aria-label"], "Talker download");
});

test("should_reassure_that_nothing_is_lost_and_offer_resume_while_paused", () => {
    const banner = bannerOf(renderWithDownload({
        huggingface_id: DOWNLOADING_MODEL_ID, state: "paused",
        bytes_completed: 4 * GIGABYTE, bytes_total: 10 * GIGABYTE
    }));

    assert.equal(banner.title, "Talker is paused");
    assert.equal(banner.detail, "4.0 of 10.0 GB is saved. Nothing is lost.");
    assert.deepEqual(banner.controls, ["Resume", "Cancel"]);
});

test("should_offer_no_control_and_a_full_bar_while_the_finished_download_is_checked", () => {
    for (const checkingState of ["verifying", "publishing"]) {
        const banner = bannerOf(renderWithDownload({
            huggingface_id: DOWNLOADING_MODEL_ID, state: checkingState,
            bytes_completed: 10 * GIGABYTE, bytes_total: 10 * GIGABYTE
        }));

        assert.equal(banner.title, "Almost ready: checking Talker", checkingState);
        assert.deepEqual(banner.controls, [], checkingState);
        assert.equal(banner.progressBars[0].attributes["aria-valuenow"], "100", checkingState);
    }
});

test("should_explain_a_stopped_download_by_its_cause_and_draw_it_as_a_problem", () => {
    const outOfSpace = bannerOf(renderWithDownload({
        huggingface_id: DOWNLOADING_MODEL_ID, state: "failed", error_code: "insufficient_disk"
    }));
    const droppedConnection = bannerOf(renderWithDownload({
        huggingface_id: DOWNLOADING_MODEL_ID, state: "failed", error_code: "something_unlisted"
    }));

    assert.equal(outOfSpace.title, "The Talker download stopped");
    assert.match(outOfSpace.detail, /enough free space/);
    assert.match(outOfSpace.className, /download-models-banner-failed/);
    assert.deepEqual(outOfSpace.controls, ["Try again", "Cancel"]);
    assert.match(droppedConnection.detail, /picks up where it left off/);
    assert.deepEqual(outOfSpace.progressBars, [], "a stopped download draws no progress bar");
});

// A model this version cannot run would be refused again on a retry, so its
// banner says so and offers only the way out.
test("should_say_this_version_cannot_run_the_model_and_offer_only_cancel", () => {
    const banner = bannerOf(renderWithDownload({
        huggingface_id: DOWNLOADING_MODEL_ID, state: "failed", error_code: "model_not_executable"
    }));

    assert.equal(banner.title, "The Talker download stopped");
    assert.equal(
        banner.detail, "This version of Astronomical cannot run this model. Cancel it to download another model."
    );
    assert.deepEqual(banner.controls, ["Cancel"]);
});

test("should_show_a_preparing_banner_without_a_bar_before_any_bytes_are_known", () => {
    const banner = bannerOf(renderWithDownload({ huggingface_id: DOWNLOADING_MODEL_ID, state: "checking_disk" }));

    assert.equal(banner.title, "Getting Talker ready to download");
    assert.deepEqual(banner.progressBars, []);
});

test("should_show_no_banner_when_nothing_is_downloading_or_the_model_is_already_here", () => {
    const idleContainer = renderWithDownload({ state: "idle" });
    const readyContainer = renderWithDownload(
        { huggingface_id: DOWNLOADING_MODEL_ID, state: "publishing", bytes_completed: 1, bytes_total: 1 },
        [validCatalogEntry({ model_id: "Talker", ready_on_this_mac: true })]
    );

    assert.deepEqual(findRenderedElementsByClass(idleContainer, "download-models-banner"), []);
    assert.deepEqual(findRenderedElementsByClass(readyContainer, "download-models-banner"), []);
});

test("should_put_the_banner_above_the_models_already_here_and_total_only_those", () => {
    const catalogContainer = renderWithDownload(
        { huggingface_id: DOWNLOADING_MODEL_ID, state: "downloading", bytes_completed: 1, bytes_total: 2 },
        [
            validCatalogEntry({ model_id: "Talker" }),
            validCatalogEntry({
                huggingface_id: "astronomical-test/here", model_id: "Here", ready_on_this_mac: true,
                approximate_size_bytes: 3 * GIGABYTE
            })
        ]
    );
    const installedLines = findRenderedElementsByClass(catalogContainer, "download-models-list")[0].children;

    assert.match(installedLines[0].className, /download-models-banner/);
    assert.match(installedLines[1].className, /download-models-installed-line/);
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-section-meta").slice(0, 1), ["1 model · 3.0 GB"]);
});

function installedActionLabelsFor(entryOverrides) {
    const catalogContainer = renderWithDownload(
        { state: "idle" }, [validCatalogEntry({ ready_on_this_mac: true, ...entryOverrides })]
    );
    return findRenderedElementsByClass(catalogContainer, "download-models-installed-actions")[0]
        .children.map((actionButton) => actionButton.textContent);
}

test("should_offer_the_thin_talk_handoff_for_a_chat_model_on_this_mac", () => {
    assert.deepEqual(installedActionLabelsFor({ removable_from_disk: true }), ["Chat with this model", "Remove"]);
});

test("should_not_offer_the_thin_talk_handoff_for_an_image_embedding_decision_or_speech_model", () => {
    const pictureModelLabels = installedActionLabelsFor({
        role: "image_generation", capabilities: { supports_image_generation: true }
    });
    const embeddingModelLabels = installedActionLabelsFor({
        role: "helper", capabilities: { supports_embeddings: true }
    });

    assert.equal(pictureModelLabels.includes("Chat with this model"), false);
    assert.equal(embeddingModelLabels.includes("Chat with this model"), false);
    const decisionModelLabels = installedActionLabelsFor({
        role: "decision", capabilities: { supports_decisions: true }
    });
    assert.equal(decisionModelLabels.includes("Chat with this model"), false);
    const speechModelLabels = installedActionLabelsFor({
        role: "speech", capabilities: { supports_speech: true }
    });
    assert.equal(speechModelLabels.includes("Chat with this model"), false);
});

test("should_show_a_copy_the_person_placed_in_finder_instead_of_offering_to_remove_it", () => {
    assert.deepEqual(installedActionLabelsFor({ removable_from_disk: false }), ["Chat with this model", "Show in Finder"]);
});

function thinTalkHandoffsFor(applicationChannel) {
    const thinTalkHandoffs = [];
    const scriptContext = createDownloadModelsContext();
    scriptContext.document = createDownloadModelsDocument();
    scriptContext.document.body = { dataset: { applicationChannel } };
    scriptContext.window = {
        location: { assign: (handoffTarget) => thinTalkHandoffs.push(handoffTarget) }
    };
    // console.js owns the channel table and its own tests pin the values; this
    // stands in for it so these tests prove the page asks for this instance's
    // channel rather than naming a link of its own.
    scriptContext.thinTalkChatLink = (requestedChannel) =>
        requestedChannel === "development" ? "development-chat-link" : null;
    const openInThinTalkButton = vm.runInContext("createDownloadModelsOpenInChatButton()", scriptContext);
    for (const clickListener of openInThinTalkButton.listeners.click) clickListener();
    return thinTalkHandoffs;
}

test("should_hand_off_to_this_channels_thin_talk", () => {
    assert.deepEqual(thinTalkHandoffsFor("development"), ["development-chat-link"]);
});

test("should_not_hand_off_when_this_channel_has_no_thin_talk_link", () => {
    assert.deepEqual(thinTalkHandoffsFor("unknown-channel"), []);
});
