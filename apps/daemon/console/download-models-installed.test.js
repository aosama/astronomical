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

function renderWithDownload(currentDownload, catalogEntries = [validCatalogEntry({ model_id: "Talker" })], stateScript = "") {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    scriptContext.catalogDocument = { schema_version: 5, entries: catalogEntries };
    scriptContext.currentDownload = currentDownload;
    vm.runInContext(
        `downloadModelsCatalogDocument = catalogDocument;
         downloadModelsCurrentDownload = currentDownload;
         ${stateScript}
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
    assert.match(installedLines[1].className, /download-models-installed-header/);
    assert.match(installedLines[2].className, /download-models-installed-line/);
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-section-meta").slice(0, 1), ["1 model · 3.0 GB"]);
});

// The row's visible actions, then its ⋯ menu's items with the menu open.
function installedActionLabelsFor(entryOverrides) {
    const catalogEntry = validCatalogEntry({ ready_on_this_mac: true, ...entryOverrides });
    const catalogContainer = renderWithDownload(
        { state: "idle" }, [catalogEntry], `downloadModelsOpenMenuHuggingfaceId = ${JSON.stringify(catalogEntry.huggingface_id)};`
    );
    return {
        visible: findRenderedElementsByClass(catalogContainer, "download-models-installed-actions")[0]
            .children.map((actionButton) => actionButton.textContent),
        menu: textsOfClass(catalogContainer, "download-models-row-menu-item")
    };
}

test("should_offer_chat_finder_and_remove_in_one_order_for_a_chat_model_download_models_installed", () => {
    const actionLabels = installedActionLabelsFor({ removable_from_disk: true });
    assert.deepEqual(actionLabels.visible, ["Chat", "⋯"], "one visible action, the rest in the menu");
    assert.deepEqual(actionLabels.menu, ["Show in Finder", "Copy model ID", "Details", "Open on Hugging Face", "Move to Trash…"]);
});

test("should_offer_finder_on_every_kind_of_model_on_this_mac", () => {
    const speechModelLabels = installedActionLabelsFor({
        role: "speech", capabilities: { supports_speech: true }, removable_from_disk: true
    });
    assert.deepEqual(speechModelLabels.visible, ["⋯"]);
    assert.equal(speechModelLabels.menu.includes("Show in Finder"), true);
});

test("should_not_offer_the_thin_talk_handoff_for_an_image_embedding_decision_or_speech_model", () => {
    const pictureModelLabels = installedActionLabelsFor({
        role: "image_generation", capabilities: { supports_image_generation: true }
    });
    const embeddingModelLabels = installedActionLabelsFor({
        role: "helper", capabilities: { supports_embeddings: true }
    });

    assert.equal(pictureModelLabels.visible.includes("Chat"), false);
    assert.equal(embeddingModelLabels.visible.includes("Chat"), false);
    const decisionModelLabels = installedActionLabelsFor({
        role: "decision", capabilities: { supports_decisions: true }
    });
    assert.equal(decisionModelLabels.visible.includes("Chat"), false);
    const speechModelLabels = installedActionLabelsFor({
        role: "speech", capabilities: { supports_speech: true }
    });
    assert.equal(speechModelLabels.visible.includes("Chat"), false);
});

test("should_show_a_copy_the_person_placed_in_finder_without_offering_to_remove_it", () => {
    const placedCopyLabels = installedActionLabelsFor({ removable_from_disk: false });
    assert.equal(placedCopyLabels.menu.includes("Show in Finder"), true);
    assert.equal(placedCopyLabels.menu.includes("Move to Trash…"), false);
    assert.equal(placedCopyLabels.menu.at(-1), "Remove it in Finder instead");
});

function thinTalkHandoffsFor(applicationChannel, installedRow = { modelId: "Talker", requestableModelId: null }) {
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
    scriptContext.installedRow = installedRow;
    const openInThinTalkButton = vm.runInContext("createDownloadModelsOpenInChatButton(installedRow)", scriptContext);
    for (const clickListener of openInThinTalkButton.listeners.click) clickListener();
    return thinTalkHandoffs;
}

test("should_hand_off_to_this_channels_thin_talk", () => {
    assert.deepEqual(thinTalkHandoffsFor("development"), ["development-chat-link?model=Talker"]);
});

test("should_name_the_model_by_the_name_chat_lists_it_under_and_encode_it", () => {
    assert.deepEqual(
        thinTalkHandoffsFor("development", { modelId: "Talker", requestableModelId: "My Talker+v2" }),
        ["development-chat-link?model=My%20Talker%2Bv2"],
        "a copy under another folder name is asked for by that name"
    );
});

test("should_not_hand_off_when_this_channel_has_no_thin_talk_link", () => {
    assert.deepEqual(thinTalkHandoffsFor("unknown-channel"), []);
});


test("should_show_each_models_kind_icon_size_and_in_memory_status_under_column_names", () => {
    const catalogContainer = renderWithDownload(
        { state: "idle" },
        [
            validCatalogEntry({ huggingface_id: "astronomical-test/Talker", model_id: "Talker", ready_on_this_mac: true, approximate_size_bytes: 6 * GIGABYTE }),
            validCatalogEntry({ huggingface_id: "astronomical-test/Reader", model_id: "Reader", role: "speech", ready_on_this_mac: true, approximate_size_bytes: GIGABYTE / 2 })
        ],
        'loadedModelId = "Talker";'
    );

    assert.deepEqual(textsOfClass(catalogContainer, "download-models-installed-sort"), ["Model", "Kind ↑", "Size"]);
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-installed-name"), ["Talker", "Reader"]);
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-installed-kind"), ["Chat", "Read aloud"]);
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-installed-size"), ["6.0 GB", "0.5 GB"]);
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-in-memory"), ["In memory"], "only the loaded model");
    assert.equal(findRenderedElementsByClass(catalogContainer, "download-models-kind-icon-chat").length, 1);
    assert.equal(findRenderedElementsByClass(catalogContainer, "download-models-kind-icon-voice").length, 1);
});

test("should_sort_by_a_clicked_column_and_reverse_on_a_second_click", () => {
    const catalogContainer = renderWithDownload(
        { state: "idle" },
        [
            validCatalogEntry({ huggingface_id: "astronomical-test/Small", model_id: "Small", ready_on_this_mac: true, approximate_size_bytes: GIGABYTE }),
            validCatalogEntry({ huggingface_id: "astronomical-test/Big", model_id: "Big", ready_on_this_mac: true, approximate_size_bytes: 9 * GIGABYTE })
        ],
        'downloadModelsInstalledSortKey = "size"; downloadModelsInstalledSortAscending = false;'
    );

    assert.deepEqual(textsOfClass(catalogContainer, "download-models-installed-name"), ["Big", "Small"]);
    assert.deepEqual(textsOfClass(catalogContainer, "download-models-installed-sort-active"), ["Size ↓"]);
});

test("should_say_where_a_hugging_face_copy_lives_and_explain_it_once_under_the_list", () => {
    const catalogEntry = validCatalogEntry({
        ready_on_this_mac: true, removable_from_disk: false,
        destination_directory: "/Users/example/.cache/huggingface/hub/models--example--model/snapshots/abc"
    });
    const catalogContainer = renderWithDownload(
        { state: "idle" }, [catalogEntry], `downloadModelsOpenMenuHuggingfaceId = ${JSON.stringify(catalogEntry.huggingface_id)};`
    );

    assert.deepEqual(textsOfClass(catalogContainer, "download-models-installed-source"), ["From your Hugging Face folder"]);
    assert.equal(textsOfClass(catalogContainer, "download-models-row-menu-item").at(-1), "Managed in your Hugging Face folder");
    assert.equal(textsOfClass(catalogContainer, "download-models-installed-footnote").length, 1);
});

function renderIconsWithContext(catalogEntries, contextAdditions = {}) {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    Object.assign(scriptContext, contextAdditions);
    scriptContext.catalogDocument = { schema_version: 5, entries: catalogEntries };
    vm.runInContext(
        `downloadModelsCatalogDocument = catalogDocument; renderDownloadModelsCatalogDocument(catalogDocument)`,
        scriptContext
    );
    return { scriptContext, catalogContainer: downloadModelsDocument.catalogContainer };
}

const SPEECH_ENTRY = validCatalogEntry({
    huggingface_id: "astronomical-test/Reader", model_id: "Reader", role: "speech", ready_on_this_mac: true,
    capabilities: { supports_speech: true }
});

test("should_make_only_chat_and_read_aloud_icons_buttons_and_keep_the_rest_plain", () => {
    const { catalogContainer } = renderIconsWithContext([
        validCatalogEntry({ huggingface_id: "astronomical-test/Talker", model_id: "Talker", ready_on_this_mac: true }),
        SPEECH_ENTRY,
        validCatalogEntry({
            huggingface_id: "astronomical-test/Painter", model_id: "Painter", role: "image_generation", ready_on_this_mac: true,
            capabilities: { supports_image_generation: true }
        })
    ]);
    const kindIcons = findRenderedElementsByClass(catalogContainer, "download-models-kind-icon");

    assert.deepEqual(
        kindIcons.map((kindIcon) => [kindIcon.tagName, kindIcon.title || ""]),
        [["BUTTON", "Open in chat"], ["SPAN", ""], ["BUTTON", "Hear a sample"]],
        "ordered by kind: chat, pictures, read aloud"
    );
});

test("should_open_the_chat_on_the_model_when_its_chat_icon_is_clicked", () => {
    const chatLinks = [];
    const { scriptContext, catalogContainer } = renderIconsWithContext(
        [validCatalogEntry({ huggingface_id: "astronomical-test/Talker", model_id: "Talker", ready_on_this_mac: true })],
        {
            thinTalkChatLink: () => "development-chat-link",
            window: { location: { assign: (chatLink) => chatLinks.push(chatLink) } }
        }
    );
    scriptContext.document.body = { dataset: { applicationChannel: "development" } };
    const chatIcon = findRenderedElementsByClass(catalogContainer, "download-models-kind-icon-action")[0];

    for (const clickListener of chatIcon.listeners.click) clickListener();

    assert.deepEqual(chatLinks, ["development-chat-link?model=Talker"]);
});

test("should_load_then_play_a_voice_sample_and_stop_it_on_a_second_click", async () => {
    const playedSamples = [];
    const speechRequests = [];
    const { scriptContext, catalogContainer } = renderIconsWithContext([SPEECH_ENTRY], {
        fetch: async (requestUrl, requestOptions) => {
            speechRequests.push([requestUrl, JSON.parse(requestOptions.body)]);
            return { ok: true, blob: async () => "wav-bytes" };
        },
        URL: { createObjectURL: () => "blob:sample", revokeObjectURL: () => {} },
        Audio: class {
            constructor(sampleUrl) { this.src = sampleUrl; this.isPaused = false; }
            addEventListener() {}
            async play() { playedSamples.push(this.src); }
            pause() { this.isPaused = true; }
        }
    });
    const iconClasses = () => findRenderedElementsByClass(catalogContainer, "download-models-kind-icon")[0].className;

    const playing = vm.runInContext(`toggleDownloadModelsVoiceSample(downloadModelsCatalogRowsFromDocument(catalogDocument)[0])`, scriptContext);
    assert.match(iconClasses(), /download-models-kind-icon-loading/, "the icon shows it is working before any sound");
    await playing;

    assert.deepEqual(speechRequests, [["/v1/audio/speech", { model: "Reader", input: "Hello. This is how I sound when I read aloud." }]]);
    assert.match(playedSamples[0], /^data:audio\/wav;base64,/, "the click itself plays silence, so Safari allows the later sound");
    assert.deepEqual(playedSamples.slice(1), ["blob:sample"]);
    assert.match(iconClasses(), /download-models-kind-icon-playing/);

    await vm.runInContext(`toggleDownloadModelsVoiceSample(downloadModelsCatalogRowsFromDocument(catalogDocument)[0])`, scriptContext);
    assert.doesNotMatch(iconClasses(), /download-models-kind-icon-(playing|loading)/, "a second click stops it");
});

test("should_free_the_sample_audio_when_a_playing_sample_is_stopped", async () => {
    const revokedUrls = [];
    const { scriptContext } = renderIconsWithContext([SPEECH_ENTRY], {
        fetch: async () => ({ ok: true, blob: async () => "wav-bytes" }),
        URL: { createObjectURL: () => "blob:stopped-sample", revokeObjectURL: (freedUrl) => revokedUrls.push(freedUrl) },
        Audio: class {
            constructor(sampleUrl) { this.src = sampleUrl; }
            addEventListener() {}
            async play() {}
            pause() {}
        }
    });
    const toggleFirstRow = `toggleDownloadModelsVoiceSample(downloadModelsCatalogRowsFromDocument(catalogDocument)[0])`;
    await vm.runInContext(toggleFirstRow, scriptContext);
    assert.deepEqual(revokedUrls, [], "a playing sample keeps its audio");

    await vm.runInContext(toggleFirstRow, scriptContext);

    assert.deepEqual(revokedUrls, ["blob:stopped-sample"]);
});

test("should_say_in_the_tooltip_when_a_sample_could_not_play", async () => {
    const { scriptContext, catalogContainer } = renderIconsWithContext([SPEECH_ENTRY], {
        fetch: async () => ({ ok: false, blob: async () => null }),
        URL: { createObjectURL: () => "blob:none", revokeObjectURL: () => {} }
    });

    await vm.runInContext(`toggleDownloadModelsVoiceSample(downloadModelsCatalogRowsFromDocument(catalogDocument)[0])`, scriptContext);

    assert.equal(
        findRenderedElementsByClass(catalogContainer, "download-models-kind-icon")[0].title,
        "The sample could not play. Try again."
    );
});
