import assert from "node:assert/strict";
import { test } from "node:test";
import { assertFocusOn, assertNoElement, waitFor } from "./support/dom";
import { ENGINE_PATIENCE } from "../src/astro/EngineWatch";
import { bootJourney, MET_ASTRO_PROFILE, OWN_STATE_DIRECTORY, sendUserMessage, waitUntilReady } from "./support/journey";
import type { Journey } from "./support/journey";

// The get-started journey and Astro's home, driven the way a person would: through the
// real page wiring, a fake host that stores astro-profile.json, and a fake
// daemon with a Library.

const CHAT_MODEL_ID = "example-org/starter-chat-4bit";
const IMAGE_MODEL_ID = "example-org/starter-image-4bit";
const STARTER_CATALOG = {
  entries: [
    { huggingface_id: CHAT_MODEL_ID, role: "chat", get_started: true, approximate_size_bytes: 5_970_332_199, ready_on_this_mac: false },
    { huggingface_id: IMAGE_MODEL_ID, role: "image_generation", get_started: true, approximate_size_bytes: 4_619_700_724, ready_on_this_mac: false },
  ],
};

function element<T extends HTMLElement>(testID: string): T {
  const found = document.querySelector<T>(`[data-testid="${testID}"]`);
  if (found === null) {
    throw new Error(`no element with data-testid="${testID}"`);
  }
  return found;
}

function isPresent(testID: string): boolean {
  const found = document.querySelector<HTMLElement>(`[data-testid="${testID}"]`);
  return found !== null && !found.hidden;
}

function overlay(): HTMLElement {
  return element("get-started");
}

function astroSays(): string {
  return element("astro-says").textContent ?? "";
}

function composerField(): HTMLTextAreaElement {
  return element<HTMLTextAreaElement>("composer-input");
}

async function waitForStep(step: string): Promise<void> {
  await waitFor(() => !overlay().hidden && overlay().dataset.step === step, `the "${step}" step`);
}

async function getStarted(): Promise<void> {
  await waitForStep("welcome");
  element<HTMLButtonElement>("astro-get-started").click();
}

function bootNewcomer(options?: Parameters<typeof bootJourney>[1]): Journey {
  return bootJourney(undefined, { libraryCatalog: STARTER_CATALOG, ...options, astroProfile: null });
}

test("should_welcome_a_newcomer_and_land_them_in_an_empty_chat_when_a_chat_model_is_installed", async () => {
  const { host, daemon } = bootNewcomer();
  await waitForStep("welcome");
  assert.equal(document.querySelector(".get-started__headline")?.textContent, "Welcome to Astronomical");
  assert.equal(astroSays(), "Astronomical runs AI models privately on this Mac. Its assistant is called Astro.");
  element<HTMLButtonElement>("astro-get-started").click();
  await waitFor(() => overlay().hidden, "the journey to close");
  assert.equal(composerField().value, "", "nothing is typed for the person");
  assert.equal(document.activeElement?.id, "composer-input", "the caret is in the message field");
  assert.deepEqual(host.astroProfile, { nickname: "", hasFinishedFirstFlight: true, isAstroShown: true });
  assert.equal(document.querySelector(".astro-home__greeting")?.textContent, "Hey");
  assert.equal(composerField().placeholder, "Ask Astro anything");
  assert.deepEqual(daemon.requestedLibraryDownloads, [], "nothing downloads when a chat model is already here");
});

test("should_go_straight_home_for_someone_astro_already_met", async () => {
  const { daemon } = bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE });
  await waitFor(() => isPresent("astro-home"), "Astro's home");
  assert.equal(overlay().hidden, true);
  assert.equal(document.querySelector(".astro-home__greeting")?.textContent, "Hey, Maya");
  assert.equal(daemon.requests.some((request) => request.url.includes("/v1/library/download")), false, "nothing downloads on its own");
});

test("should_ask_no_name_and_no_style_on_the_way_in", async () => {
  bootNewcomer({ models: [] });
  await getStarted();
  await waitForStep("download");
  assertNoElement(overlay(), "input", "the journey asks for nothing to be typed");
  assert.doesNotMatch(overlay().textContent ?? "", /brain|sketchbook|grown-up|Explorer|Pilot/i);
});

test("should_show_a_saved_name_as_text_never_as_markup", async () => {
  bootJourney(undefined, { astroProfile: { ...MET_ASTRO_PROFILE, nickname: "<b>Bo</b>" } });
  await waitFor(() => isPresent("astro-home"), "Astro's home");
  assertNoElement(document, ".astro-home__greeting b");
  assert.equal(document.querySelector(".astro-home__greeting")?.textContent, "Hey, <b>Bo</b>");
});

test("should_end_the_journey_when_the_person_returns_with_a_model_downloaded_on_the_models_page", async () => {
  const { daemon } = bootNewcomer({ models: [] });
  await getStarted();
  await waitForStep("download");
  await waitFor(() => isPresent("astro-download"), "the Download button");
  element<HTMLButtonElement>("astro-other-model").click();
  daemon.installChatModelFromElsewhere();
  window.dispatchEvent(new Event("focus"));
  await waitFor(() => overlay().hidden, "the get-started journey to end");
  assert.deepEqual(daemon.requestedLibraryDownloads, [], "the first download is not started for someone who chose another model");
});

test("should_keep_offering_the_download_when_the_person_returns_and_no_model_is_installed", async () => {
  bootNewcomer({ models: [] });
  await getStarted();
  await waitForStep("download");
  await waitFor(() => isPresent("astro-download"), "the Download button");
  window.dispatchEvent(new Event("focus"));
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(overlay().hidden, false);
  assert.equal(overlay().dataset.step, "download");
  assert.equal(element("astro-download").textContent, "Download");
});

test("should_offer_the_recommended_model_by_name_and_size_with_one_download_button", async () => {
  const { daemon, host } = bootNewcomer({ models: [] });
  await getStarted();
  await waitForStep("download");
  await waitFor(() => isPresent("astro-download"), "the Download button");
  assert.equal(document.querySelector(".get-started__headline")?.textContent, "Download a model");
  assert.equal(element("astro-download-model").textContent, "starter-chat-4bit", "the model is named without its owner");
  assert.equal(element("astro-download-size").textContent, "5.6 GB, one time");
  assert.equal(element("astro-download").textContent, "Download");
  assert.equal(element("astro-continue").textContent, "Skip for now");
  assert.equal(isPresent("astro-download-next"), false, "what comes next waits for the download to start");
  assert.deepEqual(daemon.requestedLibraryDownloads, [], "nothing downloads until Download is pressed");
  element<HTMLButtonElement>("astro-other-model").click();
  assert.ok(host.receivedActions.some((action) => JSON.stringify(action).includes("openExternal") && JSON.stringify(action).includes("/download-models")), "the Models page opens for a different model");
});

test("should_keep_the_ready_screen_when_the_person_returns_after_downloading_here", async () => {
  bootNewcomer({
    models: [],
    libraryDownloadSequence: [{ state: "idle", huggingface_id: null, bytes_completed: 0, bytes_total: 0, error_code: null }],
    installsWhenDownloadFinishes: { huggingfaceID: CHAT_MODEL_ID, model: { id: "starter-chat-4bit", input_modalities: ["text"], supported_endpoints: ["/v1/chat/completions"] } },
  });
  await getStarted();
  await waitForStep("download");
  await waitFor(() => isPresent("astro-download"), "the Download button");
  element<HTMLButtonElement>("astro-download").click();
  await waitFor(() => astroSays() === "It is on this Mac now and works without the internet.", "the ready line");
  window.dispatchEvent(new Event("focus"));
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(overlay().hidden, false, "the screen that says the model is ready stays until the person moves on");
  assert.equal(element("astro-download").textContent, "Start chatting");
});

test("should_download_the_chat_model_and_make_chat_ready_when_it_lands", async () => {
  const { daemon } = bootNewcomer({
    models: [],
    libraryDownloadSequence: [
      { state: "downloading", huggingface_id: CHAT_MODEL_ID, bytes_completed: 2_000_000_000, bytes_total: 6_000_000_000, error_code: null },
      { state: "idle", huggingface_id: null, bytes_completed: 0, bytes_total: 0, error_code: null },
    ],
    installsWhenDownloadFinishes: { huggingfaceID: CHAT_MODEL_ID, model: { id: "starter-chat-4bit", input_modalities: ["text"], supported_endpoints: ["/v1/chat/completions"] } },
  });
  await getStarted();
  await waitForStep("download");
  await waitFor(() => isPresent("astro-download"), "the Download button");
  element<HTMLButtonElement>("astro-download").click();
  await waitFor(() => daemon.requestedLibraryDownloads.length === 1, "the download request");
  assert.deepEqual(daemon.requestedLibraryDownloads, [CHAT_MODEL_ID]);
  assert.equal(element("astro-continue").textContent, "Continue in the background");
  assert.match(element("astro-download-next").textContent ?? "", /Ask a question/);
  await waitFor(() => element("astro-download-size").textContent === "1.9 of 5.6 GB", "progress");
  await waitFor(() => astroSays() === "It is on this Mac now and works without the internet.", "the ready line");
  await waitFor(() => document.getElementById("empty-state-notice")?.textContent === "", "chat to become ready");
  assert.equal(element("astro-download").textContent, "Start chatting");
  assert.equal(isPresent("astro-continue"), false, "one way forward once the model is ready");
  assert.equal(document.querySelector(".get-started__headline")?.textContent, "Your model is ready");
  assert.match(element("astro-download-next").textContent ?? "", /^You can now:/);
  assert.equal(isPresent("astro-other-model"), false, "no second download is offered as this one lands");
  element<HTMLButtonElement>("astro-download").click();
  await waitFor(() => overlay().hidden, "the journey to close");
  assert.equal(document.activeElement?.id, "composer-input");
});

test("should_finish_without_downloading_when_the_newcomer_skips", async () => {
  const { host, daemon } = bootNewcomer({ models: [] });
  await getStarted();
  await waitForStep("download");
  await waitFor(() => isPresent("astro-continue"), "Skip for now");
  element<HTMLButtonElement>("astro-continue").click();
  await waitFor(() => overlay().hidden, "the journey to close");
  assert.equal((host.astroProfile as Record<string, unknown>)["hasFinishedFirstFlight"], true);
  assert.deepEqual(daemon.requestedLibraryDownloads, []);
});

test("should_say_plainly_when_another_download_holds_the_library", async () => {
  bootNewcomer({ models: [], libraryDownloadRefusal: { status: 409, code: "library_busy" } });
  await getStarted();
  await waitForStep("download");
  await waitFor(() => isPresent("astro-download"), "the Download button");
  element<HTMLButtonElement>("astro-download").click();
  await waitFor(() => /Another download is running/.test(astroSays()), "the busy line");
  assert.equal(element<HTMLButtonElement>("astro-download").textContent, "Try again");
  element<HTMLButtonElement>("astro-continue").click();
  await waitFor(() => overlay().hidden, "the journey to close");
});

test("should_say_voice_is_coming_on_home_and_fill_nothing", async () => {
  bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE });
  await waitFor(() => isPresent("astro-home"), "Astro's home");
  element<HTMLButtonElement>("astro-home-skill-voice").click();
  await waitFor(() => /coming soon/.test(document.querySelector(".astro-home__reply")?.textContent ?? ""), "the voice reply");
  assert.equal(composerField().value, "");
});

test("should_offer_the_image_model_before_drawing_and_take_no_for_an_answer", async () => {
  const { daemon } = bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE, libraryCatalog: STARTER_CATALOG });
  await waitFor(() => isPresent("astro-home"), "Astro's home");
  element<HTMLButtonElement>("astro-home-skill-pictures").click();
  await waitFor(() => isPresent("astro-home-get-image-model"), "the image model offer");
  assert.match(document.querySelector(".astro-home__reply")?.textContent ?? "", /image model/);
  assert.equal(element("astro-home-download-size").textContent, "4.3 GB, one time");
  element<HTMLButtonElement>("astro-home-not-now").click();
  await waitFor(() => !isPresent("astro-home-get-image-model"), "the offer to close");
  assert.deepEqual(daemon.requestedLibraryDownloads, [], "declining downloads nothing");
});

test("should_hide_and_bring_back_astro_with_the_focus_switch", async () => {
  const { host } = bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE });
  await waitFor(() => isPresent("astro-home"), "Astro's home");
  window.__thintalkCommands?.run({ name: "toggleAstro" });
  assert.equal(isPresent("astro-home"), false);
  assert.equal(document.getElementById("empty-state")?.classList.contains("empty-state--astro"), false, "the classic title is back");
  assert.equal(composerField().placeholder, "Message Astronomical");
  assert.equal((host.astroProfile as Record<string, unknown>)["isAstroShown"], false);
  window.__thintalkCommands?.run({ name: "toggleAstro" });
  assert.equal(isPresent("astro-home"), true);
});

test("should_fill_the_message_field_from_a_home_idea_without_sending_it", async () => {
  const { daemon } = bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE });
  await waitFor(() => isPresent("astro-home"), "Astro's home");
  await waitUntilReady();
  element<HTMLButtonElement>("astro-home-idea-0").click();
  assert.equal(composerField().value, "Explain black holes in three sentences");
  assertFocusOn(composerField());
  assert.equal(daemon.chatRequestBody(), null, "an idea only prepares the message");
});

test("should_offer_the_image_model_from_a_drawing_idea_when_none_is_installed", async () => {
  const { daemon } = bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE, libraryCatalog: STARTER_CATALOG });
  await waitFor(() => isPresent("astro-home"), "Astro's home");
  await waitUntilReady();
  element<HTMLButtonElement>("astro-home-idea-3").click();
  await waitFor(() => isPresent("astro-home-get-image-model"), "the image model offer");
  assert.equal(composerField().value, "", "a prompt that can only fail is not offered");
  assert.equal(daemon.chatRequestBody(), null);
});

test("should_fill_the_message_field_with_the_drawing_idea_once_the_offered_image_model_lands", async () => {
  const { daemon } = bootJourney(undefined, {
    astroProfile: MET_ASTRO_PROFILE,
    libraryCatalog: STARTER_CATALOG,
    libraryDownloadSequence: [{ state: "idle", huggingface_id: null, bytes_completed: 0, bytes_total: 0, error_code: null }],
    installsWhenDownloadFinishes: { huggingfaceID: IMAGE_MODEL_ID, model: { id: "starter-image-4bit", supported_endpoints: ["/v1/images/generations"] } },
  });
  await waitFor(() => isPresent("astro-home"), "Astro's home");
  await waitUntilReady();
  element<HTMLButtonElement>("astro-home-idea-3").click();
  await waitFor(() => isPresent("astro-home-get-image-model"), "the image model offer");
  assert.equal(composerField().value, "", "nothing is filled in while the model is missing");
  element<HTMLButtonElement>("astro-home-get-image-model").click();
  await waitFor(() => composerField().value === "Draw a poster of a rocket launch", "the idea to return once the model is here");
  assert.deepEqual(daemon.requestedLibraryDownloads, [IMAGE_MODEL_ID]);
  assert.equal(daemon.chatRequestBody(), null, "the idea is prepared, not sent");
});

test("should_fill_the_message_field_from_a_drawing_idea_when_an_image_model_is_installed", async () => {
  const installedModels = [
    { id: "starter-chat-4bit" },
    { id: "starter-image-4bit", supported_endpoints: ["/v1/images/generations"] },
  ];
  bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE, models: installedModels });
  await waitFor(() => isPresent("astro-home"), "Astro's home");
  await waitUntilReady();
  element<HTMLButtonElement>("astro-home-idea-3").click();
  await waitFor(() => composerField().value === "Draw a poster of a rocket launch", "the drawing prompt");
  assert.equal(isPresent("astro-home-get-image-model"), false);
});

test("should_keep_astro_hidden_when_hidden_during_the_journey", async () => {
  const { host } = bootNewcomer({ models: [] });
  await getStarted();
  await waitForStep("download");
  window.__thintalkCommands?.run({ name: "toggleAstro" });
  await waitFor(() => isPresent("astro-continue"), "Skip for now");
  element<HTMLButtonElement>("astro-continue").click();
  await waitFor(() => overlay().hidden, "the journey to close");
  assert.deepEqual(host.astroProfile, { nickname: "", hasFinishedFirstFlight: true, isAstroShown: false });
  assert.equal(isPresent("astro-home"), false);
});

test("should_move_keyboard_focus_into_each_step_and_keep_the_window_behind_inert", async () => {
  bootNewcomer({ models: [] });
  await waitForStep("welcome");
  assert.equal(document.activeElement?.id, "astro-get-started");
  assert.ok(document.getElementById("sidebar")?.hasAttribute("inert"), "the sidebar is out of reach behind the welcome");
  element<HTMLButtonElement>("astro-get-started").click();
  await waitForStep("download");
  await waitFor(() => document.activeElement?.id === "astro-download", "focus to reach the Download button once it is drawn");
});

test("should_skip_the_journey_with_escape_and_free_the_window", async () => {
  bootNewcomer();
  await waitForStep("welcome");
  overlay().dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  await waitFor(() => overlay().hidden, "the journey to close");
  assert.equal(document.getElementById("sidebar")?.hasAttribute("inert"), false);
});

test("should_add_a_starter_question_after_a_draft_instead_of_replacing_it", async () => {
  bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE });
  await waitFor(() => isPresent("astro-home"), "Astro's home");
  await waitUntilReady();
  composerField().value = "Hi Astro!";
  element<HTMLButtonElement>("astro-home-idea-0").click();
  assert.equal(composerField().value, "Hi Astro! Explain black holes in three sentences");
});

test("should_hide_the_library_notice_while_home_offers_a_chat_model_itself", async () => {
  bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE, models: [], libraryCatalog: STARTER_CATALOG });
  await waitFor(() => isPresent("astro-home-download"), "home's chat model offer");
  assert.ok(document.getElementById("empty-state")?.classList.contains("empty-state--astro-offers-chat-model"));
  assertNoElement(document, "[data-testid=astro-home-other-model]", "home keeps to the one recommended model");
});

test("should_wait_for_an_engine_still_starting_and_then_offer_the_download", async () => {
  const savedAskEvery = ENGINE_PATIENCE.askEveryMilliseconds;
  ENGINE_PATIENCE.askEveryMilliseconds = 10;
  try {
    const { daemon } = bootNewcomer({ models: [] });
    daemon.modelsUnreachable = true;
    await getStarted();
    await waitForStep("engineStarting");
    assert.equal(astroSays(), "One moment. Starting up.");
    daemon.modelsUnreachable = false;
    await waitForStep("download");
    await waitFor(() => isPresent("astro-download"), "the Download button");
  } finally {
    ENGINE_PATIENCE.askEveryMilliseconds = savedAskEvery;
  }
});

test("should_offer_a_restart_when_the_engine_never_starts_and_wait_again", async () => {
  const savedPatience = { ...ENGINE_PATIENCE };
  ENGINE_PATIENCE.askEveryMilliseconds = 10;
  ENGINE_PATIENCE.silentAfterMilliseconds = 40;
  try {
    const { daemon, host } = bootNewcomer({ models: [], canRestartEngine: true });
    daemon.modelsUnreachable = true;
    await getStarted();
    await waitForStep("engineSilent");
    assert.equal(astroSays(), "The engine didn't start. A restart usually fixes it.");
    assert.equal(element("astro-engine-retry").textContent, "Restart the engine");
    assert.equal(isPresent("astro-skip"), true, "a silent engine never traps the person on this screen");
    element<HTMLButtonElement>("astro-engine-retry").click();
    assert.ok(host.receivedActions.some((action) => action["action"] === "restartEngine"), "the host is asked to restart");
    await waitForStep("engineStarting");
    daemon.modelsUnreachable = false;
    await waitForStep("download");
  } finally {
    Object.assign(ENGINE_PATIENCE, savedPatience);
  }
});

test("should_tell_a_newcomer_another_copy_is_running_and_go_on_once_it_quits", async () => {
  const savedPatience = { ...ENGINE_PATIENCE };
  ENGINE_PATIENCE.askEveryMilliseconds = 10;
  // Long enough that only the engine's own answer can end the wait: the
  // screen must not sit on "Starting up" until patience runs out.
  ENGINE_PATIENCE.silentAfterMilliseconds = 60_000;
  try {
    // The other install's engine answers on the shared address and already
    // has a chat model, which must not be read as this app's.
    const { daemon, host } = bootNewcomer({ canRestartEngine: true, engineStateDirectory: "~/.astronomical" });
    await getStarted();
    await waitForStep("engineSilent");
    assert.equal(document.querySelector(".get-started__headline")?.textContent, "Astronomical is already running");
    assert.match(astroSays(), /Another copy of Astronomical is already running on this Mac/);
    assert.equal(element("astro-engine-retry").textContent, "Try again", "restarting this app's engine cannot free the address");
    assert.deepEqual(daemon.requestedLibraryDownloads, [], "nothing is downloaded into the other copy");
    daemon.reportedStateDirectory = OWN_STATE_DIRECTORY;
    element<HTMLButtonElement>("astro-engine-retry").click();
    assert.equal(host.receivedActions.some((action) => action["action"] === "restartEngine"), false);
    await waitFor(() => overlay().hidden, "the journey to finish with this app's own chat model");
  } finally {
    Object.assign(ENGINE_PATIENCE, savedPatience);
  }
});

test("should_offer_try_again_instead_of_restart_where_the_host_cannot_restart", async () => {
  const savedPatience = { ...ENGINE_PATIENCE };
  ENGINE_PATIENCE.askEveryMilliseconds = 10;
  ENGINE_PATIENCE.silentAfterMilliseconds = 40;
  try {
    const { daemon, host } = bootNewcomer({ models: [] });
    daemon.modelsUnreachable = true;
    await getStarted();
    await waitForStep("engineSilent");
    assert.equal(element("astro-engine-retry").textContent, "Try again");
    element<HTMLButtonElement>("astro-engine-retry").click();
    assert.equal(host.receivedActions.some((action) => action["action"] === "restartEngine"), false, "nothing is asked of a host that cannot restart");
    await waitForStep("engineStarting");
  } finally {
    Object.assign(ENGINE_PATIENCE, savedPatience);
  }
});

test("should_offer_a_chat_model_on_home_once_an_engine_that_was_starting_answers", async () => {
  const savedAskEvery = ENGINE_PATIENCE.askEveryMilliseconds;
  ENGINE_PATIENCE.askEveryMilliseconds = 10;
  try {
    const { daemon } = bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE, models: [], libraryCatalog: STARTER_CATALOG });
    daemon.modelsUnreachable = true;
    await waitFor(() => isPresent("astro-home"), "Astro's home");
    assert.equal(isPresent("astro-home-download"), false, "nothing to offer while the engine is silent");
    daemon.modelsUnreachable = false;
    await waitFor(() => isPresent("astro-home-download"), "home's chat model offer");
  } finally {
    ENGINE_PATIENCE.askEveryMilliseconds = savedAskEvery;
  }
});

/** Every hint the message field shows from now on, in order. The reply of the
 * fake engine is over within a few milliseconds, so the hint shown while it
 * streams is recorded as it changes rather than looked for afterwards. */
function recordPlaceholders(): string[] {
  const shownPlaceholders: string[] = [];
  const placeholderObserver = new window.MutationObserver(() => shownPlaceholders.push(composerField().placeholder));
  placeholderObserver.observe(composerField(), { attributes: true, attributeFilter: ["placeholder"] });
  return shownPlaceholders;
}

// The field says "Ask Astro anything" while idle, so it must not start calling
// the same assistant "Astronomical" the moment an answer begins to stream.
test("should_name_the_same_assistant_in_the_hint_while_an_answer_streams", async () => {
  const journeys = [
    { astroProfile: MET_ASTRO_PROFILE, queueingHint: "Add a message; Astro reads it next", idleHint: "Ask Astro anything" },
    { astroProfile: { ...MET_ASTRO_PROFILE, isAstroShown: false }, queueingHint: "Add a message; Astronomical reads it next", idleHint: "Message Astronomical" },
  ];
  for (const journey of journeys) {
    const { host } = bootJourney(undefined, { astroProfile: journey.astroProfile });
    await waitUntilReady();
    await waitFor(() => composerField().placeholder === journey.idleHint, `the idle hint "${journey.idleHint}"`);
    assert.equal(composerField().getAttribute("aria-label"), journey.idleHint, "VoiceOver hears the name the field shows");
    const shownPlaceholders = recordPlaceholders();

    sendUserMessage("What is mitosis?");
    await waitFor(() => host.fileCount() === 1, "the finished turn to be saved");
    await waitFor(() => composerField().placeholder === journey.idleHint, "the idle hint to come back");

    assert.ok(shownPlaceholders.includes(journey.queueingHint), `${JSON.stringify(shownPlaceholders)} should include "${journey.queueingHint}"`);
    assert.ok(
      !shownPlaceholders.some((placeholder) => placeholder.endsWith("reads it next") && placeholder !== journey.queueingHint),
      `no other assistant is named: ${JSON.stringify(shownPlaceholders)}`,
    );
  }
}, { timeout: 30_000 });
