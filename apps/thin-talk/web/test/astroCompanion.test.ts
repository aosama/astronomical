import assert from "node:assert/strict";
import { test } from "node:test";
import { waitFor } from "./support/dom";
import { ENGINE_PATIENCE } from "../src/astro/EngineWatch";
import { bootJourney, MET_ASTRO_PROFILE } from "./support/journey";
import type { Journey } from "./support/journey";

// Astro's first flight and home, driven the way a person would: through the
// real page wiring, a fake host that stores astro-profile.json, and a fake
// daemon with a Library.

const BRAIN_ID = "example-org/starter-chat-4bit";
const SKETCHBOOK_ID = "example-org/starter-image-4bit";
const STARTER_CATALOG = {
  entries: [
    { huggingface_id: BRAIN_ID, role: "chat", get_started: true, approximate_size_bytes: 5_970_332_199, ready_on_this_mac: false },
    { huggingface_id: SKETCHBOOK_ID, role: "image_generation", get_started: true, approximate_size_bytes: 4_619_700_724, ready_on_this_mac: false },
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
  return element("first-flight");
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

async function introduceSelf(nickname: string, styleTestID: string): Promise<void> {
  await waitForStep("hello");
  element<HTMLButtonElement>("astro-greet").click();
  await waitForStep("name");
  element<HTMLInputElement>("astro-nickname").value = nickname;
  const nameForm = element<HTMLInputElement>("astro-nickname").form!;
  nameForm.dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
  await waitForStep("style");
  element<HTMLButtonElement>(styleTestID).click();
}

function bootNewcomer(options?: Parameters<typeof bootJourney>[1]): Journey {
  return bootJourney(undefined, { libraryCatalog: STARTER_CATALOG, ...options, astroProfile: null });
}

test("should_welcome_a_newcomer_and_land_them_home_with_their_first_skill_ready_to_send", async () => {
  const { host } = bootNewcomer();
  await introduceSelf("  Sam  ", "astro-style-pilot");
  await waitForStep("skills");
  element<HTMLButtonElement>("astro-skill-chat").click();
  await waitFor(() => isPresent("astro-go-home"), "the way home");
  assert.match(astroSays(), /Chat's ready/);
  element<HTMLButtonElement>("astro-go-home").click();
  await waitFor(() => overlay().hidden, "the first flight to close");
  assert.equal(composerField().value, "Explain black holes in three sentences.");
  assert.deepEqual(host.astroProfile, { nickname: "Sam", style: "pilot", trail: "rainbow", hasFinishedFirstFlight: true, isAstroShown: true });
  assert.equal(document.querySelector(".astro-home__greeting")?.textContent, "Hey, Sam");
  assert.equal(composerField().placeholder, "Ask Astro anything");
});

test("should_go_straight_home_for_someone_astro_already_met", async () => {
  const { daemon } = bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE });
  await waitFor(() => isPresent("astro-home"), "Astro's home");
  assert.equal(overlay().hidden, true);
  assert.equal(document.querySelector(".astro-home__greeting")?.textContent, "Hi, Maya!");
  assert.equal(daemon.requests.some((request) => request.url.includes("/v1/library/download")), false, "nothing downloads on its own");
});

test("should_finish_with_nothing_prepared_when_the_newcomer_explores_on_their_own", async () => {
  const { host } = bootNewcomer();
  await waitForStep("hello");
  element<HTMLButtonElement>("astro-skip").click();
  await waitFor(() => overlay().hidden, "the first flight to close");
  assert.equal(composerField().value, "");
  assert.equal((host.astroProfile as Record<string, unknown>)["hasFinishedFirstFlight"], true);
});

test("should_keep_a_name_and_style_chosen_halfway_through_and_offer_them_on_replay", async () => {
  const first = bootNewcomer();
  await introduceSelf("Ada", "astro-style-explorer");
  await waitForStep("skills");
  assert.deepEqual(first.host.astroProfile, { nickname: "Ada", style: "explorer", trail: "rainbow", hasFinishedFirstFlight: false, isAstroShown: true });
  bootJourney(first);
  await waitForStep("hello");
  element<HTMLButtonElement>("astro-greet").click();
  await waitForStep("name");
  assert.equal(element<HTMLInputElement>("astro-nickname").value, "Ada", "an unfinished flight resumes with the saved name");
});

test("should_show_a_nickname_as_text_never_as_markup", async () => {
  bootNewcomer();
  await introduceSelf("<b>Bo</b>", "astro-style-explorer");
  await waitForStep("skills");
  assert.equal(document.querySelector(".first-flight__bubble b"), null);
  element<HTMLButtonElement>("astro-skip").click();
  await waitFor(() => overlay().hidden, "the first flight to close");
  assert.equal(document.querySelector(".astro-home__greeting")?.textContent, "Hi, <b>Bo</b>!");
});

test("should_detour_to_the_grown_up_note_and_come_back_to_the_style_choice", async () => {
  bootNewcomer();
  await introduceSelf("", "astro-style-grown-up");
  await waitForStep("grownUp");
  assert.match(overlay().textContent ?? "", /runs entirely on this Mac/);
  element<HTMLButtonElement>("astro-grown-up-back").click();
  await waitForStep("style");
  assert.equal(astroSays(), "Nice to meet you! How do you like your space?", "no name means no dangling comma");
});

test("should_say_voice_is_coming_and_keep_the_flight_open", async () => {
  bootNewcomer();
  await introduceSelf("", "astro-style-explorer");
  await waitForStep("skills");
  element<HTMLButtonElement>("astro-skill-voice").click();
  await waitFor(() => /coming soon/.test(astroSays()), "the voice reply");
  assert.equal(isPresent("astro-go-home"), false);
});

test("should_offer_the_sketchbook_before_drawing_and_take_no_for_an_answer", async () => {
  const { daemon } = bootNewcomer();
  await introduceSelf("", "astro-style-explorer");
  await waitForStep("skills");
  element<HTMLButtonElement>("astro-skill-pictures").click();
  await waitFor(() => isPresent("astro-get-sketchbook"), "the sketchbook offer");
  assert.match(astroSays(), /sketchbook/);
  assert.equal(element("astro-download-size").textContent, "4.6 GB, one time");
  element<HTMLButtonElement>("astro-not-now").click();
  await waitFor(() => !isPresent("astro-get-sketchbook"), "the offer to close");
  assert.deepEqual(daemon.requestedLibraryDownloads, [], "declining downloads nothing");
});

test("should_wake_astro_on_a_mac_without_a_brain_and_make_chat_ready_when_it_lands", async () => {
  const { daemon } = bootNewcomer({
    models: [],
    libraryDownloadSequence: [
      { state: "downloading", huggingface_id: BRAIN_ID, bytes_completed: 2_000_000_000, bytes_total: 6_000_000_000, error_code: null },
      { state: "idle", huggingface_id: null, bytes_completed: 0, bytes_total: 0, error_code: null },
    ],
    installsWhenDownloadFinishes: { huggingfaceID: BRAIN_ID, model: { id: "starter-chat-4bit", input_modalities: ["text"], supported_endpoints: ["/v1/chat/completions"] } },
  });
  await introduceSelf("", "astro-style-explorer");
  await waitForStep("wake");
  await waitFor(() => isPresent("astro-wake"), "the wake-up button");
  assert.equal(element("astro-download-size").textContent, "6.0 GB, one time");
  element<HTMLButtonElement>("astro-wake").click();
  await waitFor(() => daemon.requestedLibraryDownloads.length === 1, "the download request");
  assert.deepEqual(daemon.requestedLibraryDownloads, [BRAIN_ID]);
  await waitFor(() => element("astro-download-size").textContent === "2.0 of 6.0 GB", "progress");
  await waitFor(() => astroSays() === "I'm awake! Thank you!", "Astro to wake up");
  await waitFor(() => document.getElementById("empty-state-notice")?.textContent === "", "chat to become ready");
  element<HTMLButtonElement>("astro-wake").click();
  await waitForStep("skills");
});

test("should_save_a_trail_picked_while_the_brain_downloads", async () => {
  const { host } = bootNewcomer({ models: [] });
  await introduceSelf("", "astro-style-explorer");
  await waitForStep("wake");
  await waitFor(() => isPresent("astro-trail-sunset"), "the trail chips");
  element<HTMLButtonElement>("astro-trail-sunset").click();
  await waitFor(() => (host.astroProfile as Record<string, unknown>)["trail"] === "sunset", "the trail to save");
  assert.equal(element("astro-trail-sunset").getAttribute("aria-pressed"), "true");
  assert.ok(document.getElementById("app")?.classList.contains("astro-trail--sunset"), "every Astro takes the trail");
});

test("should_tell_a_newcomer_kindly_when_another_download_holds_the_library", async () => {
  bootNewcomer({ models: [], libraryDownloadRefusal: { status: 409, code: "library_busy" } });
  await introduceSelf("", "astro-style-explorer");
  await waitForStep("wake");
  await waitFor(() => isPresent("astro-wake"), "the wake-up button");
  element<HTMLButtonElement>("astro-wake").click();
  await waitFor(() => /Something else is downloading/.test(astroSays()), "the busy line");
  assert.equal(element<HTMLButtonElement>("astro-wake").textContent, "Try again");
  element<HTMLButtonElement>("astro-keep-going").click();
  await waitForStep("skills");
});

test("should_hide_and_bring_back_astro_with_the_focus_switch", async () => {
  const { host } = bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE });
  await waitFor(() => isPresent("astro-home"), "Astro's home");
  window.__thintalkCommands?.run({ name: "toggleAstro" });
  assert.equal(isPresent("astro-home"), false);
  assert.equal(document.getElementById("empty-state")?.classList.contains("empty-state--astro"), false, "the classic title is back");
  assert.equal(composerField().placeholder, "Message Thin Talk");
  assert.equal((host.astroProfile as Record<string, unknown>)["isAstroShown"], false);
  window.__thintalkCommands?.run({ name: "toggleAstro" });
  assert.equal(isPresent("astro-home"), true);
});

test("should_fill_the_message_field_from_a_home_idea_without_sending_it", async () => {
  const { daemon } = bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE });
  await waitFor(() => isPresent("astro-home-idea-0"), "the home ideas");
  element<HTMLButtonElement>("astro-home-idea-0").click();
  assert.equal(composerField().value, "Why is the sky blue?");
  assert.equal(document.activeElement, composerField());
  assert.equal(daemon.chatRequestBody(), null, "an idea only prepares the message");
});

test("should_keep_astro_hidden_when_hidden_during_the_first_flight", async () => {
  const { host } = bootNewcomer();
  await introduceSelf("Kim", "astro-style-explorer");
  await waitForStep("skills");
  window.__thintalkCommands?.run({ name: "toggleAstro" });
  element<HTMLButtonElement>("astro-skip").click();
  await waitFor(() => overlay().hidden, "the first flight to close");
  assert.deepEqual(host.astroProfile, { nickname: "Kim", style: "explorer", trail: "rainbow", hasFinishedFirstFlight: true, isAstroShown: false });
  assert.equal(isPresent("astro-home"), false);
});

test("should_move_keyboard_focus_into_each_step_and_keep_the_window_behind_inert", async () => {
  bootNewcomer();
  await waitForStep("hello");
  assert.equal(document.activeElement, element("astro-greet"));
  assert.ok(document.getElementById("sidebar")?.hasAttribute("inert"), "the sidebar is out of reach behind the welcome");
  element<HTMLButtonElement>("astro-greet").click();
  await waitForStep("name");
  assert.equal(document.activeElement, element("astro-nickname"));
  element<HTMLInputElement>("astro-nickname").form!.dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
  await waitForStep("style");
  assert.equal(document.activeElement, element("astro-style-explorer"), "focus follows to the style choice");
  element<HTMLButtonElement>("astro-style-explorer").click();
  await waitForStep("skills");
  assert.ok(overlay().contains(document.activeElement), "focus stays inside the welcome");
});

test("should_skip_the_first_flight_with_escape_and_free_the_window", async () => {
  bootNewcomer();
  await waitForStep("hello");
  overlay().dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  await waitFor(() => overlay().hidden, "the first flight to close");
  assert.equal(document.getElementById("sidebar")?.hasAttribute("inert"), false);
});

test("should_add_a_starter_question_after_a_draft_instead_of_replacing_it", async () => {
  bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE });
  await waitFor(() => isPresent("astro-home-idea-0"), "the home ideas");
  composerField().value = "Hi Astro!";
  element<HTMLButtonElement>("astro-home-idea-0").click();
  assert.equal(composerField().value, "Hi Astro! Why is the sky blue?");
});

test("should_hide_the_library_notice_while_home_offers_a_brain_itself", async () => {
  bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE, models: [], libraryCatalog: STARTER_CATALOG });
  await waitFor(() => isPresent("astro-home-wake"), "home's brain offer");
  assert.ok(document.getElementById("empty-state")?.classList.contains("empty-state--astro-offers-brain"));
});

test("should_wait_for_an_engine_still_starting_and_then_offer_the_brain", async () => {
  const savedAskEvery = ENGINE_PATIENCE.askEveryMilliseconds;
  ENGINE_PATIENCE.askEveryMilliseconds = 10;
  try {
    const { daemon } = bootNewcomer({ models: [] });
    daemon.modelsUnreachable = true;
    await introduceSelf("", "astro-style-explorer");
    await waitForStep("engineStarting");
    assert.equal(astroSays(), "Hang on… I'm just waking up!");
    daemon.modelsUnreachable = false;
    await waitForStep("wake");
    await waitFor(() => isPresent("astro-wake"), "the brain offer");
  } finally {
    ENGINE_PATIENCE.askEveryMilliseconds = savedAskEvery;
  }
});

test("should_offer_a_restart_when_the_engine_never_wakes_and_wait_again", async () => {
  const savedPatience = { ...ENGINE_PATIENCE };
  ENGINE_PATIENCE.askEveryMilliseconds = 10;
  ENGINE_PATIENCE.silentAfterMilliseconds = 40;
  try {
    const { daemon, host } = bootNewcomer({ models: [], canRestartEngine: true });
    daemon.modelsUnreachable = true;
    await introduceSelf("", "astro-style-explorer");
    await waitForStep("engineSilent");
    assert.equal(astroSays(), "I can't wake up right now. A restart usually fixes it!");
    assert.equal(element("astro-engine-retry").textContent, "Restart Astro's engine");
    element<HTMLButtonElement>("astro-engine-retry").click();
    assert.ok(host.receivedActions.some((action) => action["action"] === "restartEngine"), "the host is asked to restart");
    await waitForStep("engineStarting");
    daemon.modelsUnreachable = false;
    await waitForStep("wake");
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
    await introduceSelf("", "astro-style-explorer");
    await waitForStep("engineSilent");
    assert.equal(element("astro-engine-retry").textContent, "Try again");
    element<HTMLButtonElement>("astro-engine-retry").click();
    assert.equal(host.receivedActions.some((action) => action["action"] === "restartEngine"), false, "nothing is asked of a host that cannot restart");
    await waitForStep("engineStarting");
  } finally {
    Object.assign(ENGINE_PATIENCE, savedPatience);
  }
});

test("should_offer_a_brain_on_home_once_an_engine_that_was_starting_answers", async () => {
  const savedAskEvery = ENGINE_PATIENCE.askEveryMilliseconds;
  ENGINE_PATIENCE.askEveryMilliseconds = 10;
  try {
    const { daemon } = bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE, models: [], libraryCatalog: STARTER_CATALOG });
    daemon.modelsUnreachable = true;
    await waitFor(() => isPresent("astro-home"), "Astro's home");
    assert.equal(isPresent("astro-home-wake"), false, "nothing to offer while the engine is silent");
    daemon.modelsUnreachable = false;
    await waitFor(() => isPresent("astro-home-wake"), "home's brain offer");
  } finally {
    ENGINE_PATIENCE.askEveryMilliseconds = savedAskEvery;
  }
});
