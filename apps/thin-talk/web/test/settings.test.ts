import assert from "node:assert/strict";
import { test } from "node:test";
import { waitFor } from "./support/dom";
import { bootJourney, MET_ASTRO_PROFILE } from "./support/journey";
import type { Journey } from "./support/journey";

// Settings, driven the way a person would: Settings (⌘,) sends the host
// command, and every change goes through the real page wiring to a fake host
// that stores astro-profile.json. Focus checks compare ids, never elements:
// a failing assert on two happy-dom nodes spends minutes printing them.

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

function page(): HTMLElement {
  return element("settings");
}

function astroSays(): string {
  return element("settings-says").textContent ?? "";
}

function savedProfile(journey: Journey): Record<string, unknown> {
  return journey.host.astroProfile as Record<string, unknown>;
}

async function waitForStep(step: string): Promise<void> {
  await waitFor(() => !page().hidden && page().dataset.step === step, `the "${step}" screen`);
}

async function openSettings(profile: unknown = MET_ASTRO_PROFILE, options?: Parameters<typeof bootJourney>[1]): Promise<Journey> {
  const journey = bootJourney(undefined, { ...options, astroProfile: profile });
  await waitFor(() => journey.host.receivedRequests.some((request) => request.op === "loadAstroProfile"), "the profile to load");
  await waitFor(() => isPresent("astro-home") || (profile as { isAstroShown?: boolean }).isAstroShown === false, "the window to settle");
  window.__thintalkCommands?.run({ name: "openSettings" });
  await waitForStep("page");
  return journey;
}

function submitName(nickname: string): void {
  const nameField = element<HTMLInputElement>("settings-nickname");
  nameField.value = nickname;
  nameField.form!.dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
}

const SETTINGS_LINE = "These settings are kept on this Mac.";

test("should_open_settings_with_the_saved_name_and_plain_section_names", async () => {
  await openSettings();
  assert.equal(document.querySelector(".settings__title")?.textContent, "Settings");
  assert.equal(element("settings-nickname-shown").textContent, "Maya");
  assert.equal(element("settings-shown").getAttribute("aria-checked"), "true");
  assert.equal(astroSays(), SETTINGS_LINE);
  const sectionTitles = [...page().querySelectorAll(".settings__section-title")].map((title) => title.textContent);
  assert.deepEqual(sectionTitles, ["About you", "Models", "Appearance", "Privacy"]);
  assert.doesNotMatch(page().textContent ?? "", /brain|sketchbook|gifts|grown-up|Explorer|Pilot|Colours/i);
  assert.ok(document.getElementById("sidebar")?.hasAttribute("inert"), "the window behind is out of reach");
  assert.equal(document.activeElement?.id, "settings-done", "focus lands inside the page");
});

test("should_change_only_the_name_and_come_back_to_the_page", async () => {
  const journey = await openSettings();
  element<HTMLButtonElement>("settings-name-change").click();
  await waitForStep("name");
  assert.equal(element<HTMLInputElement>("settings-nickname").value, "Maya", "the field starts from the saved name");
  submitName("  Mia ");
  await waitForStep("page");
  assert.deepEqual(savedProfile(journey), { ...MET_ASTRO_PROFILE, nickname: "Mia" });
  assert.equal(element("settings-nickname-shown").textContent, "Mia");
  assert.equal(document.querySelector(".astro-home__greeting")?.textContent, "Hey, Mia", "home behind the page greets the new name");
});

test("should_keep_the_saved_name_when_the_change_is_cancelled", async () => {
  const journey = await openSettings();
  element<HTMLButtonElement>("settings-name-change").click();
  await waitForStep("name");
  element<HTMLInputElement>("settings-nickname").value = "Zed";
  element<HTMLButtonElement>("settings-name-cancel").click();
  await waitForStep("page");
  assert.equal(savedProfile(journey)["nickname"], "Maya");
  assert.equal(element("settings-nickname-shown").textContent, "Maya");
});

test("should_show_an_empty_name_as_not_set_and_clear_a_name_that_is_saved_empty", async () => {
  const journey = await openSettings({ ...MET_ASTRO_PROFILE, nickname: "" });
  assert.match(element("settings-nickname-shown").textContent ?? "", /Not set/);
  assert.equal(element("settings-name-change").textContent, "Add a name");
  element<HTMLButtonElement>("settings-name-change").click();
  await waitForStep("name");
  submitName("   ");
  await waitForStep("page");
  assert.equal(savedProfile(journey)["nickname"], "", "spaces alone are no name");
});

test("should_follow_the_view_menu_switch_while_the_page_is_open", async () => {
  const journey = await openSettings();
  window.__thintalkCommands?.run({ name: "toggleAstro" });
  await waitFor(() => element("settings-shown").getAttribute("aria-checked") === "false", "the switch to follow");
  element<HTMLButtonElement>("settings-shown").focus();
  element<HTMLButtonElement>("settings-shown").click();
  await waitFor(() => savedProfile(journey)["isAstroShown"] === true, "Astro to come back");
  assert.equal(element("settings-shown").getAttribute("aria-checked"), "true");
  assert.equal(document.activeElement?.id, "settings-shown", "focus stays on the switch after the redraw");
});

test("should_say_when_a_change_was_not_saved_and_save_it_on_try_again", async () => {
  const journey = await openSettings();
  journey.host.failsAstroProfileSaves = true;
  element<HTMLButtonElement>("settings-shown").click();
  await waitFor(() => isPresent("settings-save-failed"), "the failed-save alert");
  assert.equal(astroSays(), "That didn't save. Try again?");
  assert.equal(savedProfile(journey)["isAstroShown"], true, "the disk still holds the old profile");
  journey.host.failsAstroProfileSaves = false;
  element<HTMLButtonElement>("settings-retry-save").click();
  await waitFor(() => !isPresent("settings-save-failed"), "the alert to clear");
  assert.equal(savedProfile(journey)["isAstroShown"], false);
});

test("should_close_with_escape_from_the_page_and_back_out_of_an_answer_first", async () => {
  await openSettings();
  element<HTMLButtonElement>("settings-name-change").click();
  await waitForStep("name");
  page().dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  await waitForStep("page");
  page().dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  await waitFor(() => page().hidden, "the page to close");
  assert.equal(document.getElementById("sidebar")?.hasAttribute("inert"), false);
  assert.equal(document.activeElement?.id, "composer-input", "focus returns to the message field");
});

test("should_ignore_settings_while_the_get_started_journey_is_open", async () => {
  bootJourney(undefined, { astroProfile: null });
  await waitFor(() => !element("get-started").hidden, "the get-started journey");
  window.__thintalkCommands?.run({ name: "openSettings" });
  assert.equal(page().hidden, true, "the journey covers the window");
});

test("should_show_the_welcome_again_from_settings_and_bring_a_hidden_astro_back", async () => {
  const journey = await openSettings({ ...MET_ASTRO_PROFILE, isAstroShown: false });
  assert.equal(element("settings-replay").textContent, "Show");
  element<HTMLButtonElement>("settings-replay").click();
  await waitFor(() => !element("get-started").hidden && element("get-started").dataset.step === "welcome", "the welcome");
  assert.equal(page().hidden, true);
  assert.equal(savedProfile(journey)["isAstroShown"], true);
  assert.ok(document.getElementById("sidebar")?.hasAttribute("inert"), "the journey keeps the window behind it inert");
});

test("should_offer_a_chat_model_on_the_page_when_none_is_installed", async () => {
  await openSettings(MET_ASTRO_PROFILE, { models: [] });
  await waitFor(() => isPresent("settings-chat-model-shown"), "the chat model row");
  assert.match(element("settings-chat-model-shown").textContent ?? "", /Not installed/);
  await waitFor(() => isPresent("settings-model-download") || /download stopped/.test(astroSays()), "the chat model offer");
});

test("should_say_the_engine_is_silent_instead_of_offering_downloads", async () => {
  const journey = await openSettings();
  journey.daemon.modelsUnreachable = true;
  element<HTMLButtonElement>("settings-done").click();
  window.__thintalkCommands?.run({ name: "openSettings" });
  await waitFor(() => isPresent("settings-check-again"), "the engine-silent row");
  assert.equal(isPresent("settings-image-model"), false, "no image model offer while nothing can be checked");
  journey.daemon.modelsUnreachable = false;
  element<HTMLButtonElement>("settings-check-again").click();
  await waitFor(() => isPresent("settings-chat-model-shown") && element("settings-chat-model-shown").textContent === "Installed", "the chat model row to read installed");
  assert.equal(astroSays(), SETTINGS_LINE, "the engine line does not linger");
});

test("should_open_settings_asked_for_before_the_profile_is_read_once_it_is", async () => {
  bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE });
  window.__thintalkCommands?.run({ name: "openSettings" });
  assert.equal(page().hidden, true, "a page drawn now would show a newcomer's blank answers");
  await waitForStep("page");
  assert.equal(element("settings-nickname-shown").textContent, "Maya", "it opens with the saved answers");
});

test("should_drop_settings_asked_for_early_when_a_newcomer_journey_opens_instead", async () => {
  bootJourney(undefined, { astroProfile: null });
  window.__thintalkCommands?.run({ name: "openSettings" });
  await waitFor(() => !element("get-started").hidden, "the get-started journey");
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(page().hidden, true, "the journey covers the window");
});

test("should_open_the_privacy_note_from_the_page_and_come_back", async () => {
  await openSettings();
  element<HTMLButtonElement>("settings-privacy-read").click();
  await waitForStep("privacy");
  assert.match(page().textContent ?? "", /runs entirely on this Mac/);
  element<HTMLButtonElement>("settings-privacy-back").click();
  await waitForStep("page");
});

test("should_close_the_image_model_offer_on_not_now_and_download_nothing", async () => {
  const imageModelID = "example-org/starter-image-4bit";
  const journey = await openSettings(MET_ASTRO_PROFILE, {
    libraryCatalog: { entries: [{ huggingface_id: imageModelID, role: "image_generation", get_started: true, approximate_size_bytes: 4_619_700_724, ready_on_this_mac: false }] },
  });
  await waitFor(() => isPresent("settings-image-model"), "the image model row's offer");
  element<HTMLButtonElement>("settings-image-model").click();
  await waitFor(() => isPresent("settings-model-get-image-model"), "the image model offer");
  assert.match(astroSays(), /image model/);
  element<HTMLButtonElement>("settings-model-not-now").click();
  assert.equal(isPresent("settings-model-get-image-model"), false);
  assert.equal(astroSays(), SETTINGS_LINE);
  assert.deepEqual(journey.daemon.requestedLibraryDownloads, [], "declining downloads nothing");
});
