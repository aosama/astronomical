import assert from "node:assert/strict";
import { test } from "node:test";
import { waitFor } from "./support/dom";
import { bootJourney, MET_ASTRO_PROFILE } from "./support/journey";
import type { Journey } from "./support/journey";

// Your Astro, driven the way a person would: Settings (⌘,) sends the host
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
  return element("your-astro");
}

function astroSays(): string {
  return element("your-astro-says").textContent ?? "";
}

function savedProfile(journey: Journey): Record<string, unknown> {
  return journey.host.astroProfile as Record<string, unknown>;
}

async function waitForStep(step: string): Promise<void> {
  await waitFor(() => !page().hidden && page().dataset.step === step, `the "${step}" screen`);
}

async function openYourAstro(profile: unknown = MET_ASTRO_PROFILE, options?: Parameters<typeof bootJourney>[1]): Promise<Journey> {
  const journey = bootJourney(undefined, { ...options, astroProfile: profile });
  await waitFor(() => journey.host.receivedRequests.some((request) => request.op === "loadAstroProfile"), "the profile to load");
  await waitFor(() => isPresent("astro-home") || (profile as { isAstroShown?: boolean }).isAstroShown === false, "the window to settle");
  window.__thintalkCommands?.run({ name: "openYourAstro" });
  await waitForStep("page");
  return journey;
}

function submitName(nickname: string): void {
  const nameField = element<HTMLInputElement>("your-astro-nickname");
  nameField.value = nickname;
  nameField.form!.dispatchEvent(new window.Event("submit", { bubbles: true, cancelable: true }));
}

test("should_open_your_astro_from_settings_with_every_saved_answer_shown", async () => {
  await openYourAstro({ ...MET_ASTRO_PROFILE, trail: "aurora" });
  assert.equal(element("your-astro-nickname-shown").textContent, "Maya");
  assert.match(element("your-astro-about-you").textContent ?? "", /Explorer · big buttons/);
  assert.equal(element("your-astro-trail-aurora").getAttribute("aria-pressed"), "true");
  assert.equal(element("your-astro-shown").getAttribute("aria-checked"), "true");
  assert.equal(astroSays(), "Here's everything I remember about you, Maya. Change anything you like!");
  assert.ok(document.getElementById("sidebar")?.hasAttribute("inert"), "the window behind is out of reach");
  assert.equal(document.activeElement?.id, "your-astro-done", "focus lands inside the page");
});

test("should_change_only_the_name_and_come_back_to_the_page", async () => {
  const journey = await openYourAstro();
  element<HTMLButtonElement>("your-astro-name-change").click();
  await waitForStep("name");
  assert.equal(element<HTMLInputElement>("your-astro-nickname").value, "Maya", "the field starts from the saved name");
  submitName("  Mia ");
  await waitForStep("page");
  assert.deepEqual(savedProfile(journey), { ...MET_ASTRO_PROFILE, nickname: "Mia", trail: "rainbow" });
  assert.equal(element("your-astro-nickname-shown").textContent, "Mia");
  assert.equal(document.querySelector(".astro-home__greeting")?.textContent, "Hi, Mia!", "home behind the page greets the new name");
});

test("should_keep_the_saved_name_when_the_change_is_cancelled", async () => {
  const journey = await openYourAstro();
  element<HTMLButtonElement>("your-astro-name-change").click();
  await waitForStep("name");
  element<HTMLInputElement>("your-astro-nickname").value = "Zed";
  element<HTMLButtonElement>("your-astro-name-cancel").click();
  await waitForStep("page");
  assert.equal(savedProfile(journey)["nickname"], "Maya");
  assert.equal(element("your-astro-nickname-shown").textContent, "Maya");
});

test("should_show_an_empty_name_as_not_set_and_clear_a_name_that_is_saved_empty", async () => {
  const journey = await openYourAstro({ ...MET_ASTRO_PROFILE, nickname: "" });
  assert.match(element("your-astro-nickname-shown").textContent ?? "", /Not set/);
  assert.equal(element("your-astro-name-change").textContent, "Add a name");
  element<HTMLButtonElement>("your-astro-name-change").click();
  await waitForStep("name");
  submitName("   ");
  await waitForStep("page");
  assert.equal(savedProfile(journey)["nickname"], "", "spaces alone are no name");
  assert.equal(astroSays(), "Here's everything I remember about you. Change anything you like!", "no dangling comma without a name");
});

test("should_preview_a_style_without_saving_it_until_save", async () => {
  const journey = await openYourAstro();
  element<HTMLButtonElement>("your-astro-style-change").click();
  await waitForStep("style");
  assert.equal(element("your-astro-style-explorer").getAttribute("aria-pressed"), "true");
  element<HTMLButtonElement>("your-astro-style-pilot").click();
  assert.equal(astroSays(), "Pilot mode. Sleek and quick. Save it?", "Astro speaks in the previewed voice");
  assert.equal(element("your-astro-style-pilot").getAttribute("aria-pressed"), "true");
  assert.equal(savedProfile(journey)["style"], "explorer", "a preview saves nothing");
  element<HTMLButtonElement>("your-astro-style-cancel").click();
  await waitForStep("page");
  assert.equal(savedProfile(journey)["style"], "explorer");
  element<HTMLButtonElement>("your-astro-style-change").click();
  await waitForStep("style");
  element<HTMLButtonElement>("your-astro-style-pilot").click();
  element<HTMLButtonElement>("your-astro-style-save").click();
  await waitForStep("page");
  assert.equal(savedProfile(journey)["style"], "pilot");
  assert.ok(document.getElementById("app")?.classList.contains("app--astro-pilot"), "the window takes the Pilot look");
});

test("should_save_a_trail_and_draw_every_astro_with_it", async () => {
  const journey = await openYourAstro();
  element<HTMLButtonElement>("your-astro-trail-sunset").click();
  await waitFor(() => savedProfile(journey)["trail"] === "sunset", "the trail to save");
  assert.ok(document.getElementById("app")?.classList.contains("astro-trail--sunset"));
  element<HTMLButtonElement>("your-astro-trail-rainbow").click();
  await waitFor(() => savedProfile(journey)["trail"] === "rainbow", "the trail to save");
  assert.equal(document.getElementById("app")?.classList.contains("astro-trail--sunset"), false, "rainbow is the stylesheet's own palette");
});

test("should_follow_the_view_menu_switch_while_the_page_is_open", async () => {
  const journey = await openYourAstro();
  window.__thintalkCommands?.run({ name: "toggleAstro" });
  await waitFor(() => element("your-astro-shown").getAttribute("aria-checked") === "false", "the switch to follow");
  element<HTMLButtonElement>("your-astro-shown").focus();
  element<HTMLButtonElement>("your-astro-shown").click();
  await waitFor(() => savedProfile(journey)["isAstroShown"] === true, "Astro to come back");
  assert.equal(element("your-astro-shown").getAttribute("aria-checked"), "true");
  assert.equal(document.activeElement?.id, "your-astro-shown", "focus stays on the switch after the redraw");
});

test("should_say_when_a_change_was_not_saved_and_save_it_on_try_again", async () => {
  const journey = await openYourAstro();
  journey.host.failsAstroProfileSaves = true;
  element<HTMLButtonElement>("your-astro-trail-aurora").click();
  await waitFor(() => isPresent("your-astro-save-failed"), "the failed-save alert");
  assert.equal(astroSays(), "Uh-oh, I couldn't remember that. Let's try again!");
  assert.equal(savedProfile(journey)["trail"], undefined, "the disk still holds the old profile");
  journey.host.failsAstroProfileSaves = false;
  element<HTMLButtonElement>("your-astro-retry-save").click();
  await waitFor(() => !isPresent("your-astro-save-failed"), "the alert to clear");
  assert.equal(savedProfile(journey)["trail"], "aurora");
});

test("should_close_with_escape_from_the_page_and_back_out_of_an_answer_first", async () => {
  await openYourAstro();
  element<HTMLButtonElement>("your-astro-name-change").click();
  await waitForStep("name");
  page().dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  await waitForStep("page");
  page().dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  await waitFor(() => page().hidden, "the page to close");
  assert.equal(document.getElementById("sidebar")?.hasAttribute("inert"), false);
  assert.equal(document.activeElement?.id, "composer-input", "focus returns to the message field");
});

test("should_ignore_settings_while_the_first_flight_is_open", async () => {
  bootJourney(undefined, { astroProfile: null });
  await waitFor(() => !element("first-flight").hidden, "the first flight");
  window.__thintalkCommands?.run({ name: "openYourAstro" });
  assert.equal(page().hidden, true, "the flight already asks every question");
});

test("should_replay_the_first_flight_from_your_astro_and_bring_a_hidden_astro_back", async () => {
  const journey = await openYourAstro({ ...MET_ASTRO_PROFILE, isAstroShown: false });
  element<HTMLButtonElement>("your-astro-replay").click();
  await waitFor(() => !element("first-flight").hidden && element("first-flight").dataset.step === "hello", "the flight's hello");
  assert.equal(page().hidden, true);
  assert.equal(savedProfile(journey)["isAstroShown"], true);
  assert.ok(document.getElementById("sidebar")?.hasAttribute("inert"), "the flight keeps the window behind it inert");
});

test("should_offer_a_brain_on_the_page_when_none_is_installed", async () => {
  await openYourAstro(MET_ASTRO_PROFILE, { models: [] });
  await waitFor(() => isPresent("your-astro-brain-shown"), "the brain row");
  assert.match(element("your-astro-brain-shown").textContent ?? "", /Not yet/);
  await waitFor(() => isPresent("your-astro-gift-wake") || /didn't arrive/.test(astroSays()), "the brain offer");
});

test("should_say_the_engine_is_silent_instead_of_offering_downloads", async () => {
  const journey = await openYourAstro();
  journey.daemon.modelsUnreachable = true;
  element<HTMLButtonElement>("your-astro-done").click();
  window.__thintalkCommands?.run({ name: "openYourAstro" });
  await waitFor(() => isPresent("your-astro-check-again"), "the engine-silent row");
  assert.equal(isPresent("your-astro-sketchbook"), false, "no sketchbook offer while nothing can be checked");
  journey.daemon.modelsUnreachable = false;
  element<HTMLButtonElement>("your-astro-check-again").click();
  await waitFor(() => isPresent("your-astro-brain-shown") && element("your-astro-brain-shown").textContent === "Ready", "the brain row to read ready");
  assert.equal(astroSays(), "Here's everything I remember about you, Maya. Change anything you like!", "the engine line does not linger");
});

test("should_ignore_settings_until_the_saved_profile_is_read", async () => {
  const journey = bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE });
  window.__thintalkCommands?.run({ name: "openYourAstro" });
  assert.equal(page().hidden, true, "a page drawn now would show a newcomer's blank answers");
  await waitFor(() => journey.host.receivedRequests.some((request) => request.op === "loadAstroProfile") && isPresent("astro-home"), "the profile to load");
  window.__thintalkCommands?.run({ name: "openYourAstro" });
  await waitForStep("page");
});

test("should_open_the_grown_up_note_from_the_page_and_come_back", async () => {
  await openYourAstro();
  element<HTMLButtonElement>("your-astro-grown-up-read").click();
  await waitForStep("grownUp");
  assert.match(page().textContent ?? "", /runs entirely on this Mac/);
  element<HTMLButtonElement>("your-astro-grown-up-back").click();
  await waitForStep("page");
});

test("should_close_the_sketchbook_offer_on_not_now_and_download_nothing", async () => {
  const sketchbookID = "example-org/starter-image-4bit";
  const journey = await openYourAstro(MET_ASTRO_PROFILE, {
    libraryCatalog: { entries: [{ huggingface_id: sketchbookID, display_name: "Starter Image", role: "image_generation", get_started: true, approximate_size_bytes: 4_619_700_724, ready_on_this_mac: false }] },
  });
  await waitFor(() => isPresent("your-astro-sketchbook"), "the sketchbook row's offer");
  element<HTMLButtonElement>("your-astro-sketchbook").click();
  await waitFor(() => isPresent("your-astro-gift-get-sketchbook"), "the sketchbook offer");
  assert.match(astroSays(), /sketchbook/);
  element<HTMLButtonElement>("your-astro-gift-not-now").click();
  assert.equal(isPresent("your-astro-gift-get-sketchbook"), false);
  assert.equal(astroSays(), "Here's everything I remember about you, Maya. Change anything you like!");
  assert.deepEqual(journey.daemon.requestedLibraryDownloads, [], "declining downloads nothing");
});
