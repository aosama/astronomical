import assert from "node:assert/strict";
import { test } from "node:test";
import { waitFor } from "./support/dom";
import { bootJourney, waitUntilReady } from "./support/journey";

/**
 * The composer's picture pill: which installed image model draws the next
 * picture. It starts on the Library's Pictures pick, remembers the person's
 * choice across a relaunch, and stays out of the way with one image model or
 * when tools cannot run.
 */

const CHAT_MODEL = { id: "model-alpha", input_modalities: ["text"], supported_endpoints: ["/v1/chat/completions"], supports_tool_calls: true };
const FULL_PRECISION_IMAGE_MODEL = { id: "FLUX.2-klein-4B", input_modalities: ["text"], supported_endpoints: ["/v1/images/generations"] };
const FOUR_BIT_IMAGE_MODEL = { id: "flux2-klein-4b-4bit", input_modalities: ["text"], supported_endpoints: ["/v1/images/generations"] };
const PICTURES_LIBRARY = {
  entries: [
    { display_name: "FLUX.2 Klein 4B", role: "image_generation", get_started: false, requestable_model_id: "FLUX.2-klein-4B" },
    { display_name: "FLUX.2 Klein 4B 4-bit", role: "image_generation", get_started: true, requestable_model_id: "flux2-klein-4b-4bit" },
  ],
};
const TWO_IMAGE_MODELS = [CHAT_MODEL, FULL_PRECISION_IMAGE_MODEL, FOUR_BIT_IMAGE_MODEL];

function imageModelPill(): HTMLButtonElement {
  const pill = document.querySelector<HTMLButtonElement>('[data-testid="composer-image-model"]');
  if (!pill) {
    throw new Error("the composer picture pill exists");
  }
  return pill;
}

function imageModelLabel(): string | null {
  const label = document.querySelector<HTMLElement>('[data-testid="composer-image-model-label"]');
  return label != null ? label.textContent : null;
}

function imageModelChoice(imageModelID: string): HTMLButtonElement | null {
  return document.querySelector<HTMLButtonElement>(`[data-testid="composer-image-model-${imageModelID}"]`);
}

function chooseImageModel(imageModelID: string): void {
  imageModelPill().dispatchEvent(new window.Event("click", { bubbles: true }));
  const choice = imageModelChoice(imageModelID);
  if (!choice) {
    throw new Error(`the picture choice ${imageModelID} exists`);
  }
  choice.dispatchEvent(new window.Event("click", { bubbles: true }));
}

test("should_start_the_picture_pill_on_the_library_pictures_pick", async () => {
  bootJourney(undefined, { models: TWO_IMAGE_MODELS, libraryCatalog: PICTURES_LIBRARY });
  await waitUntilReady();
  await waitFor(() => !imageModelPill().hidden, "the picture pill to appear");

  assert.equal(imageModelLabel(), "FLUX.2 Klein 4B 4-bit");
  assert.equal(imageModelChoice("flux2-klein-4b-4bit")?.getAttribute("aria-checked"), "true");
  assert.equal(imageModelChoice("FLUX.2-klein-4B")?.textContent, "FLUX.2 Klein 4B");
}, { timeout: 30_000 });

test("should_remember_the_chosen_image_model_after_a_relaunch", async () => {
  const first = bootJourney(undefined, { models: TWO_IMAGE_MODELS, libraryCatalog: PICTURES_LIBRARY });
  await waitUntilReady();
  await waitFor(() => !imageModelPill().hidden, "the picture pill to appear");
  chooseImageModel("FLUX.2-klein-4B");
  await waitFor(() => imageModelLabel() === "FLUX.2 Klein 4B", "the pill to show the chosen model");
  await waitFor(
    () => first.host.receivedRequests.some((request) => request.op === "savePrefs"),
    "the picture choice to be persisted",
  );

  bootJourney(first, { models: TWO_IMAGE_MODELS, libraryCatalog: PICTURES_LIBRARY });
  await waitUntilReady();
  await waitFor(() => imageModelLabel() === "FLUX.2 Klein 4B", "the relaunched pill to show the saved choice");
  assert.equal(imageModelChoice("FLUX.2-klein-4B")?.getAttribute("aria-checked"), "true");
}, { timeout: 30_000 });

test("should_keep_only_one_composer_menu_open", async () => {
  bootJourney(undefined, { models: TWO_IMAGE_MODELS, libraryCatalog: PICTURES_LIBRARY });
  await waitUntilReady();
  await waitFor(() => !imageModelPill().hidden, "the picture pill to appear");
  const chatModelPill = document.querySelector<HTMLButtonElement>('[data-testid="composer-model"]');
  chatModelPill?.dispatchEvent(new window.Event("click", { bubbles: true }));
  imageModelPill().dispatchEvent(new window.Event("click", { bubbles: true }));

  assert.equal(imageModelPill().getAttribute("aria-expanded"), "true");
  assert.equal(chatModelPill?.getAttribute("aria-expanded"), "false");
}, { timeout: 30_000 });

test("should_hide_the_picture_pill_with_one_or_no_image_model", async () => {
  for (const installedModels of [[CHAT_MODEL, FOUR_BIT_IMAGE_MODEL], [CHAT_MODEL]]) {
    bootJourney(undefined, { models: installedModels, libraryCatalog: PICTURES_LIBRARY });
    await waitUntilReady();
    assert.equal(imageModelPill().hidden, true, JSON.stringify(installedModels.map((model) => model.id)));
  }
}, { timeout: 30_000 });

test("should_close_the_picture_menu_on_escape_and_on_a_click_outside", async () => {
  bootJourney(undefined, { models: TWO_IMAGE_MODELS, libraryCatalog: PICTURES_LIBRARY });
  await waitUntilReady();
  await waitFor(() => !imageModelPill().hidden, "the picture pill to appear");
  const pictureMenu = document.querySelector<HTMLElement>('[data-testid="composer-image-model-menu"]');

  imageModelPill().dispatchEvent(new window.Event("click", { bubbles: true }));
  document.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape" }));
  assert.equal(pictureMenu?.hidden, true, "Escape closes the menu");

  imageModelPill().dispatchEvent(new window.Event("click", { bubbles: true }));
  document.querySelector("#transcript")?.dispatchEvent(new window.Event("click", { bubbles: true }));
  assert.equal(pictureMenu?.hidden, true, "a click outside closes the menu");
  assert.equal(imageModelPill().getAttribute("aria-expanded"), "false");
}, { timeout: 30_000 });

test("should_show_a_model_name_with_markup_as_plain_text", async () => {
  const markupLibrary = {
    entries: [
      { display_name: "<b>Bold</b> model", role: "image_generation", get_started: true, requestable_model_id: "flux2-klein-4b-4bit" },
    ],
  };
  bootJourney(undefined, { models: TWO_IMAGE_MODELS, libraryCatalog: markupLibrary });
  await waitUntilReady();
  await waitFor(() => !imageModelPill().hidden, "the picture pill to appear");

  assert.equal(imageModelLabel(), "<b>Bold</b> model");
  assert.equal(imageModelChoice("flux2-klein-4b-4bit")?.children.length, 0, "the name is text, never markup");
  assert.equal(imageModelPill().getAttribute("aria-label"), "Image model: <b>Bold</b> model");
}, { timeout: 30_000 });

test("should_hide_the_picture_pill_when_tools_are_off", async () => {
  bootJourney(undefined, { models: TWO_IMAGE_MODELS, libraryCatalog: PICTURES_LIBRARY, seedPreferences: { toolsEnabled: false } });
  await waitUntilReady();
  await new Promise((resolve) => setTimeout(resolve, 64));

  assert.equal(imageModelPill().hidden, true, "a pill that cannot take effect is not offered");
}, { timeout: 30_000 });
