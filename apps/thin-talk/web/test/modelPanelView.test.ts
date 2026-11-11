import assert from "node:assert/strict";
import { test } from "node:test";
import { waitFor } from "./support/dom";
import { bootJourney, waitUntilReady } from "./support/journey";

/**
 * The composer's brain control and its one panel: the chat model in use under
 * its one name, its id, every installed chat and picture model in the panel, the
 * picture choice remembered across a relaunch, and the way to the Models page.
 */

const CHAT_MODEL = { id: "Ornith-1.5-35B-A3B-MLX-6bit-vision", input_modalities: ["text", "image"], supported_endpoints: ["/v1/chat/completions"], supports_tool_calls: true };
const SECOND_CHAT_MODEL = { id: "Ornith-1.5-9B-MLX-6bit-vision", input_modalities: ["text"], supported_endpoints: ["/v1/chat/completions"], supports_tool_calls: true };
const FULL_PRECISION_IMAGE_MODEL = { id: "FLUX.2-klein-4B", input_modalities: ["text"], supported_endpoints: ["/v1/images/generations"] };
const FOUR_BIT_IMAGE_MODEL = { id: "flux2-klein-4b-4bit", input_modalities: ["text"], supported_endpoints: ["/v1/images/generations"] };
const ONE_GIGABYTE = 1024 * 1024 * 1024;
const LIBRARY_CATALOG = {
  entries: [
    { model_id: CHAT_MODEL.id, approximate_size_bytes: 27.1 * ONE_GIGABYTE, capabilities: { supports_vision: true }, role: "chat", requestable_model_id: CHAT_MODEL.id },
    { model_id: SECOND_CHAT_MODEL.id, approximate_size_bytes: 7.6 * ONE_GIGABYTE, role: "chat", requestable_model_id: SECOND_CHAT_MODEL.id },
    { model_id: FULL_PRECISION_IMAGE_MODEL.id, role: "image_generation", get_started: false, requestable_model_id: FULL_PRECISION_IMAGE_MODEL.id },
    { model_id: FOUR_BIT_IMAGE_MODEL.id, role: "image_generation", get_started: true, requestable_model_id: FOUR_BIT_IMAGE_MODEL.id },
  ],
};
const EVERY_MODEL = [CHAT_MODEL, SECOND_CHAT_MODEL, FULL_PRECISION_IMAGE_MODEL, FOUR_BIT_IMAGE_MODEL];

function brainControl(): HTMLButtonElement {
  return document.querySelector<HTMLButtonElement>('[data-testid="composer-model"]')!;
}

function brainLabel(): string {
  return document.querySelector<HTMLElement>('[data-testid="composer-model-label"]')!.textContent ?? "";
}

function brainPanel(): HTMLElement {
  return document.querySelector<HTMLElement>('[data-testid="composer-model-menu"]')!;
}

function panelChoice(testID: string): HTMLButtonElement | null {
  return document.querySelector<HTMLButtonElement>(`[data-testid="${testID}"]`);
}

function click(element: Element): void {
  element.dispatchEvent(new window.Event("click", { bubbles: true }));
}

async function bootWithEveryModel(seedPreferences?: Record<string, unknown>): Promise<ReturnType<typeof bootJourney>> {
  const journey = bootJourney(undefined, { models: EVERY_MODEL, libraryCatalog: LIBRARY_CATALOG, seedPreferences });
  await waitUntilReady();
  await waitFor(() => brainLabel() === CHAT_MODEL.id, "the model id on the brain control");
  return journey;
}

test("should_name_the_chat_model_by_its_id_on_the_control_in_the_tooltip_and_for_voiceover", async () => {
  await bootWithEveryModel();
  assert.equal(brainLabel(), CHAT_MODEL.id);
  assert.equal(brainControl().title, CHAT_MODEL.id);
  assert.equal(brainControl().getAttribute("aria-label"), `Brain: ${CHAT_MODEL.id}`);
}, { timeout: 30_000 });

test("should_list_chat_models_then_picture_models_in_one_panel_with_details", async () => {
  await bootWithEveryModel();
  click(brainControl());
  await waitFor(() => panelChoice(`composer-image-model-${FOUR_BIT_IMAGE_MODEL.id}`) !== null, "the Pictures section");
  const headings = [...brainPanel().querySelectorAll(".composer__menu-heading")].map((heading) => heading.textContent);
  assert.deepEqual(headings, ["Chat", "Pictures"]);
  const chosenChat = panelChoice(`composer-model-${CHAT_MODEL.id}`)!;
  assert.equal(chosenChat.textContent?.includes(CHAT_MODEL.id), true, "the panel row shows the model id");
  assert.equal(chosenChat.querySelector(".composer__model-detail")?.textContent, "27.1 GB · sees images");
  assert.equal(chosenChat.getAttribute("aria-checked"), "true");
  // The Library's Image Generation pick is chosen until the person picks another.
  assert.equal(panelChoice(`composer-image-model-${FOUR_BIT_IMAGE_MODEL.id}`)?.getAttribute("aria-checked"), "true");
}, { timeout: 30_000 });

test("should_remember_the_chosen_picture_model_after_a_relaunch", async () => {
  const first = await bootWithEveryModel();
  click(brainControl());
  await waitFor(() => panelChoice(`composer-image-model-${FULL_PRECISION_IMAGE_MODEL.id}`) !== null, "the picture choices");
  click(panelChoice(`composer-image-model-${FULL_PRECISION_IMAGE_MODEL.id}`)!);
  assert.equal(brainPanel().hidden, true, "choosing closes the panel");
  await waitFor(() => first.host.receivedRequests.some((request) => request.op === "savePrefs"), "the choice to be saved");

  bootJourney(first, { models: EVERY_MODEL, libraryCatalog: LIBRARY_CATALOG });
  await waitUntilReady();
  await waitFor(
    () => panelChoice(`composer-image-model-${FULL_PRECISION_IMAGE_MODEL.id}`)?.getAttribute("aria-checked") === "true",
    "the relaunched panel to keep the saved picture model",
  );
}, { timeout: 30_000 });

test("should_leave_out_the_pictures_section_with_one_picture_model_or_with_tools_off", async () => {
  const layouts = [
    { models: [CHAT_MODEL, FOUR_BIT_IMAGE_MODEL], seedPreferences: undefined },
    { models: EVERY_MODEL, seedPreferences: { toolsEnabled: false } },
  ];
  for (const layout of layouts) {
    bootJourney(undefined, { models: layout.models, libraryCatalog: LIBRARY_CATALOG, seedPreferences: layout.seedPreferences });
    await waitUntilReady();
    await new Promise((resolve) => setTimeout(resolve, 64));
    const headings = [...brainPanel().querySelectorAll(".composer__menu-heading")].map((heading) => heading.textContent);
    assert.deepEqual(headings, ["Chat"], JSON.stringify(layout.seedPreferences ?? layout.models.length));
  }
}, { timeout: 30_000 });

test("should_ask_the_host_to_open_the_models_page_from_the_panel", async () => {
  const { host } = await bootWithEveryModel();
  click(brainControl());
  click(panelChoice("composer-models-page")!);
  assert.equal(host.receivedActions.length, 1);
  const openAction = JSON.stringify(host.receivedActions[0]);
  assert.match(openAction, /openExternal/);
  assert.match(openAction, /\/download-models/);
  assert.equal(brainPanel().hidden, true, "the panel closes once the page is asked for");
}, { timeout: 30_000 });

test("should_close_the_panel_on_escape_and_on_a_click_outside", async () => {
  await bootWithEveryModel();
  click(brainControl());
  assert.equal(brainControl().getAttribute("aria-expanded"), "true");
  document.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape" }));
  assert.equal(brainPanel().hidden, true, "Escape closes the panel");

  click(brainControl());
  click(document.querySelector("#transcript")!);
  assert.equal(brainPanel().hidden, true, "a click outside closes the panel");
  assert.equal(brainControl().getAttribute("aria-expanded"), "false");
}, { timeout: 30_000 });

test("should_show_a_model_id_with_markup_as_plain_text", async () => {
  const markupModel = { ...CHAT_MODEL, id: "<b>Bold</b>-model" };
  bootJourney(undefined, { models: [markupModel] });
  await waitUntilReady();
  await waitFor(() => brainLabel() === "<b>Bold</b>-model", "the escaped name");
  click(brainControl());
  assert.equal(brainPanel().querySelector("b"), null, "the name is text, never markup");
}, { timeout: 30_000 });

test("should_fall_back_to_the_model_id_when_the_library_cannot_be_read", async () => {
  bootJourney(undefined, { models: [CHAT_MODEL] });
  await waitUntilReady();
  await new Promise((resolve) => setTimeout(resolve, 64));
  assert.equal(brainLabel(), CHAT_MODEL.id);
}, { timeout: 30_000 });
