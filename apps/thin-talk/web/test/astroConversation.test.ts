import assert from "node:assert/strict";
import { test } from "node:test";
import { waitFor } from "./support/dom";
import { bootJourney, MET_ASTRO_PROFILE, sendUserMessage, waitUntilReady } from "./support/journey";
import { chunkLine, toolCallLine } from "./support/scriptedChatFetch";

// The conversation with Astro shown, driven through the real page wiring:
// Astro's face and tags on answers, skill moments, the skill buttons, the
// friendly pill names, chat colours in the sidebar, and the Focus switch back
// to classic Thin Talk.

const TOOL_CAPABLE_MODEL = { id: "tool-model", input_modalities: ["text"], supported_endpoints: ["/v1/chat/completions"], supports_tool_calls: true };
const IMAGE_MODEL = { id: "FLUX.2-klein-4B", input_modalities: ["text"], supported_endpoints: ["/v1/images/generations"] };
const DRAW_THEN_ANSWER = [
  [toolCallLine("call-1", "generate_image", JSON.stringify({ prompt: "Astro past the moon" })), chunkLine({}, "tool_calls"), "data: [DONE]"],
  [chunkLine({ content: "Ta-da!" }, "stop"), "data: [DONE]"],
];
const PILOT_PROFILE = { ...MET_ASTRO_PROFILE, nickname: "Sam", style: "pilot" };

function assistantArticles(): HTMLElement[] {
  return [...document.querySelectorAll<HTMLElement>("article.message--assistant")];
}

function text(selector: string): string {
  return document.querySelector(selector)?.textContent ?? "";
}

test("should_answer_as_astro_with_a_face_a_name_and_a_skill_tag", async () => {
  bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE });
  await waitUntilReady();
  sendUserMessage("Why is the sky blue?");
  await waitFor(() => assistantArticles().length === 1 && text("article.message--assistant .message__state") === "", "a finished answer");
  const answer = assistantArticles()[0]!;
  assert.ok(answer.querySelector(".astro-avatar img"), "Astro's face");
  assert.equal(answer.querySelector(".message__role")?.textContent, "Astro");
  assert.equal(answer.querySelector("[data-testid=astro-skill-tag]")?.textContent, "Chat");
  assert.equal(answer.querySelector(".astro-avatar")?.getAttribute("data-mood"), "done");
  assert.match(text("article.message--user .message__role"), /Maya/);
  assert.ok(document.getElementById("app")?.classList.contains("app--astro"));
});

test("should_animate_only_the_newest_answer", async () => {
  bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE });
  await waitUntilReady();
  sendUserMessage("First question");
  await waitFor(() => assistantArticles().length === 1, "the first answer");
  await waitUntilReady();
  sendUserMessage("Second question");
  await waitFor(() => assistantArticles().length === 2, "the second answer");
  await waitFor(() => assistantArticles()[1]!.classList.contains("message--latest"), "the newest answer to be marked");
  assert.equal(assistantArticles()[0]!.classList.contains("message--latest"), false);
});

test("should_tell_a_drawing_as_a_sketchbook_moment_tagged_pictures", async () => {
  bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE, models: [TOOL_CAPABLE_MODEL, IMAGE_MODEL], chatReplySequence: DRAW_THEN_ANSWER });
  await waitUntilReady();
  sendUserMessage("Draw yourself past the moon");
  await waitFor(() => text(".tool-step .skill-moment__words") === "Drew it in my sketchbook", "the finished sketchbook moment");
  assert.equal(document.querySelector(".tool-step")?.getAttribute("data-skill"), "pictures");
  await waitFor(() => text("[data-testid=astro-skill-tag]") === "Pictures", "the Pictures tag");
});

test("should_colour_the_chat_in_the_sidebar_by_its_main_skill", async () => {
  const { host } = bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE, models: [TOOL_CAPABLE_MODEL, IMAGE_MODEL], chatReplySequence: DRAW_THEN_ANSWER });
  await waitUntilReady();
  sendUserMessage("Draw yourself past the moon");
  await waitFor(() => document.querySelector(".sidebar__session-skill")?.getAttribute("data-skill") === "pictures", "the sidebar colour");
  const savedDocument = host.receivedRequests.filter((request) => request.op === "save").pop()?.payload as Record<string, unknown>;
  assert.equal(savedDocument["skill"], "pictures", "the main skill is saved with the chat");
});

test("should_use_friendly_pill_names_for_explorers_and_real_names_for_pilots", async () => {
  bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE });
  await waitUntilReady();
  assert.equal(text("[data-testid=composer-model-label]"), "Astro's brain");
  assert.equal(document.querySelector<HTMLButtonElement>("[data-testid=composer-model]")?.title, "fake-chat-model", "the tooltip still names the model");

  bootJourney(undefined, { astroProfile: PILOT_PROFILE });
  await waitUntilReady();
  assert.equal(text("[data-testid=composer-model-label]"), "fake-chat-model");
  assert.ok(document.getElementById("app")?.classList.contains("app--astro-pilot"));
});

test("should_fill_the_message_field_from_a_skill_button_without_sending", async () => {
  const { daemon } = bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE });
  await waitUntilReady();
  assert.equal(document.getElementById("skill-launchers")?.hidden, false);
  document.querySelector<HTMLButtonElement>("[data-testid=composer-skill-look]")!.click();
  const field = document.querySelector<HTMLTextAreaElement>("#composer-input")!;
  assert.equal(field.value, "Look at a photo I pick and tell me what you see.");
  assert.equal(daemon.chatRequestBody(), null);
});

test("should_keep_the_skill_buttons_inside_the_message_field_with_a_name_each", async () => {
  bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE });
  await waitUntilReady();
  const launcherGroup = document.getElementById("skill-launchers")!;
  assert.ok(document.getElementById("composer")!.contains(launcherGroup));
  const launcherButtons = [...launcherGroup.querySelectorAll<HTMLButtonElement>("button[data-skill]")];
  // An icon alone has to be guessed, so each button names itself to a pointer
  // (title) and to assistive technology (aria-label).
  assert.deepEqual(launcherButtons.map((launcherButton) => launcherButton.title), ["Look at a photo", "Read a file", "Draw"]);
  for (const launcherButton of launcherButtons) {
    assert.equal(launcherButton.getAttribute("aria-label"), launcherButton.title);
  }
});

test("should_not_offer_a_skill_button_for_talking_before_talking_exists", async () => {
  bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE });
  await waitUntilReady();
  // Booleans, not elements: a failed comparison of DOM nodes makes the assertion
  // print the whole document, which stalls the run for minutes.
  const launcherGroup = document.getElementById("skill-launchers")!;
  assert.equal(launcherGroup.querySelector("[data-skill=voice]") === null, true);
  assert.equal(document.getElementById("composer-skill-voice") === null, true);
});

test("should_leave_the_message_field_alone_when_the_gap_between_skill_buttons_is_clicked", async () => {
  bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE });
  await waitUntilReady();
  document.getElementById("skill-launchers")!.click();
  assert.equal(document.querySelector<HTMLTextAreaElement>("#composer-input")!.value, "");
});

test("should_return_to_classic_thin_talk_with_the_focus_switch_and_back", async () => {
  bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE });
  await waitUntilReady();
  sendUserMessage("Hello");
  await waitFor(() => assistantArticles().length === 1, "an answer");
  window.__thintalkCommands?.run({ name: "toggleAstro" });
  await waitFor(() => text("article.message--assistant .message__role") === "Assistant", "the classic label");
  assert.equal(document.querySelector(".astro-avatar"), null);
  assert.equal(document.getElementById("skill-launchers")?.hidden, true);
  assert.equal(text("[data-testid=composer-model-label]"), "fake-chat-model");
  assert.equal(document.getElementById("app")?.classList.contains("app--astro"), false);
  window.__thintalkCommands?.run({ name: "toggleAstro" });
  await waitFor(() => text("article.message--assistant .message__role") === "Astro", "Astro back");
});

test("should_keep_classic_thin_talk_untouched_for_someone_who_hid_astro", async () => {
  bootJourney();
  await waitUntilReady();
  sendUserMessage("Hello");
  await waitFor(() => assistantArticles().length === 1, "an answer");
  assert.equal(text("article.message--assistant .message__role"), "Assistant");
  assert.equal(text("article.message--user .message__role"), "You");
  assert.equal(document.getElementById("skill-launchers")?.hidden, true);
});

test("should_offer_the_sketchbook_in_a_new_chat_when_draw_is_pressed_without_a_picture_model", async () => {
  const sketchbookCatalog = {
    entries: [{ huggingface_id: "example-org/starter-image-4bit", role: "image_generation", get_started: true, approximate_size_bytes: 4_619_700_724, ready_on_this_mac: false }],
  };
  const { daemon } = bootJourney(undefined, { astroProfile: MET_ASTRO_PROFILE, libraryCatalog: sketchbookCatalog });
  await waitUntilReady();
  document.querySelector<HTMLButtonElement>("[data-testid=composer-skill-pictures]")!.click();
  await waitFor(() => document.querySelector("[data-testid=astro-home-get-sketchbook]") !== null, "the sketchbook offer");
  assert.equal(document.querySelector<HTMLTextAreaElement>("#composer-input")!.value, "", "no drawing request that would fail");
  assert.equal(daemon.chatRequestBody(), null);
  assert.deepEqual(daemon.requestedLibraryDownloads, [], "nothing downloads until the person agrees");
});
