import assert from "node:assert/strict";
import { test } from "node:test";
import { waitFor } from "./support/dom";
import { TrustBoundary } from "../src/TrustBoundary";
import { MathPipeline } from "../src/MathPipeline";
import { DiagramPipeline } from "../src/DiagramPipeline";
import { Renderer } from "../src/Renderer";
import { TranscriptView } from "../src/TranscriptView";
import { TranscriptMessage } from "../src/TranscriptMessage";
import type { TranscriptAction } from "../src/TranscriptAction";
import { ChatRole } from "../src/ChatRole";
import { ChatMessageState } from "../src/ChatMessageState";
import { MessagePicker, PICKING_CLASS } from "../src/MessagePicker";

interface PickerScene {
  picker: MessagePicker;
  appRoot: HTMLElement;
  transcriptElement: HTMLElement;
  pickBar: HTMLElement;
  pickCount: HTMLElement;
  doneButton: HTMLButtonElement;
  cancelButton: HTMLButtonElement;
  reportedActions: TranscriptAction[];
}

async function pickerScene(): Promise<PickerScene> {
  const trust = new TrustBoundary();
  const appRoot = document.createElement("div");
  const transcriptElement = document.createElement("div");
  appRoot.appendChild(transcriptElement);
  const reportedActions: TranscriptAction[] = [];
  const transcript = new TranscriptView({
    transcript: transcriptElement,
    emptyState: document.createElement("div"),
    renderer: new Renderer({ trust, math: new MathPipeline(trust), diagrams: new DiagramPipeline() }),
    onAction: (action) => reportedActions.push(action),
    onError: (detail) => detail,
  });
  transcript.render({
    messages: [
      new TranscriptMessage({ id: "q1", role: ChatRole.USER, markdown: "What is mitosis?", reasoning: "", state: ChatMessageState.COMPLETE }),
      new TranscriptMessage({ id: "a1", role: ChatRole.ASSISTANT, markdown: "Cell division. [More](https://example.com)", reasoning: "", state: ChatMessageState.COMPLETE }),
      new TranscriptMessage({ id: "q2", role: ChatRole.USER, markdown: "Thanks", reasoning: "", state: ChatMessageState.COMPLETE }),
    ],
    notice: null,
  });
  await waitFor(() => transcript.messageArticle("q2") !== null, "the conversation to be drawn");
  const pickBar = document.createElement("div");
  pickBar.hidden = true;
  const pickCount = document.createElement("p");
  const doneButton = document.createElement("button");
  const cancelButton = document.createElement("button");
  const picker = new MessagePicker({ appRoot, transcriptElement, pickBar, pickCount, doneButton, cancelButton }, transcript);
  return { picker, appRoot, transcriptElement, pickBar, pickCount, doneButton, cancelButton, reportedActions };
}

function articleOf(scene: PickerScene, messageID: string): HTMLElement {
  const article = scene.transcriptElement.querySelector<HTMLElement>(`[data-message-id="${messageID}"]`);
  assert.ok(article, `message ${messageID} is drawn`);
  return article;
}

function click(target: Element): void {
  target.dispatchEvent(new window.MouseEvent("click", { bubbles: true, cancelable: true }));
}

test("should_tick_clicked_messages_and_return_them_in_conversation_order", async () => {
  const scene = await pickerScene();
  const picked = scene.picker.pick(["q1", "a1", "q2"], []);

  assert.ok(scene.appRoot.classList.contains(PICKING_CLASS));
  assert.equal(scene.pickBar.hidden, false);
  assert.equal(scene.pickCount.textContent, "Click the messages to export");
  assert.equal(scene.doneButton.disabled, true, "there is nothing to export yet");
  click(articleOf(scene, "q2"));
  const answerText = articleOf(scene, "a1").querySelector(".markdown-body");
  assert.ok(answerText, `the answer has a body: ${articleOf(scene, "a1").innerHTML.slice(0, 300)}`);
  click(answerText);
  assert.equal(scene.pickCount.textContent, "2 messages picked");
  assert.equal(articleOf(scene, "a1").getAttribute("aria-checked"), "true");
  click(scene.doneButton);

  assert.deepEqual(await picked, ["a1", "q2"]);
  assert.equal(scene.appRoot.classList.contains(PICKING_CLASS), false);
  assert.equal(scene.pickBar.hidden, true);
  assert.equal(articleOf(scene, "a1").getAttribute("role"), null, "the marks go once picking ends");
});

test("should_untick_a_message_clicked_twice_and_start_from_earlier_picks", async () => {
  const scene = await pickerScene();
  const picked = scene.picker.pick(["q1", "a1", "q2"], ["q1", "not-pickable"]);

  assert.equal(scene.pickCount.textContent, "1 message picked");
  click(articleOf(scene, "a1"));
  click(articleOf(scene, "a1"));
  click(scene.doneButton);

  assert.deepEqual(await picked, ["q1"]);
});

test("should_cancel_with_the_cancel_button_or_escape_and_return_nothing", async () => {
  const scene = await pickerScene();
  const cancelledByButton = scene.picker.pick(["q1", "a1"], []);
  click(articleOf(scene, "q1"));
  click(scene.cancelButton);
  assert.equal(await cancelledByButton, null);

  const cancelledByEscape = scene.picker.pick(["q1", "a1"], []);
  articleOf(scene, "a1").dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  assert.equal(await cancelledByEscape, null);
});

test("should_tick_the_focused_message_with_the_space_bar", async () => {
  const scene = await pickerScene();
  const picked = scene.picker.pick(["q1", "a1"], []);

  articleOf(scene, "a1").dispatchEvent(new window.KeyboardEvent("keydown", { key: " ", bubbles: true, cancelable: true }));
  click(scene.doneButton);

  assert.deepEqual(await picked, ["a1"]);
});

test("should_keep_links_and_copy_buttons_quiet_while_picking", async () => {
  const scene = await pickerScene();
  const picked = scene.picker.pick(["q1", "a1", "q2"], []);
  const copyButton = articleOf(scene, "a1").querySelector('[data-action="copy"]');
  assert.ok(copyButton);

  click(copyButton);

  assert.deepEqual(scene.reportedActions, [], "the click ticked the message instead of copying it");
  assert.equal(articleOf(scene, "a1").getAttribute("aria-checked"), "true");
  click(scene.cancelButton);
  await picked;
});

test("should_ignore_messages_it_was_not_given", async () => {
  const scene = await pickerScene();
  const picked = scene.picker.pick(["a1"], []);

  click(articleOf(scene, "q1"));

  assert.equal(articleOf(scene, "q1").getAttribute("aria-checked"), null);
  assert.equal(scene.pickCount.textContent, "Click the messages to export");
  click(scene.cancelButton);
  await picked;
});
