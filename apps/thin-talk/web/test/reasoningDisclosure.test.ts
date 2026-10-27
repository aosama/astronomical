import assert from "node:assert/strict";
import { test } from "node:test";
import { installDom, waitFor } from "./support/dom";
import { Renderer } from "../src/Renderer";
import { TranscriptView } from "../src/TranscriptView";
import { TranscriptMessage } from "../src/TranscriptMessage";
import { ChatRole } from "../src/ChatRole";
import { ChatMessageState } from "../src/ChatMessageState";
import { TrustBoundary } from "../src/TrustBoundary";
import { MathPipeline } from "../src/MathPipeline";
import { DiagramPipeline } from "../src/DiagramPipeline";
import { reasoningDisclosureIdentifier } from "../src/ReasoningDisclosure";

/**
 * Reasoning disclosure behaviour.
 *
 * These tests drive the real Renderer and TranscriptView against the real
 * index.html shell, because the defect they pin down only appears in the interplay
 * of the streamed template, morphdom patching, and the reader's own clicks: the
 * renderer emits a collapsed `<details>` on every snapshot, so a manually expanded
 * reasoning block used to be re-collapsed by the very next delta.
 */

const SHAKESPEARE_REASONING_SNIPPET =
  "The prince asks about the feud; I should answer with the prologue's verdict that the star-crossed lovers' death buries their parents' strife.";

function transcriptElements(): { transcript: HTMLElement; emptyState: HTMLElement } {
  const transcript = document.querySelector<HTMLElement>("#transcript");
  const emptyState = document.querySelector<HTMLElement>("#empty-state");
  if (!transcript || !emptyState) {
    throw new Error("index.html must carry #transcript and #empty-state");
  }
  return { transcript, emptyState };
}

function buildTranscriptView(): TranscriptView {
  const { transcript, emptyState } = transcriptElements();
  const trust = new TrustBoundary();
  return new TranscriptView({
    transcript,
    emptyState,
    renderer: new Renderer({
      trust,
      math: new MathPipeline(trust),
      diagrams: new DiagramPipeline(),
    }),
    onAction: () => undefined,
    onError: (detail) => {
      throw new Error(`transcript reported an error: ${detail}`);
    },
  });
}

function reasoningMessage(init: {
  id: string;
  reasoning: string;
  state: ChatMessageState;
}): TranscriptMessage {
  return new TranscriptMessage({
    id: init.id,
    role: ChatRole.ASSISTANT,
    markdown: init.state === ChatMessageState.STREAMING ? "" : "It buried the strife.",
    reasoning: init.reasoning,
    state: init.state,
  });
}

function renderedDisclosure(messageID: string): HTMLDetailsElement {
  const disclosure = document.querySelector<HTMLDetailsElement>(
    `details.reasoning#${CSS.escape(reasoningDisclosureIdentifier(messageID))}`,
  );
  if (!disclosure) {
    throw new Error(`the reasoning disclosure for ${messageID} is not in the DOM`);
  }
  return disclosure;
}

/** Emulates a reader clicking the disclosure: the browser flips `open` and the
 * toggle event carries the reader's intent to the transcript. */
function clickDisclosure(disclosure: HTMLDetailsElement): void {
  disclosure.open = !disclosure.open;
}

async function flushTranscript(): Promise<void> {
  // TranscriptView batches on requestAnimationFrame plus a 16 ms timer; the
  // wait loop drains both so the flush has definitely run.
  await new Promise((resolve) => setTimeout(resolve, 32));
}

/** Renders one snapshot of these messages; `notice` stays null because these
 * tests never exercise the empty state. */
function renderMessages(view: TranscriptView, messages: readonly TranscriptMessage[]): void {
  view.render({ messages, notice: null });
}

test("should_render_reasoning_expanded_by_default_while_streaming", async () => {
  installDom();
  const view = buildTranscriptView();
  renderMessages(view, [
    reasoningMessage({ id: "m-stream", reasoning: SHAKESPEARE_REASONING_SNIPPET, state: ChatMessageState.STREAMING }),
  ]);
  await flushTranscript();
  const disclosure = renderedDisclosure("m-stream");
  assert.equal(disclosure.open, true, "streaming reasoning must start expanded");
  assert.equal(
    disclosure.getAttribute("data-testid"),
    "reasoning:m-stream",
    "the disclosure must carry its stable selector",
  );
});

test("should_keep_the_reader_collapsed_choice_through_streamed_re-renders", async () => {
  installDom();
  const view = buildTranscriptView();
  const messageID = "m-collapse";
  renderMessages(view, [
    reasoningMessage({ id: messageID, reasoning: SHAKESPEARE_REASONING_SNIPPET, state: ChatMessageState.STREAMING }),
  ]);
  await flushTranscript();
  clickDisclosure(renderedDisclosure(messageID));
  assert.equal(renderedDisclosure(messageID).open, false, "the click must collapse it first");

  // The next streaming snapshot (longer reasoning, same message) must not
  // resurrect the block the reader just folded away.
  renderMessages(view, [
    reasoningMessage({
      id: messageID,
      reasoning: `${SHAKESPEARE_REASONING_SNIPPET} More deliberation arrives with the next delta.`,
      state: ChatMessageState.STREAMING,
    }),
  ]);
  await flushTranscript();
  assert.equal(renderedDisclosure(messageID).open, false, "the collapsed choice must survive the patch");
});

test("should_collapse_reasoning_once_the_message_completes", async () => {
  installDom();
  const view = buildTranscriptView();
  const messageID = "m-complete";
  renderMessages(view, [
    reasoningMessage({ id: messageID, reasoning: SHAKESPEARE_REASONING_SNIPPET, state: ChatMessageState.STREAMING }),
  ]);
  await flushTranscript();
  assert.equal(renderedDisclosure(messageID).open, true, "streaming starts expanded");

  renderMessages(view, [
    reasoningMessage({ id: messageID, reasoning: SHAKESPEARE_REASONING_SNIPPET, state: ChatMessageState.COMPLETE }),
  ]);
  await flushTranscript();
  assert.equal(renderedDisclosure(messageID).open, false, "finished reasoning must fold away");
});

test("should_reopen_reasoning_after_completion_and_keep_it_open", async () => {
  installDom();
  const view = buildTranscriptView();
  const messageID = "m-reopen";
  renderMessages(view, [
    reasoningMessage({ id: messageID, reasoning: SHAKESPEARE_REASONING_SNIPPET, state: ChatMessageState.STREAMING }),
  ]);
  await flushTranscript();
  renderMessages(view, [
    reasoningMessage({ id: messageID, reasoning: SHAKESPEARE_REASONING_SNIPPET, state: ChatMessageState.COMPLETE }),
  ]);
  await flushTranscript();
  assert.equal(renderedDisclosure(messageID).open, false, "completion collapses first");

  clickDisclosure(renderedDisclosure(messageID));
  // A later re-render of the finished message (a sidebar switch back, a final
  // snapshot) must not slam the reader's reopened panel shut again.
  renderMessages(view, [
    reasoningMessage({ id: messageID, reasoning: SHAKESPEARE_REASONING_SNIPPET, state: ChatMessageState.COMPLETE }),
  ]);
  await flushTranscript();
  assert.equal(renderedDisclosure(messageID).open, true, "the reopened panel must stay open");
});

test("should_render_no_disclosure_for_a_message_without_reasoning", async () => {
  installDom();
  const view = buildTranscriptView();
  renderMessages(view, [
    new TranscriptMessage({
      id: "m-plain",
      role: ChatRole.ASSISTANT,
      markdown: "A plain answer.",
      reasoning: "",
      state: ChatMessageState.COMPLETE,
    }),
  ]);
  await flushTranscript();
  const count = document.querySelectorAll("details.reasoning").length;
  assert.equal(count, 0, "no reasoning means no disclosure element at all");
});

test("should_stream_the_reasoning_body_into_the_expanded_panel", async () => {
  installDom();
  const view = buildTranscriptView();
  renderMessages(view, [
    reasoningMessage({ id: "m-body", reasoning: SHAKESPEARE_REASONING_SNIPPET, state: ChatMessageState.STREAMING }),
  ]);
  await waitFor(
    () => document.querySelectorAll("details.reasoning").length > 0,
    "the streaming disclosure to appear",
  );
  const disclosure = renderedDisclosure("m-body");
  const body = disclosure.querySelector<HTMLElement>(".reasoning__body");
  assert.ok(body, "the disclosure exposes a body element");
  const bodyText: string | null = body.textContent;
  assert.match(bodyText != null ? bodyText : "", /star-crossed lovers/, "the reasoning text must be visible");
});
