import assert from "node:assert/strict";
import { test } from "node:test";
import { waitFor } from "./support/dom";
import { TrustBoundary } from "../src/TrustBoundary";
import { MathPipeline } from "../src/MathPipeline";
import { DiagramPipeline } from "../src/DiagramPipeline";
import { Renderer } from "../src/Renderer";
import { TranscriptView } from "../src/TranscriptView";
import { TranscriptMessage } from "../src/TranscriptMessage";
import { TranscriptAction } from "../src/TranscriptAction";
import { TranscriptAttachment } from "../src/TranscriptAttachment";
import { ChatRole } from "../src/ChatRole";
import { ChatMessageState } from "../src/ChatMessageState";

// A published snapshot used to re-run the markdown parse and the DOM diff for
// every message in the conversation, so one streamed delta cost work
// proportional to the whole transcript's length. The render gate must skip a
// byte-identical message and still re-render one whose content changed.

function makeRenderCountingTranscriptView(): { view: TranscriptView; messageHtmlCallCount: () => number } {
  const trust = new TrustBoundary();
  const renderer = new Renderer({ trust, math: new MathPipeline(trust), diagrams: new DiagramPipeline() });
  let callCount = 0;
  const originalMessageHtml = renderer.messageHtml.bind(renderer);
  renderer.messageHtml = (message: TranscriptMessage): string => {
    callCount += 1;
    return originalMessageHtml(message);
  };
  const transcriptHost = document.createElement("div");
  const emptyStateHost = document.createElement("div");
  const view = new TranscriptView({
    transcript: transcriptHost,
    emptyState: emptyStateHost,
    renderer,
    onAction: (action: TranscriptAction) => action,
    onError: (detail: string) => detail,
  });
  return { view, messageHtmlCallCount: () => callCount };
}

test("should_skip_re-rendering_a_message_whose_snapshot_is_unchanged", async () => {
  const { view, messageHtmlCallCount } = makeRenderCountingTranscriptView();
  const settledMessage = new TranscriptMessage({
    id: "settled-message-id",
    role: ChatRole.USER,
    markdown: "What does the prologue ask of the audience?",
    reasoning: "",
    state: ChatMessageState.COMPLETE,
  });

  view.render({ messages: [settledMessage], notice: null });
  await waitFor(() => messageHtmlCallCount() >= 1, "the first render to draw the message");
  const firstRenderCallCount = messageHtmlCallCount();

  // An identical snapshot — the streaming neighbour grew, this message did
  // not — must not touch the settled message again.
  view.render({ messages: [settledMessage], notice: null });
  await new Promise((resolveTick) => setTimeout(resolveTick, 40));
  assert.equal(
    messageHtmlCallCount(),
    firstRenderCallCount,
    "an unchanged message must stay on its rendered node without another parse",
  );

  // Changed content must still re-render.
  const editedMessage = new TranscriptMessage({
    id: "settled-message-id",
    role: ChatRole.USER,
    markdown: "What does the prologue ask of the audience, twice?",
    reasoning: "",
    state: ChatMessageState.COMPLETE,
  });
  view.render({ messages: [editedMessage], notice: null });
  await waitFor(() => messageHtmlCallCount() > firstRenderCallCount, "the changed message to re-render");
}, { timeout: 30_000 });

test("should_re-render_a_message_whose_attachments_changed", async () => {
  const { view, messageHtmlCallCount } = makeRenderCountingTranscriptView();
  const bareMessage = new TranscriptMessage({
    id: "attachment-gated-message-id",
    role: ChatRole.USER,
    markdown: "See this figure.",
    reasoning: "",
    state: ChatMessageState.COMPLETE,
  });
  view.render({ messages: [bareMessage], notice: null });
  await waitFor(() => messageHtmlCallCount() >= 1, "the bare message to render");
  const afterBareRender = messageHtmlCallCount();

  // Same prose, but an attachment appeared: the strip the renderer draws
  // depends on it, so the gate must treat the message as changed even though
  // its text did not move.
  const withAttachmentMessage = new TranscriptMessage({
    id: "attachment-gated-message-id",
    role: ChatRole.USER,
    markdown: "See this figure.",
    reasoning: "",
    state: ChatMessageState.COMPLETE,
    attachments: [
      new TranscriptAttachment({ id: "asset-gated", assetURL: "thintalk-asset://attachment/gated", label: "figure" }),
    ],
  });
  view.render({ messages: [withAttachmentMessage], notice: null });
  await waitFor(() => messageHtmlCallCount() > afterBareRender, "the attachment-bearing message to re-render");
}, { timeout: 30_000 });

test("should_re-render_a_message_whose_visible_state_changed", async () => {
  const { view, messageHtmlCallCount } = makeRenderCountingTranscriptView();
  const streamingMessage = new TranscriptMessage({
    id: "streaming-message-id",
    role: ChatRole.ASSISTANT,
    markdown: "Two households",
    reasoning: "",
    state: ChatMessageState.STREAMING,
  });
  view.render({ messages: [streamingMessage], notice: null });
  await waitFor(() => messageHtmlCallCount() >= 1, "the streaming message to render");
  const afterStreamingRender = messageHtmlCallCount();

  // Identical markdown, but the turn finished: the state label and diagram
  // policy both depend on the state, so the transition must re-render.
  const finishedMessage = new TranscriptMessage({
    id: "streaming-message-id",
    role: ChatRole.ASSISTANT,
    markdown: "Two households",
    reasoning: "",
    state: ChatMessageState.COMPLETE,
  });
  view.render({ messages: [finishedMessage], notice: null });
  await waitFor(() => messageHtmlCallCount() > afterStreamingRender, "the finished message to re-render");
}, { timeout: 30_000 });
