import assert from "node:assert/strict";
import { test } from "node:test";
import { waitFor } from "./support/dom";
import { bootJourney, sendUserMessage, waitUntilReady } from "./support/journey";
import { chunkLine, toolCallLine } from "./support/scriptedChatFetch";
import { FAKE_GENERATED_IMAGE_BASE64 } from "./support/fakeDaemon";

/**
 * Tool use as a reader sees it: a model that can call tools asks for the
 * current time, the transcript shows a collapsed "Current time" row, and the
 * answer follows. The Tools switch in the model menu turns this off, and a
 * model that cannot call tools shows the switch disabled.
 */

const TOOL_CAPABLE_MODEL = {
  id: "tool-model",
  input_modalities: ["text"],
  supported_endpoints: ["/v1/chat/completions"],
  supports_tool_calls: true,
};

const PLAIN_MODEL = { id: "plain-model", input_modalities: ["text"], supported_endpoints: ["/v1/chat/completions"] };

const TIME_TOOL_THEN_ANSWER = [
  [toolCallLine("call-1", "get_current_time", "{}"), chunkLine({}, "tool_calls"), "data: [DONE]"],
  [chunkLine({ content: "It is two in the afternoon." }, "stop"), "data: [DONE]"],
];

function toolsSwitch(): HTMLButtonElement | null {
  return document.querySelector<HTMLButtonElement>('[data-testid="composer-tools"]');
}

function toolStepRows(): HTMLDetailsElement[] {
  return [...document.querySelectorAll<HTMLDetailsElement>("details.tool-step")];
}

test("should_show_a_finished_tool_step_row_before_the_answer", async () => {
  const { daemon, host } = bootJourney(undefined, { models: [TOOL_CAPABLE_MODEL], chatReplySequence: TIME_TOOL_THEN_ANSWER });
  await waitUntilReady();
  sendUserMessage("What time is it in Verona?");
  await waitFor(() => toolStepRows()[0]?.dataset.state === "succeeded", "the tool step to finish");
  await waitFor(() => (document.querySelector(".message--assistant .markdown-body, article:last-of-type .markdown-body")?.textContent ?? "").includes("two in the afternoon"), "the answer to render");
  const stepRow = toolStepRows()[0];
  assert.ok(stepRow);
  assert.equal(stepRow.open, false, "the row starts collapsed");
  assert.match(stepRow.querySelector("summary")?.textContent ?? "", /Current time: done/);
  assert.equal(daemon.chatRequestBodies().length, 2);
  assert.ok(Array.isArray(daemon.chatRequestBodies()[0]?.["tools"]), "the first request declares the tool");
  await waitFor(() => host.receivedRequests.some((request) => request.op === "save" && JSON.stringify(request.payload).includes("get_current_time")), "the step to be saved");
}, { timeout: 30_000 });

test("should_send_no_tools_after_the_reader_switches_them_off", async () => {
  const { daemon, host } = bootJourney(undefined, { models: [TOOL_CAPABLE_MODEL] });
  await waitUntilReady();
  await waitFor(() => toolsSwitch() !== null, "the Tools switch to render");
  assert.equal(toolsSwitch()?.getAttribute("aria-checked"), "true");
  toolsSwitch()?.dispatchEvent(new window.Event("click", { bubbles: true }));
  await waitFor(() => toolsSwitch()?.getAttribute("aria-checked") === "false", "the switch to turn off");
  await waitFor(() => host.receivedRequests.some((request) => request.op === "savePrefs" && JSON.stringify(request.payload).includes('"toolsEnabled":false')), "the choice to be saved");
  sendUserMessage("Who wrote Hamlet?");
  await waitFor(() => daemon.chatRequestBody() !== null, "the chat request to leave");
  assert.equal("tools" in (daemon.chatRequestBody() ?? {}), false);
}, { timeout: 30_000 });

test("should_disable_the_tools_switch_for_a_model_that_cannot_call_tools", async () => {
  const { daemon } = bootJourney(undefined, { models: [PLAIN_MODEL] });
  await waitUntilReady();
  await waitFor(() => toolsSwitch() !== null, "the Tools switch to render");
  assert.equal(toolsSwitch()?.disabled, true);
  assert.equal(toolsSwitch()?.getAttribute("aria-checked"), "false");
  sendUserMessage("Who wrote Hamlet?");
  await waitFor(() => daemon.chatRequestBody() !== null, "the chat request to leave");
  assert.equal("tools" in (daemon.chatRequestBody() ?? {}), false);
}, { timeout: 30_000 });

test("should_disable_the_switch_after_moving_to_a_model_without_tools_and_restore_it_after_moving_back", async () => {
  bootJourney(undefined, { models: [TOOL_CAPABLE_MODEL, PLAIN_MODEL] });
  await waitUntilReady();
  const chooseModel = (modelID: string): void => {
    document.querySelector<HTMLButtonElement>(`[data-testid="composer-model-${modelID}"]`)?.dispatchEvent(new window.Event("click", { bubbles: true }));
  };
  await waitFor(() => toolsSwitch()?.disabled === false, "the switch to start enabled");
  chooseModel("plain-model");
  await waitFor(() => toolsSwitch()?.disabled === true, "the switch to disable for the plain model");
  assert.equal(toolsSwitch()?.getAttribute("aria-checked"), "false");
  chooseModel("tool-model");
  await waitFor(() => toolsSwitch()?.disabled === false, "the switch to enable again");
  assert.equal(toolsSwitch()?.getAttribute("aria-checked"), "true", "the reader's choice was kept");
}, { timeout: 30_000 });

test("should_read_a_file_the_reader_allows_and_chooses", async () => {
  const fileThenAnswer = [
    [toolCallLine("call-1", "read_file", "{}"), chunkLine({}, "tool_calls"), "data: [DONE]"],
    [chunkLine({ content: "Your file is a sonnet." }, "stop"), "data: [DONE]"],
  ];
  const { daemon, host } = bootJourney(undefined, { models: [TOOL_CAPABLE_MODEL], chatReplySequence: fileThenAnswer });
  host.chosenTextFile = { fileName: "sonnet-18.txt", text: "Shall I compare thee to a summer's day?" };
  await waitUntilReady();
  sendUserMessage("Read my sonnet file.");
  await waitFor(() => toolStepRows()[0]?.dataset.state === "succeeded", "the file step to finish");
  const followUpMessages = daemon.chatRequestBodies()[1]?.["messages"] as { role: string; content: string }[];
  assert.match(followUpMessages[followUpMessages.length - 1]?.content ?? "", /Contents of sonnet-18\.txt:[\s\S]*summer's day/);
  const hostToolRequest = host.receivedRequests.find((request) => request.op === "useHostTool");
  const savedConversationIDs = host.receivedRequests.filter((request) => request.op === "save").map((request) => (request.payload as { id: string }).id);
  assert.equal((hostToolRequest?.payload as { conversationID: string }).conversationID, savedConversationIDs[0], "the approval is asked for the open conversation");
}, { timeout: 30_000 });

test("should_tell_the_model_when_the_reader_does_not_allow_the_file", async () => {
  const fileThenAnswer = [
    [toolCallLine("call-1", "read_file", "{}"), chunkLine({}, "tool_calls"), "data: [DONE]"],
    [chunkLine({ content: "Understood." }, "stop"), "data: [DONE]"],
  ];
  const { daemon, host } = bootJourney(undefined, { models: [TOOL_CAPABLE_MODEL], chatReplySequence: fileThenAnswer });
  host.toolApprovalAnswer = false;
  await waitUntilReady();
  sendUserMessage("Read my file.");
  await waitFor(() => toolStepRows()[0]?.dataset.state === "failed", "the refused step");
  await waitFor(() => daemon.chatRequestBodies().length === 2, "the model to hear about the refusal");
  const followUpMessages = daemon.chatRequestBodies()[1]?.["messages"] as { content: string }[];
  assert.match(followUpMessages[followUpMessages.length - 1]?.content ?? "", /did not allow/);
  assert.equal(host.receivedRequests.filter((request) => request.op === "useHostTool").length, 1, "one sheet, and the model was told not to ask again");
}, { timeout: 30_000 });

const IMAGE_MODEL = { id: "FLUX.2-klein-4B", input_modalities: ["text"], supported_endpoints: ["/v1/images/generations"] };
const SEEING_TOOL_MODEL = { ...TOOL_CAPABLE_MODEL, id: "seeing-model", input_modalities: ["text", "image"] };
const GENERATED_IMAGE_DATA_URL = `data:image/png;base64,${FAKE_GENERATED_IMAGE_BASE64}`;

const DRAW_THEN_ANSWER_THEN_FOLLOW_UP = [
  [toolCallLine("call-1", "generate_image", JSON.stringify({ prompt: "A boy in a striped shirt" })), chunkLine({}, "tool_calls"), "data: [DONE]"],
  [chunkLine({ content: "Here is the boy." }, "stop"), "data: [DONE]"],
  [chunkLine({ content: "His shirt has rainbow stripes." }, "stop"), "data: [DONE]"],
];

function sentImageURLs(requestBody: Record<string, unknown> | undefined): string[] {
  return [...JSON.stringify(requestBody?.["messages"] ?? []).matchAll(/"url":"(data:[^"]+)"/g)].map((urlMatch) => urlMatch[1] as string);
}

test("should_show_a_seeing_model_the_image_it_generated_and_again_on_the_next_turn", async () => {
  const { daemon } = bootJourney(undefined, { models: [SEEING_TOOL_MODEL, IMAGE_MODEL], chatReplySequence: DRAW_THEN_ANSWER_THEN_FOLLOW_UP });
  await waitUntilReady();
  sendUserMessage("Draw a boy in a striped shirt.");
  await waitFor(() => daemon.chatRequestBodies().length === 2, "the answer request after the tool");
  assert.deepEqual(sentImageURLs(daemon.chatRequestBodies()[0]), [], "nothing to see before drawing");
  assert.deepEqual(sentImageURLs(daemon.chatRequestBodies()[1]), [GENERATED_IMAGE_DATA_URL], "the model sees the image it just made");
  await waitUntilReady();
  sendUserMessage("What colour is his shirt?");
  await waitFor(() => daemon.chatRequestBodies().length === 3, "the follow-up request");
  assert.deepEqual(sentImageURLs(daemon.chatRequestBodies()[2]), [GENERATED_IMAGE_DATA_URL], "the image comes back on a later turn");
}, { timeout: 30_000 });

test("should_never_send_pixels_to_a_text_only_model", async () => {
  const { daemon } = bootJourney(undefined, { models: [TOOL_CAPABLE_MODEL, IMAGE_MODEL], chatReplySequence: DRAW_THEN_ANSWER_THEN_FOLLOW_UP });
  await waitUntilReady();
  sendUserMessage("Draw a boy in a striped shirt.");
  await waitFor(() => daemon.chatRequestBodies().length === 2, "the answer request after the tool");
  await waitUntilReady();
  sendUserMessage("What colour is his shirt?");
  await waitFor(() => daemon.chatRequestBodies().length === 3, "the follow-up request");
  for (const requestBody of daemon.chatRequestBodies()) {
    assert.deepEqual(sentImageURLs(requestBody), []);
  }
  assert.match(JSON.stringify(daemon.chatRequestBodies()[1]?.["messages"]), /cannot see it/);
}, { timeout: 30_000 });
