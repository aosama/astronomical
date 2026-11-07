import assert from "node:assert/strict";
import { test } from "node:test";
import { waitFor } from "./support/dom";
import { bootJourney, sendUserMessage, waitUntilReady } from "./support/journey";
import { chunkLine, toolCallLine } from "./support/scriptedChatFetch";

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
