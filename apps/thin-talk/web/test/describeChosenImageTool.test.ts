import assert from "node:assert/strict";
import { test } from "node:test";
import { AppConfig } from "../src/AppConfig";
import { ChatFailure } from "../src/ChatFailure";
import { ChatFailureKind } from "../src/ChatFailureKind";
import { DaemonClient } from "../src/DaemonClient";
import { DescribeChosenImageTool } from "../src/DescribeChosenImageTool";
import { DAEMON_BASE_URL } from "./support/scriptedChatFetch";
import { scriptedBridge } from "./support/scriptedBridge";
import type { SessionBridgeRequest } from "../src/SessionBridgeRequest";

const VISION_MODEL = { id: "vision-model", input_modalities: ["text", "image"], supported_endpoints: ["/v1/chat/completions"] };
const SECOND_VISION_MODEL = { id: "other-vision-model", input_modalities: ["text", "image"], supported_endpoints: ["/v1/chat/completions"] };
const TEXT_MODEL = { id: "text-model", input_modalities: ["text"], supported_endpoints: ["/v1/chat/completions"] };
const CHOSEN_IMAGE = { fileName: "rose.png", base64: "iVBORw0KGgo=", mimeType: "image/png" };

/** A daemon that lists models and answers description requests, recording
 * each request body. */
function installVisionDaemon(models: unknown[], descriptionStatus = 200): Record<string, unknown>[] {
  const descriptionRequestBodies: Record<string, unknown>[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    if (String(input).endsWith("/v1/models")) {
      return Response.json({ data: models });
    }
    descriptionRequestBodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
    if (descriptionStatus !== 200) {
      return Response.json({ error: { message: "model_load_failed" } }, { status: descriptionStatus });
    }
    return Response.json({ choices: [{ message: { role: "assistant", content: " A red rose on white. " } }] });
  }) as typeof fetch;
  return descriptionRequestBodies;
}

function describeTool(chosenImage: unknown = CHOSEN_IMAGE, isApproved = true): DescribeChosenImageTool {
  const client = new DaemonClient(
    new AppConfig({ daemonBaseURL: DAEMON_BASE_URL, expectedChannel: "development", expectedStateDirectory: "~/.astronomical-dev" }),
  );
  const hostAnswer = !isApproved ? { outcome: "refused" } : chosenImage == null ? { outcome: "nothingChosen" } : { outcome: "chosen", ...(chosenImage as object) };
  return new DescribeChosenImageTool(client, scriptedBridge({ useHostTool: hostAnswer }));
}

test("should_ask_the_turns_own_model_when_it_can_see_images", async () => {
  const descriptionRequestBodies = installVisionDaemon([SECOND_VISION_MODEL, VISION_MODEL]);
  const toolResult = await describeTool().execute({ question: "What flower is this?" }, new AbortController().signal, { conversationID: "c1", modelID: "vision-model" });
  assert.equal(toolResult, "What vision-model sees in rose.png:\n\nA red rose on white.");
  const requestBody = descriptionRequestBodies[0] as { model: string; stream: boolean; messages: { content: unknown[] }[] };
  assert.equal(requestBody.model, "vision-model");
  assert.equal(requestBody.stream, false);
  assert.deepEqual(requestBody.messages[0]?.content, [
    { type: "text", text: "What flower is this?" },
    { type: "image_url", image_url: { url: "data:image/png;base64,iVBORw0KGgo=" } },
  ]);
});

test("should_use_the_first_vision_model_and_a_general_question_otherwise", async () => {
  const descriptionRequestBodies = installVisionDaemon([TEXT_MODEL, VISION_MODEL, SECOND_VISION_MODEL]);
  await describeTool().execute({}, new AbortController().signal, { conversationID: "c1", modelID: "text-model" });
  const requestBody = descriptionRequestBodies[0] as { model: string; messages: { content: { text?: string }[] }[] };
  assert.equal(requestBody.model, "vision-model");
  assert.equal(requestBody.messages[0]?.content[0]?.text, "Describe this image in detail.");
});

test("should_say_so_when_no_model_can_see_images_without_asking_the_user", async () => {
  installVisionDaemon([TEXT_MODEL]);
  const sentRequests: SessionBridgeRequest[] = [];
  const client = new DaemonClient(
    new AppConfig({ daemonBaseURL: DAEMON_BASE_URL, expectedChannel: "development", expectedStateDirectory: "~/.astronomical-dev" }),
  );
  const tool = new DescribeChosenImageTool(client, scriptedBridge({ useHostTool: { outcome: "approved" } }, sentRequests));
  await assert.rejects(tool.execute({}, new AbortController().signal, { conversationID: "c1", modelID: "text-model" }), /No model on this Mac can look at images/);
  assert.equal(sentRequests.length, 0, "no sheet for a tool that cannot work");
});

test("should_cap_the_description_length", async () => {
  const descriptionRequestBodies = installVisionDaemon([VISION_MODEL]);
  await describeTool().execute({}, new AbortController().signal, { conversationID: "c1", modelID: "vision-model" });
  assert.equal(descriptionRequestBodies[0]?.["max_tokens"], 1024);
});

test("should_send_nothing_when_refused_or_cancelled", async () => {
  const descriptionRequestBodies = installVisionDaemon([VISION_MODEL]);
  await assert.rejects(describeTool(CHOSEN_IMAGE, false).execute({}, new AbortController().signal, { conversationID: "c1", modelID: "vision-model" }), /did not allow/);
  await assert.rejects(describeTool(null).execute({}, new AbortController().signal, { conversationID: "c1", modelID: "vision-model" }), /did not choose anything/);
  assert.equal(descriptionRequestBodies.length, 0);
});

test("should_report_a_daemon_refusal_as_a_classified_failure", async () => {
  installVisionDaemon([VISION_MODEL], 503);
  await assert.rejects(
    describeTool().execute({}, new AbortController().signal, { conversationID: "c1", modelID: "vision-model" }),
    (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.MODEL_LOAD_FAILED,
  );
});
