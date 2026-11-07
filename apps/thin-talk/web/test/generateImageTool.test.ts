import assert from "node:assert/strict";
import { test } from "node:test";
import { AppConfig } from "../src/AppConfig";
import { ChatFailure } from "../src/ChatFailure";
import { ChatFailureKind } from "../src/ChatFailureKind";
import { DaemonClient } from "../src/DaemonClient";
import { GenerateImageTool } from "../src/GenerateImageTool";
import { SessionBridge } from "../src/SessionBridge";
import { DAEMON_BASE_URL } from "./support/scriptedChatFetch";

const STORED_IMAGE_NAME = "0f3c2a51-7d7e-4c55-9e1a-2b8d4f6a9c10.png";
const IMAGE_MODEL = { id: "FLUX.2-klein-4B", input_modalities: ["text"], supported_endpoints: ["/v1/images/generations"] };
const CHAT_MODEL = { id: "chat-model", input_modalities: ["text"], supported_endpoints: ["/v1/chat/completions"] };

interface ImageDaemonScript {
  models: unknown[];
  imageStatus?: number;
  imageBody?: unknown;
}

/** A daemon that lists models and answers image requests, recording each
 * image request body. */
function installImageDaemon(script: ImageDaemonScript): Record<string, unknown>[] {
  const imageRequestBodies: Record<string, unknown>[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith("/v1/models")) {
      return Response.json({ data: script.models });
    }
    if (url.endsWith("/v1/images/generations")) {
      imageRequestBodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
      const imageBody = script.imageBody !== undefined ? script.imageBody : { data: [{ b64_json: "iVBORw0KGgo=", mime_type: "image/png" }] };
      return Response.json(imageBody, { status: script.imageStatus ?? 200 });
    }
    return new Response("not found", { status: 404 });
  }) as typeof fetch;
  return imageRequestBodies;
}

/** A bridge whose host stores every image under one known name. */
function storingBridge(storedImages: string[]): SessionBridge {
  const bridge = new SessionBridge((request) => {
    if (request.op === "storeGeneratedImage") {
      storedImages.push((request.payload as { base64: string }).base64);
      queueMicrotask(() => window.__thintalkBridge?.resolve(request.callId, { imageName: STORED_IMAGE_NAME }));
    }
  });
  return bridge;
}

function imageTool(storedImages: string[] = []): GenerateImageTool {
  const client = new DaemonClient(
    new AppConfig({ daemonBaseURL: DAEMON_BASE_URL, expectedChannel: "development", expectedStateDirectory: "~/.astronomical-dev" }),
  );
  return new GenerateImageTool(client, storingBridge(storedImages));
}

test("should_generate_with_the_first_image_model_store_the_image_and_report_it", async () => {
  const imageRequestBodies = installImageDaemon({ models: [CHAT_MODEL, IMAGE_MODEL] });
  const storedImages: string[] = [];
  const toolOutput = await imageTool(storedImages).execute({ prompt: "A rose in Verona at dusk", shape: "landscape" }, new AbortController().signal);
  assert.deepEqual(imageRequestBodies, [{
    model: "FLUX.2-klein-4B", prompt: "A rose in Verona at dusk", width: 1024, height: 768, response_format: "b64_json", n: 1,
  }]);
  assert.deepEqual(storedImages, ["iVBORw0KGgo="]);
  assert.equal(toolOutput.generatedImageName, STORED_IMAGE_NAME);
  assert.match(toolOutput.text, /1024×768 image with FLUX\.2-klein-4B/);
  assert.match(toolOutput.text, /do not describe/);
});

test("should_fall_back_to_a_square_image_for_an_unknown_shape", async () => {
  const imageRequestBodies = installImageDaemon({ models: [IMAGE_MODEL] });
  await imageTool().execute({ prompt: "A crown", shape: "hexagon" }, new AbortController().signal);
  assert.equal(imageRequestBodies[0]?.["width"], 1024);
  assert.equal(imageRequestBodies[0]?.["height"], 1024);
});

test("should_refuse_an_empty_prompt_without_calling_the_daemon", async () => {
  const imageRequestBodies = installImageDaemon({ models: [IMAGE_MODEL] });
  await assert.rejects(imageTool().execute({ prompt: "   " }, new AbortController().signal), /needs a description/);
  assert.equal(imageRequestBodies.length, 0);
});

test("should_say_so_when_no_image_model_is_installed", async () => {
  installImageDaemon({ models: [CHAT_MODEL] });
  await assert.rejects(imageTool().execute({ prompt: "A crown" }, new AbortController().signal), /No image model is installed/);
});

test("should_report_a_daemon_refusal_and_a_response_without_an_image", async () => {
  installImageDaemon({ models: [IMAGE_MODEL], imageStatus: 503, imageBody: { error: { message: "chat_worker_unavailable" } } });
  await assert.rejects(
    imageTool().execute({ prompt: "A crown" }, new AbortController().signal),
    (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.WORKER_UNAVAILABLE,
  );
  installImageDaemon({ models: [IMAGE_MODEL], imageBody: { data: [] } });
  await assert.rejects(
    imageTool().execute({ prompt: "A crown" }, new AbortController().signal),
    (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.MALFORMED_MODEL_OUTPUT,
  );
});

test("should_make_a_portrait_image_and_refuse_a_prompt_that_is_not_text", async () => {
  const imageRequestBodies = installImageDaemon({ models: [IMAGE_MODEL] });
  await imageTool().execute({ prompt: "A tall tower", shape: "portrait" }, new AbortController().signal);
  assert.equal(imageRequestBodies[0]?.["width"], 768);
  assert.equal(imageRequestBodies[0]?.["height"], 1024);
  await assert.rejects(imageTool().execute({ prompt: 42 }, new AbortController().signal), /needs a description/);
});

test("should_fail_when_the_host_cannot_store_the_image", async () => {
  installImageDaemon({ models: [IMAGE_MODEL] });
  const client = new DaemonClient(
    new AppConfig({ daemonBaseURL: DAEMON_BASE_URL, expectedChannel: "development", expectedStateDirectory: "~/.astronomical-dev" }),
  );
  const refusingBridge = new SessionBridge((request) => {
    queueMicrotask(() => window.__thintalkBridge?.reject(request.callId, "the disk is full"));
  });
  await assert.rejects(new GenerateImageTool(client, refusingBridge).execute({ prompt: "A crown" }, new AbortController().signal), /the disk is full/);
});

test("should_pass_a_stop_through_to_the_daemon_request", async () => {
  let imageRequestSignal: AbortSignal | null = null;
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    if (String(input).endsWith("/v1/models")) {
      return Response.json({ data: [IMAGE_MODEL] });
    }
    imageRequestSignal = init?.signal ?? null;
    return new Promise<Response>((_resolve, reject) => init?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError"))));
  }) as typeof fetch;
  const stopController = new AbortController();
  const toolRun = imageTool().execute({ prompt: "A crown" }, stopController.signal);
  await new Promise((resolve) => setTimeout(resolve, 10));
  stopController.abort();
  await assert.rejects(toolRun);
  assert.equal((imageRequestSignal as AbortSignal | null)?.aborted, true);
});
