import assert from "node:assert/strict";
import { test } from "node:test";
import { AppConfig } from "../src/AppConfig";
import { ChatFailure } from "../src/ChatFailure";
import { ChatFailureKind } from "../src/ChatFailureKind";
import { DaemonClient } from "../src/DaemonClient";
import { GenerateImageTool } from "../src/GenerateImageTool";
import { ImageModelSelection } from "../src/ImageModelSelection";
import { imageMimeType } from "../src/GeneratedImageReview";
import { SessionBridge } from "../src/SessionBridge";
import { DAEMON_BASE_URL } from "./support/scriptedChatFetch";

const STORED_IMAGE_NAME = "0f3c2a51-7d7e-4c55-9e1a-2b8d4f6a9c10.png";
const IMAGE_MODEL = { id: "FLUX.2-klein-4B", input_modalities: ["text"], supported_endpoints: ["/v1/images/generations"] };
const CHAT_MODEL = { id: "chat-model", input_modalities: ["text"], supported_endpoints: ["/v1/chat/completions"] };
const VISION_MODEL = { id: "vision-model", input_modalities: ["text", "image"], supported_endpoints: ["/v1/chat/completions"] };
const TOOL_CONTEXT = { conversationID: "conversation-1", modelID: "chat-model" };

interface ImageDaemonScript {
  models: unknown[];
  imageStatus?: number;
  imageBody?: unknown;
  /** What the vision model answers; a number is an HTTP failure status. */
  imageDescription?: string | number;
  describeRequestBodies?: Record<string, unknown>[];
  /** The model list fails after this many successful reads. */
  failModelListAfter?: number;
  /** The Library catalog; without one the endpoint is not found. */
  libraryCatalog?: unknown;
}

/** A daemon that lists models and answers image requests, recording each
 * image request body. */
function installImageDaemon(script: ImageDaemonScript): Record<string, unknown>[] {
  const imageRequestBodies: Record<string, unknown>[] = [];
  let modelListRequestCount = 0;
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith("/v1/models")) {
      modelListRequestCount += 1;
      if (script.failModelListAfter != null && modelListRequestCount > script.failModelListAfter) {
        return Response.json({ error: { message: "server_capacity: busy" } }, { status: 503 });
      }
      return Response.json({ data: script.models });
    }
    if (url.endsWith("/v1/library/catalog") && script.libraryCatalog !== undefined) {
      return Response.json(script.libraryCatalog);
    }
    if (url.endsWith("/v1/images/generations")) {
      imageRequestBodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
      const imageBody = script.imageBody !== undefined ? script.imageBody : { data: [{ b64_json: "iVBORw0KGgo=", mime_type: "image/png" }] };
      return Response.json(imageBody, { status: script.imageStatus ?? 200 });
    }
    if (url.endsWith("/v1/chat/completions")) {
      script.describeRequestBodies?.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
      const imageDescription = script.imageDescription ?? "A red rose on a stone balcony.";
      if (typeof imageDescription === "number") {
        return Response.json({ error: { message: "server_capacity: busy" } }, { status: imageDescription });
      }
      return Response.json({ choices: [{ message: { role: "assistant", content: imageDescription }, finish_reason: "stop" }] });
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

function imageTool(storedImages: string[] = [], imageModelSelection = new ImageModelSelection()): GenerateImageTool {
  const client = new DaemonClient(
    new AppConfig({ daemonBaseURL: DAEMON_BASE_URL, expectedChannel: "development", expectedStateDirectory: "~/.astronomical-dev" }),
  );
  return new GenerateImageTool(client, storingBridge(storedImages), imageModelSelection);
}

const FOUR_BIT_IMAGE_MODEL = { id: "flux2-klein-4b-4bit", input_modalities: ["text"], supported_endpoints: ["/v1/images/generations"] };
const PICTURES_LIBRARY = {
  entries: [
    { display_name: "FLUX.2 Klein 4B", role: "image_generation", get_started: false, requestable_model_id: "FLUX.2-klein-4B" },
    { display_name: "FLUX.2 Klein 4B 4-bit", role: "image_generation", get_started: true, requestable_model_id: "flux2-klein-4b-4bit" },
  ],
};

test("should_draw_with_the_library_pictures_pick_and_name_it", async () => {
  const imageRequestBodies = installImageDaemon({ models: [CHAT_MODEL, IMAGE_MODEL, FOUR_BIT_IMAGE_MODEL], libraryCatalog: PICTURES_LIBRARY });

  const toolOutput = await imageTool().execute({ prompt: "A crown" }, new AbortController().signal, TOOL_CONTEXT);

  assert.equal(imageRequestBodies[0]?.["model"], "flux2-klein-4b-4bit");
  assert.equal(toolOutput.generatedImageModelName, "FLUX.2 Klein 4B 4-bit");
  assert.match(toolOutput.text, /with FLUX\.2 Klein 4B 4-bit and showed it/);
});

test("should_draw_with_the_model_the_person_chose", async () => {
  const imageRequestBodies = installImageDaemon({ models: [CHAT_MODEL, IMAGE_MODEL, FOUR_BIT_IMAGE_MODEL], libraryCatalog: PICTURES_LIBRARY });

  const toolOutput = await imageTool([], new ImageModelSelection("FLUX.2-klein-4B"))
    .execute({ prompt: "A crown" }, new AbortController().signal, TOOL_CONTEXT);

  assert.equal(imageRequestBodies[0]?.["model"], "FLUX.2-klein-4B");
  assert.equal(toolOutput.generatedImageModelName, "FLUX.2 Klein 4B");
});

test("should_still_draw_when_the_library_cannot_be_read", async () => {
  const imageRequestBodies = installImageDaemon({ models: [CHAT_MODEL, IMAGE_MODEL] });

  const toolOutput = await imageTool().execute({ prompt: "A crown" }, new AbortController().signal, TOOL_CONTEXT);

  assert.equal(imageRequestBodies[0]?.["model"], "FLUX.2-klein-4B");
  assert.equal(toolOutput.generatedImageModelName, "FLUX.2-klein-4B", "without the Library the id is the name");
});

test("should_generate_with_the_only_image_model_store_the_image_and_report_it", async () => {
  const imageRequestBodies = installImageDaemon({ models: [CHAT_MODEL, IMAGE_MODEL] });
  const storedImages: string[] = [];
  const toolOutput = await imageTool(storedImages).execute({ prompt: "A rose in Verona at dusk", shape: "landscape" }, new AbortController().signal, TOOL_CONTEXT);
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
  await imageTool().execute({ prompt: "A crown", shape: "hexagon" }, new AbortController().signal, TOOL_CONTEXT);
  assert.equal(imageRequestBodies[0]?.["width"], 1024);
  assert.equal(imageRequestBodies[0]?.["height"], 1024);
});

test("should_refuse_an_empty_prompt_without_calling_the_daemon", async () => {
  const imageRequestBodies = installImageDaemon({ models: [IMAGE_MODEL] });
  await assert.rejects(imageTool().execute({ prompt: "   " }, new AbortController().signal, TOOL_CONTEXT), /needs a description/);
  assert.equal(imageRequestBodies.length, 0);
});

test("should_say_so_when_no_image_model_is_installed", async () => {
  installImageDaemon({ models: [CHAT_MODEL] });
  await assert.rejects(imageTool().execute({ prompt: "A crown" }, new AbortController().signal, TOOL_CONTEXT), /No image model is installed/);
});

test("should_report_a_daemon_refusal_and_a_response_without_an_image", async () => {
  installImageDaemon({ models: [IMAGE_MODEL], imageStatus: 503, imageBody: { error: { message: "chat_worker_unavailable" } } });
  await assert.rejects(
    imageTool().execute({ prompt: "A crown" }, new AbortController().signal, TOOL_CONTEXT),
    (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.WORKER_UNAVAILABLE,
  );
  installImageDaemon({ models: [IMAGE_MODEL], imageBody: { data: [] } });
  await assert.rejects(
    imageTool().execute({ prompt: "A crown" }, new AbortController().signal, TOOL_CONTEXT),
    (error: unknown) => error instanceof ChatFailure && error.kind === ChatFailureKind.MALFORMED_MODEL_OUTPUT,
  );
});

test("should_make_a_portrait_image_and_refuse_a_prompt_that_is_not_text", async () => {
  const imageRequestBodies = installImageDaemon({ models: [IMAGE_MODEL] });
  await imageTool().execute({ prompt: "A tall tower", shape: "portrait" }, new AbortController().signal, TOOL_CONTEXT);
  assert.equal(imageRequestBodies[0]?.["width"], 768);
  assert.equal(imageRequestBodies[0]?.["height"], 1024);
  await assert.rejects(imageTool().execute({ prompt: 42 }, new AbortController().signal, TOOL_CONTEXT), /needs a description/);
});

test("should_fail_when_the_host_cannot_store_the_image", async () => {
  installImageDaemon({ models: [IMAGE_MODEL] });
  const client = new DaemonClient(
    new AppConfig({ daemonBaseURL: DAEMON_BASE_URL, expectedChannel: "development", expectedStateDirectory: "~/.astronomical-dev" }),
  );
  const refusingBridge = new SessionBridge((request) => {
    queueMicrotask(() => window.__thintalkBridge?.reject(request.callId, "the disk is full"));
  });
  await assert.rejects(new GenerateImageTool(client, refusingBridge, new ImageModelSelection()).execute({ prompt: "A crown" }, new AbortController().signal, TOOL_CONTEXT), /the disk is full/);
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

test("should_hand_the_image_itself_to_a_chat_model_that_can_see", async () => {
  const describeRequestBodies: Record<string, unknown>[] = [];
  const seeingChatModel = { ...VISION_MODEL, id: "chat-model" };
  installImageDaemon({ models: [seeingChatModel, IMAGE_MODEL, VISION_MODEL], describeRequestBodies });
  const toolOutput = await imageTool().execute({ prompt: "A rose in Verona at dusk" }, new AbortController().signal, TOOL_CONTEXT);
  assert.deepEqual(toolOutput.imageForModel, { base64: "iVBORw0KGgo=", mimeType: "image/png" });
  assert.match(toolOutput.text, /attached right after this result/);
  assert.equal(describeRequestBodies.length, 0, "no second model describes it: the chat model looks itself");
});

test("should_have_another_vision_model_look_for_a_text_only_chat_model", async () => {
  const describeRequestBodies: Record<string, unknown>[] = [];
  installImageDaemon({ models: [CHAT_MODEL, IMAGE_MODEL, VISION_MODEL], imageDescription: "A rose with three stems.", describeRequestBodies });
  const toolOutput = await imageTool().execute({ prompt: "A rose in Verona at dusk" }, new AbortController().signal, TOOL_CONTEXT);
  assert.equal(toolOutput.imageForModel, undefined, "a text-only model is never sent pixels");
  assert.match(toolOutput.text, /vision-model looked at it for you/);
  assert.match(toolOutput.text, /A rose with three stems\./);
  assert.equal(describeRequestBodies[0]?.["model"], "vision-model");
  const sentContent = JSON.stringify(describeRequestBodies[0]?.["messages"]);
  assert.match(sentContent, /data:image\/png;base64,iVBORw0KGgo=/, "the generated image itself is described");
  assert.match(sentContent, /Prompt: A rose in Verona at dusk/, "the vision model is told what was asked for");
});

test("should_tell_a_text_only_model_it_cannot_see_when_no_vision_model_is_installed", async () => {
  installImageDaemon({ models: [CHAT_MODEL, IMAGE_MODEL] });
  const toolOutput = await imageTool().execute({ prompt: "A crown" }, new AbortController().signal, TOOL_CONTEXT);
  assert.equal(toolOutput.imageForModel, undefined);
  assert.match(toolOutput.text, /cannot see it; do not describe/);
});

test("should_keep_the_image_and_say_so_when_looking_at_it_fails", async () => {
  const storedImages: string[] = [];
  installImageDaemon({ models: [IMAGE_MODEL, VISION_MODEL], imageDescription: 503 });
  const toolOutput = await imageTool(storedImages).execute({ prompt: "A crown" }, new AbortController().signal, TOOL_CONTEXT);
  assert.equal(toolOutput.generatedImageName, STORED_IMAGE_NAME, "the image the user sees is still reported");
  assert.match(toolOutput.text, /Looking at the image failed/);
  assert.match(toolOutput.text, /do not describe/);
});

test("should_recognise_a_jpeg_and_default_to_png", () => {
  assert.equal(imageMimeType("/9j/4AAQSkZJRg=="), "image/jpeg");
  assert.equal(imageMimeType("iVBORw0KGgo="), "image/png");
  assert.equal(imageMimeType(""), "image/png");
});

test("should_keep_the_stored_image_when_the_vision_model_list_cannot_be_read", async () => {
  const storedImages: string[] = [];
  installImageDaemon({ models: [CHAT_MODEL, IMAGE_MODEL, VISION_MODEL], failModelListAfter: 1 });
  const toolOutput = await imageTool(storedImages).execute({ prompt: "A crown" }, new AbortController().signal, TOOL_CONTEXT);
  assert.equal(toolOutput.generatedImageName, STORED_IMAGE_NAME, "the image on screen is still reported");
  assert.equal(storedImages.length, 1);
  assert.match(toolOutput.text, /cannot see it; do not describe/);
});
