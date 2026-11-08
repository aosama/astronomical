import assert from "node:assert/strict";
import { test } from "node:test";
import { AppConfig } from "../src/AppConfig";
import { DaemonClient } from "../src/DaemonClient";
import { ImageModelDirectory } from "../src/ImageModelDirectory";
import { DAEMON_BASE_URL } from "./support/scriptedChatFetch";

const FULL_PRECISION_IMAGE_MODEL = { id: "FLUX.2-klein-4B", input_modalities: ["text"], supported_endpoints: ["/v1/images/generations"] };
const FOUR_BIT_IMAGE_MODEL = { id: "flux2-klein-4b-4bit", input_modalities: ["text"], supported_endpoints: ["/v1/images/generations"] };
const IMAGE_GENERATION_CATALOG = {
  entries: [
    { display_name: "FLUX.2 Klein 4B full precision", role: "image_generation", get_started: false, requestable_model_id: "FLUX.2-klein-4B" },
    { display_name: "FLUX.2 Klein 4B 4-bit", role: "image_generation", get_started: true, requestable_model_id: "flux2-klein-4b-4bit" },
  ],
};

/** Each catalog read answers with the next scripted reply: a document, an
 * HTTP status, or text that is not a catalog. */
function installLibraryDaemon(models: unknown[], catalogReplies: (unknown | number | string)[]): string[] {
  const requestedPaths: string[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = new URL(String(input));
    requestedPaths.push(url.pathname);
    if (url.pathname === "/v1/models") {
      return Response.json({ data: models });
    }
    if (url.pathname === "/v1/library/catalog") {
      const catalogReply = catalogReplies.length > 1 ? catalogReplies.shift() : catalogReplies[0];
      if (typeof catalogReply === "number") {
        return Response.json({ error: { message: "busy" } }, { status: catalogReply });
      }
      return typeof catalogReply === "string" ? new Response(catalogReply) : Response.json(catalogReply);
    }
    return new Response("not found", { status: 404 });
  }) as typeof fetch;
  return requestedPaths;
}

function imageModelDirectory(): ImageModelDirectory {
  return new ImageModelDirectory(
    new DaemonClient(new AppConfig({ daemonBaseURL: DAEMON_BASE_URL, expectedChannel: "development", expectedStateDirectory: "~/.astronomical-dev" })),
  );
}

function names(imageModels: { name: string }[]): string[] {
  return imageModels.map((imageModel) => imageModel.name);
}

test("should_name_and_order_installed_models_as_the_library_does", async () => {
  installLibraryDaemon([FULL_PRECISION_IMAGE_MODEL, FOUR_BIT_IMAGE_MODEL], [IMAGE_GENERATION_CATALOG]);
  assert.deepEqual(names(await imageModelDirectory().load()), ["FLUX.2 Klein 4B 4-bit", "FLUX.2 Klein 4B full precision"]);
});

test("should_keep_the_last_library_answer_when_a_later_read_fails", async () => {
  installLibraryDaemon([FULL_PRECISION_IMAGE_MODEL, FOUR_BIT_IMAGE_MODEL], [IMAGE_GENERATION_CATALOG, 503, "not a catalog"]);
  const directory = imageModelDirectory();
  await directory.load();

  const afterFailedRead = await directory.load();
  const afterUnreadableRead = await directory.load();

  // Without the kept answer the Image Generation pick would silently become the first installed model.
  assert.equal(afterFailedRead[0]?.isImageGenerationPick, true);
  assert.deepEqual(names(afterFailedRead), ["FLUX.2 Klein 4B 4-bit", "FLUX.2 Klein 4B full precision"]);
  assert.deepEqual(names(afterUnreadableRead), ["FLUX.2 Klein 4B 4-bit", "FLUX.2 Klein 4B full precision"]);
});

test("should_use_model_ids_when_the_library_has_never_answered", async () => {
  installLibraryDaemon([FULL_PRECISION_IMAGE_MODEL, FOUR_BIT_IMAGE_MODEL], [404]);
  assert.deepEqual(names(await imageModelDirectory().load()), ["FLUX.2-klein-4B", "flux2-klein-4b-4bit"]);
});

test("should_not_read_the_library_when_no_image_model_is_installed", async () => {
  const requestedPaths = installLibraryDaemon([], [IMAGE_GENERATION_CATALOG]);
  assert.deepEqual(await imageModelDirectory().load(), []);
  assert.equal(requestedPaths.includes("/v1/library/catalog"), false);
});

test("should_pass_a_stop_during_the_library_read_through_instead_of_falling_back", async () => {
  const stop = new AbortController();
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    if (new URL(String(input)).pathname === "/v1/models") {
      return Response.json({ data: [FULL_PRECISION_IMAGE_MODEL] });
    }
    // The person stops the turn while the Library is being read.
    stop.abort();
    throw new DOMException("The operation was aborted.", "AbortError");
  }) as typeof fetch;

  await assert.rejects(imageModelDirectory().load(stop.signal));
});
