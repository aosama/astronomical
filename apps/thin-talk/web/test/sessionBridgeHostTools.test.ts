import assert from "node:assert/strict";
import { test } from "node:test";
import { scriptedBridge } from "./support/scriptedBridge";

async function hostToolUseFor(hostAnswer: unknown): Promise<unknown> {
  return scriptedBridge({ useHostTool: hostAnswer }).useHostTool("read_file", "c1", new AbortController().signal);
}

test("should_read_each_host_outcome", async () => {
  assert.deepEqual(await hostToolUseFor({ outcome: "approved" }), { outcome: "approved" });
  assert.deepEqual(await hostToolUseFor({ outcome: "refused" }), { outcome: "refused" });
  assert.deepEqual(await hostToolUseFor({ outcome: "unavailable", reason: "closed" }), { outcome: "unavailable", reason: "closed" });
  assert.deepEqual(await hostToolUseFor({ outcome: "chosen", fileName: "a.txt", text: "hi" }), { outcome: "chosenText", fileName: "a.txt", text: "hi" });
  const pngImage = { fileName: "a.png", base64: "iVBO", mimeType: "image/png" };
  assert.deepEqual(await hostToolUseFor({ outcome: "chosen", ...pngImage }), { outcome: "chosenImage", ...pngImage });
});

test("should_reject_an_unknown_outcome_or_an_unusable_file", async () => {
  await assert.rejects(hostToolUseFor({ outcome: "maybe" }), /unknown tool outcome/);
  await assert.rejects(hostToolUseFor(null), /unknown tool outcome/);
  await assert.rejects(hostToolUseFor({ outcome: "chosen", fileName: "a.svg", base64: "x", mimeType: "image/svg+xml" }), /unreadable file/);
});

test("should_not_call_the_host_when_already_stopped", async () => {
  const sentOps: string[] = [];
  const stopController = new AbortController();
  stopController.abort();
  const bridge = scriptedBridge({ useHostTool: { outcome: "approved" } });
  const originalPost = (bridge as unknown as { post: (request: { op: string }) => void }).post;
  (bridge as unknown as { post: (request: { op: string }) => void }).post = (request) => {
    sentOps.push(request.op);
    originalPost(request);
  };
  await assert.rejects(bridge.useHostTool("read_file", "c1", stopController.signal), /was stopped/);
  assert.deepEqual(sentOps, []);
});
