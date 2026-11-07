import assert from "node:assert/strict";
import { test } from "node:test";
import { SessionBridge } from "../src/SessionBridge";

/** A bridge whose host answers every call with the given reply. */
function bridgeAnswering(hostReply: unknown): SessionBridge {
  return new SessionBridge((request) => {
    queueMicrotask(() => window.__thintalkBridge?.resolve(request.callId, hostReply));
  });
}

test("should_return_the_stored_name_the_host_gives", async () => {
  const storedName = "0f3c2a51-7d7e-4c55-9e1a-2b8d4f6a9c10.png";
  assert.equal(await bridgeAnswering({ imageName: storedName }).storeGeneratedImage("iVBORw0KGgo="), storedName);
});

test("should_reject_a_missing_or_unusable_stored_name", async () => {
  for (const hostReply of [null, {}, { imageName: 7 }, { imageName: "../sessions/a.json" }]) {
    await assert.rejects(bridgeAnswering(hostReply).storeGeneratedImage("iVBORw0KGgo="), /usable stored image name/, JSON.stringify(hostReply));
  }
});

test("should_send_the_names_to_delete", async () => {
  const sentPayloads: unknown[] = [];
  const bridge = new SessionBridge((request) => {
    sentPayloads.push(request.payload);
    queueMicrotask(() => window.__thintalkBridge?.resolve(request.callId, null));
  });
  await bridge.deleteGeneratedImages(["a.png", "b.jpg"]);
  assert.deepEqual(sentPayloads, [{ imageNames: ["a.png", "b.jpg"] }]);
});
