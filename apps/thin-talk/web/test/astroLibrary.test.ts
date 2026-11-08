import assert from "node:assert/strict";
import { test } from "node:test";
import type { DaemonClient } from "../src/DaemonClient";
import { AstroLibrary } from "../src/astro/AstroLibrary";
import { ModelDownloadPhase } from "../src/astro/ModelDownload";
import type { ModelDownloadProgress } from "../src/astro/ModelDownload";
import { waitFor } from "./support/dom";

const BRAIN_ID = "example-org/brain-4bit";

/** A daemon whose download finishes on the second status poll. */
function finishingDaemon(): { client: DaemonClient; startRequests: string[] } {
  const startRequests: string[] = [];
  let statusReads = 0;
  let isInstalled = false;
  const client = {
    startLibraryDownload: async (huggingfaceID: string) => {
      startRequests.push(huggingfaceID);
      return { state: "checking_disk", huggingfaceID, bytesCompleted: 0, bytesTotal: 0, errorCode: null };
    },
    resumeLibraryDownload: async () => undefined,
    fetchLibraryDownloadStatus: async () => {
      statusReads += 1;
      if (statusReads === 2) {
        return { state: "downloading", huggingfaceID: BRAIN_ID, bytesCompleted: 5, bytesTotal: 10, errorCode: null };
      }
      if (statusReads >= 3) {
        isInstalled = true;
      }
      return { state: "idle", huggingfaceID: null, bytesCompleted: 0, bytesTotal: 0, errorCode: null };
    },
    fetchLibraryCatalogText: async () => JSON.stringify({ entries: [{ huggingface_id: BRAIN_ID, ready_on_this_mac: isInstalled }] }),
  };
  return { client: client as unknown as DaemonClient, startRequests };
}

test("should_follow_one_download_for_every_screen_watching_the_same_model", async () => {
  const { client, startRequests } = finishingDaemon();
  const library = new AstroLibrary(client);
  let installNotices = 0;
  library.onModelInstalled(() => {
    installNotices += 1;
  });
  const firstScreenPhases: ModelDownloadPhase[] = [];
  const secondScreenPhases: ModelDownloadPhase[] = [];
  library.followDownload(BRAIN_ID, (progress: ModelDownloadProgress) => firstScreenPhases.push(progress.phase));
  const stopSecondScreen = library.followDownload(BRAIN_ID, (progress: ModelDownloadProgress) => secondScreenPhases.push(progress.phase));
  assert.deepEqual(secondScreenPhases, [ModelDownloadPhase.STARTING], "a late screen gets the latest progress at once");
  stopSecondScreen();
  await waitFor(() => firstScreenPhases.includes(ModelDownloadPhase.READY), "the download to land");
  assert.deepEqual(startRequests, [BRAIN_ID], "one download however many screens watch it");
  assert.deepEqual(secondScreenPhases, [ModelDownloadPhase.STARTING], "a screen that stopped listening hears nothing more");
  assert.equal(installNotices, 1, "the chat window hears about the new model once");
});

test("should_start_a_fresh_download_after_the_previous_one_finished", async () => {
  const { client, startRequests } = finishingDaemon();
  const library = new AstroLibrary(client);
  const phases: ModelDownloadPhase[] = [];
  library.followDownload(BRAIN_ID, (progress: ModelDownloadProgress) => phases.push(progress.phase));
  await waitFor(() => phases.includes(ModelDownloadPhase.READY), "the first download to land");
  library.followDownload(BRAIN_ID, () => undefined);
  // A finished follower is forgotten, so trying again starts afresh.
  await waitFor(() => startRequests.length === 2, "a second download request");
});
