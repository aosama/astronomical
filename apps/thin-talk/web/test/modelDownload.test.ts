import assert from "node:assert/strict";
import { test } from "node:test";
import { LibraryDownloadRefusal } from "../src/DaemonClient";
import { IDLE_DOWNLOAD_STATUS } from "../src/LibraryDownloadStatus";
import type { LibraryDownloadStatus } from "../src/LibraryDownloadStatus";
import { ModelDownload, ModelDownloadPhase } from "../src/astro/ModelDownload";
import type { ModelDownloadDependencies, ModelDownloadProgress } from "../src/astro/ModelDownload";

const BRAIN_ID = "example-org/brain-4bit";

function status(state: string, bytesCompleted = 0, bytesTotal = 0, huggingfaceID: string | null = BRAIN_ID, errorCode: string | null = null): LibraryDownloadStatus {
  return { state, huggingfaceID, bytesCompleted, bytesTotal, errorCode };
}

/** A daemon and clock under the test's control: each poll is run by hand. */
class ScriptedLibrary implements ModelDownloadDependencies {
  public readonly startedIDs: string[] = [];
  public resumeCount = 0;
  public refusal: LibraryDownloadRefusal | null = null;
  public installed = false;
  public statusAnswers: LibraryDownloadStatus[] = [];
  public readonly pendingPolls: (() => void)[] = [];
  private clockMilliseconds = 0;

  public async startDownload(huggingfaceID: string): Promise<LibraryDownloadStatus> {
    this.startedIDs.push(huggingfaceID);
    if (this.refusal !== null) {
      throw this.refusal;
    }
    return status("checking_disk");
  }

  public async resumeDownload(): Promise<void> {
    this.resumeCount += 1;
  }

  public async readStatus(): Promise<LibraryDownloadStatus> {
    const answer = this.statusAnswers.shift();
    return answer !== undefined ? answer : IDLE_DOWNLOAD_STATUS;
  }

  public async isInstalled(): Promise<boolean> {
    return this.installed;
  }

  public nowMilliseconds(): number {
    return this.clockMilliseconds;
  }

  public schedule(callback: () => void): void {
    this.pendingPolls.push(callback);
  }

  /** Runs the next scheduled poll after `elapsedMilliseconds` on the clock. */
  public async runNextPoll(elapsedMilliseconds = 1_000): Promise<void> {
    const nextPoll = this.pendingPolls.shift();
    assert.ok(nextPoll !== undefined, "a poll was scheduled");
    this.clockMilliseconds += elapsedMilliseconds;
    nextPoll();
    await new Promise((resolve) => setImmediate(resolve));
  }
}

async function startedDownload(library: ScriptedLibrary): Promise<{ download: ModelDownload; reports: ModelDownloadProgress[] }> {
  const reports: ModelDownloadProgress[] = [];
  const download = new ModelDownload(BRAIN_ID, library, (progress) => reports.push(progress));
  await download.start();
  return { download, reports };
}

test("should_start_the_download_then_follow_it_until_the_model_is_installed", async () => {
  const library = new ScriptedLibrary();
  library.statusAnswers = [IDLE_DOWNLOAD_STATUS, status("downloading", 1_000, 10_000), status("downloading", 6_000, 10_000), status("publishing", 10_000, 10_000), IDLE_DOWNLOAD_STATUS];
  const { reports } = await startedDownload(library);
  assert.deepEqual(library.startedIDs, [BRAIN_ID]);
  for (let pollNumber = 0; pollNumber < 3; pollNumber += 1) {
    await library.runNextPoll(5_000);
  }
  library.installed = true;
  await library.runNextPoll();
  assert.deepEqual(reports.map((report) => report.phase), [
    ModelDownloadPhase.STARTING, ModelDownloadPhase.DOWNLOADING, ModelDownloadPhase.DOWNLOADING, ModelDownloadPhase.FINISHING, ModelDownloadPhase.READY,
  ]);
  assert.equal(reports[2]!.bytesCompleted, 6_000);
  assert.equal(Math.round(reports[2]!.secondsLeft!), 4, "5,000 bytes in 5 s with 4,000 to go");
  assert.equal(library.pendingPolls.length, 0, "polling stops once the model is installed");
});

test("should_join_a_download_of_the_same_model_that_is_already_running", async () => {
  const library = new ScriptedLibrary();
  library.statusAnswers = [status("downloading", 2_000, 10_000)];
  await startedDownload(library);
  assert.deepEqual(library.startedIDs, [], "no second request for a download already under way");
  assert.equal(library.pendingPolls.length, 1);
});

test("should_report_busy_without_requesting_when_another_model_holds_the_library", async () => {
  const library = new ScriptedLibrary();
  library.statusAnswers = [status("downloading", 1, 2, "someone-else/other-model")];
  const { reports } = await startedDownload(library);
  assert.deepEqual(library.startedIDs, []);
  assert.deepEqual(reports.map((report) => report.phase), [ModelDownloadPhase.STARTING, ModelDownloadPhase.BUSY]);
  assert.equal(library.pendingPolls.length, 0);
});

test("should_tell_busy_from_failed_when_the_daemon_refuses_to_start", async () => {
  // The refusal codes the daemon's start endpoint actually sends; null is a
  // request that never reached it.
  const expectedPhaseByRefusalCode: [string | null, ModelDownloadPhase][] = [
    ["library_busy", ModelDownloadPhase.BUSY],
    ["catalog_entry_not_found", ModelDownloadPhase.FAILED],
    ["download_failed", ModelDownloadPhase.FAILED],
    [null, ModelDownloadPhase.FAILED],
  ];
  for (const [refusalCode, expectedPhase] of expectedPhaseByRefusalCode) {
    const library = new ScriptedLibrary();
    library.refusal = new LibraryDownloadRefusal(refusalCode);
    const { reports } = await startedDownload(library);
    assert.equal(reports[reports.length - 1]!.phase, expectedPhase, String(refusalCode));
    assert.equal(library.pendingPolls.length, 0);
  }
});

test("should_report_ready_when_the_job_fails_because_the_model_is_already_installed", async () => {
  const library = new ScriptedLibrary();
  library.statusAnswers = [IDLE_DOWNLOAD_STATUS, status("failed", 0, 0, BRAIN_ID, "model_already_present")];
  const { reports } = await startedDownload(library);
  await library.runNextPoll();
  assert.equal(reports[reports.length - 1]!.phase, ModelDownloadPhase.READY);
});

test("should_resume_a_paused_download_of_the_same_model_when_joining_it", async () => {
  const library = new ScriptedLibrary();
  library.statusAnswers = [status("paused", 3_000, 10_000), status("downloading", 4_000, 10_000)];
  const { reports } = await startedDownload(library);
  assert.equal(library.resumeCount, 1, "the paused download is resumed, not watched");
  assert.deepEqual(library.startedIDs, [], "no second download is requested");
  await library.runNextPoll();
  assert.equal(reports[reports.length - 1]!.phase, ModelDownloadPhase.DOWNLOADING);
});

test("should_resume_once_then_offer_try_again_when_a_download_stays_paused", async () => {
  const library = new ScriptedLibrary();
  library.statusAnswers = [IDLE_DOWNLOAD_STATUS, ...Array.from({ length: 6 }, () => status("paused", 3_000, 10_000))];
  const { reports } = await startedDownload(library);
  for (let pausedPoll = 0; pausedPoll < 4; pausedPoll += 1) {
    await library.runNextPoll();
  }
  assert.equal(library.resumeCount, 1, "one resume, not one per poll");
  assert.notEqual(reports[reports.length - 1]!.phase, ModelDownloadPhase.FAILED, "four paused polls are still waiting");
  await library.runNextPoll();
  assert.equal(reports[reports.length - 1]!.phase, ModelDownloadPhase.FAILED);
  assert.equal(reports.some((report) => report.phase === ModelDownloadPhase.DOWNLOADING), false, "a paused job never shows a moving bar or a time");
});

test("should_still_request_the_download_when_the_first_status_read_fails", async () => {
  const library = new ScriptedLibrary();
  library.readStatus = async () => {
    throw new Error("the daemon blinked");
  };
  await startedDownload(library);
  assert.deepEqual(library.startedIDs, [BRAIN_ID]);
});

test("should_report_failed_when_the_job_fails", async () => {
  const library = new ScriptedLibrary();
  library.statusAnswers = [IDLE_DOWNLOAD_STATUS, status("failed")];
  const { reports } = await startedDownload(library);
  await library.runNextPoll();
  assert.equal(reports[reports.length - 1]!.phase, ModelDownloadPhase.FAILED);
  assert.equal(library.pendingPolls.length, 0);
});

test("should_give_up_after_ten_idle_polls_when_the_model_never_lands", async () => {
  const library = new ScriptedLibrary();
  const { reports } = await startedDownload(library);
  for (let idlePoll = 0; idlePoll < 9; idlePoll += 1) {
    await library.runNextPoll();
  }
  assert.notEqual(reports[reports.length - 1]!.phase, ModelDownloadPhase.FAILED, "nine idle polls are still waiting");
  await library.runNextPoll();
  assert.equal(reports[reports.length - 1]!.phase, ModelDownloadPhase.FAILED);
});

test("should_keep_polling_through_a_failed_status_read", async () => {
  const library = new ScriptedLibrary();
  const { reports } = await startedDownload(library);
  library.readStatus = async () => {
    throw new Error("the daemon blinked");
  };
  await library.runNextPoll();
  assert.equal(library.pendingPolls.length, 1, "the next poll is still scheduled");
  assert.equal(reports.length, 1, "a missed read reports nothing");
});

test("should_stop_reporting_and_polling_once_stopped", async () => {
  const library = new ScriptedLibrary();
  library.statusAnswers = [IDLE_DOWNLOAD_STATUS, status("downloading", 1, 10)];
  const { download, reports } = await startedDownload(library);
  download.stop();
  await library.runNextPoll();
  assert.equal(reports.length, 1);
  assert.equal(library.pendingPolls.length, 0);
});
