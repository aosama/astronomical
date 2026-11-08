import assert from "node:assert/strict";
import { test } from "node:test";
import { DownloadTimeEstimate, describeGigabytes, describeTimeLeft } from "../src/astro/DownloadTimeEstimate";

const GIGABYTE = 1_000_000_000;

test("should_estimate_the_time_left_from_the_recent_speed", () => {
  const estimate = new DownloadTimeEstimate();
  estimate.record(0, 0);
  estimate.record(10_000, 100_000_000);
  // 10 MB/s with 900 MB to go: 90 seconds.
  assert.equal(Math.round(estimate.secondsLeft(GIGABYTE)!), 90);
});

test("should_promise_no_time_before_two_seconds_of_history_or_any_progress", () => {
  const estimate = new DownloadTimeEstimate();
  assert.equal(estimate.secondsLeft(GIGABYTE), null, "no samples");
  estimate.record(0, 0);
  estimate.record(1_000, 50_000_000);
  assert.equal(estimate.secondsLeft(GIGABYTE), null, "one second is too little history");
  const stalledEstimate = new DownloadTimeEstimate();
  stalledEstimate.record(0, 500);
  stalledEstimate.record(5_000, 500);
  assert.equal(stalledEstimate.secondsLeft(GIGABYTE), null, "no bytes moved");
  assert.equal(new DownloadTimeEstimate().secondsLeft(0), null, "unknown total");
});

test("should_measure_a_resumed_download_from_its_own_speed_not_the_bytes_already_on_disk", () => {
  const estimate = new DownloadTimeEstimate();
  estimate.record(0, 4 * GIGABYTE);
  estimate.record(10_000, 4 * GIGABYTE + 100_000_000);
  assert.equal(Math.round(estimate.secondsLeft(5 * GIGABYTE)!), 90);
});

test("should_forget_samples_older_than_the_window_so_a_speed_change_shows", () => {
  const estimate = new DownloadTimeEstimate();
  estimate.record(0, 0);
  estimate.record(10_000, 10_000_000);
  estimate.record(20_000, 20_000_000);
  estimate.record(30_000, 220_000_000);
  // Only the last 15 s count: 210 MB over 20 s at most, so far faster than the start's 1 MB/s.
  assert.ok(estimate.secondsLeft(GIGABYTE)! < 100);
});

test("should_restart_the_estimate_when_the_byte_count_goes_backwards", () => {
  const estimate = new DownloadTimeEstimate();
  estimate.record(0, 0);
  estimate.record(10_000, 500_000_000);
  estimate.record(11_000, 100_000_000);
  assert.equal(estimate.secondsLeft(GIGABYTE), null);
});

test("should_describe_waits_in_plain_words", () => {
  assert.equal(describeTimeLeft(null), "working out the time…");
  assert.equal(describeTimeLeft(0), "less than a minute left");
  assert.equal(describeTimeLeft(59), "less than a minute left");
  assert.equal(describeTimeLeft(60), "about 1 minute left");
  assert.equal(describeTimeLeft(6 * 60), "about 6 minutes left");
  assert.equal(describeTimeLeft(89 * 60), "about 89 minutes left");
  assert.equal(describeTimeLeft(90 * 60), "about 2 hours left");
  assert.equal(describeTimeLeft(60 * 60 + 1), "about 60 minutes left");
});

test("should_describe_sizes_in_decimal_gigabytes_with_one_decimal", () => {
  assert.equal(describeGigabytes(5_970_332_199), "6.0 GB");
  assert.equal(describeGigabytes(4_619_700_724), "4.6 GB");
  assert.equal(describeGigabytes(0), "0.0 GB");
  assert.equal(describeGigabytes(-5), "0.0 GB");
});
