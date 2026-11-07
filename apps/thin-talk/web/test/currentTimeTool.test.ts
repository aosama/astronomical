import assert from "node:assert/strict";
import { test } from "node:test";
import { CurrentTimeTool } from "../src/CurrentTimeTool";

test("should_report_the_pinned_time_with_its_iso_form_and_time_zone", async () => {
  const pinnedInstant = new Date("2026-11-07T19:30:00.000Z");
  const currentTime = await new CurrentTimeTool(() => pinnedInstant).execute();
  assert.match(currentTime, /2026/);
  assert.match(currentTime, /ISO 8601 2026-11-07T19:30:00\.000Z/);
  assert.match(currentTime, /time zone \S+/);
});

test("should_declare_no_arguments_and_leave_extra_ones_allowed", () => {
  // The runner test proves a call with stray arguments succeeds end to end;
  // this pins the schema that makes it possible.
  const tool = new CurrentTimeTool();
  assert.deepEqual(tool.parameters.properties, {});
  assert.equal(tool.parameters.additionalProperties, undefined);
});

test("should_read_the_clock_on_every_call", async () => {
  let clockReadCount = 0;
  const tool = new CurrentTimeTool(() => {
    clockReadCount += 1;
    return new Date(0);
  });
  await tool.execute();
  await tool.execute();
  assert.equal(clockReadCount, 2);
});
