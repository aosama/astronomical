import assert from "node:assert/strict";
import { test } from "node:test";
import type { DaemonClient } from "../src/DaemonClient";
import { ModelDescriptionDirectory } from "../src/ModelDescriptionDirectory";

/** A client whose catalog answers are scripted one read at a time. */
function clientAnswering(catalogAnswers: Array<string | Error>): DaemonClient {
  const remainingAnswers = [...catalogAnswers];
  return {
    fetchLibraryCatalogText: async () => {
      const nextAnswer = remainingAnswers.shift();
      if (nextAnswer === undefined || nextAnswer instanceof Error) {
        throw nextAnswer ?? new Error("no scripted answer");
      }
      return nextAnswer;
    },
  } as unknown as DaemonClient;
}

const ORNITH_CATALOG = JSON.stringify({ entries: [{ requestable_model_id: "ornith-folder", short_name: "Ornith 1.5 9B" }] });

test("should_read_plain_names_from_the_library", async () => {
  const directory = new ModelDescriptionDirectory(clientAnswering([ORNITH_CATALOG]), () => undefined);
  await directory.refresh();
  assert.equal(directory.descriptions.get("ornith-folder")?.shortName, "Ornith 1.5 9B");
});

test("should_keep_the_last_names_when_a_later_read_fails", async () => {
  const directory = new ModelDescriptionDirectory(clientAnswering([ORNITH_CATALOG, new Error("engine busy")]), () => undefined);
  await directory.refresh();
  await directory.refresh();
  assert.equal(directory.descriptions.get("ornith-folder")?.shortName, "Ornith 1.5 9B");
});

test("should_keep_the_last_names_when_a_later_answer_is_not_a_catalog", async () => {
  const directory = new ModelDescriptionDirectory(clientAnswering([ORNITH_CATALOG, "<html>not a catalog</html>"]), () => undefined);
  await directory.refresh();
  await directory.refresh();
  assert.equal(directory.descriptions.size, 1);
});

test("should_announce_only_reads_that_produced_names", async () => {
  let announcementCount = 0;
  const directory = new ModelDescriptionDirectory(
    clientAnswering([ORNITH_CATALOG, new Error("engine busy"), "not a catalog"]),
    () => { announcementCount += 1; },
  );
  await directory.refresh();
  await directory.refresh();
  await directory.refresh();
  assert.equal(announcementCount, 1);
});

test("should_start_empty_and_stay_empty_when_the_library_never_answers", async () => {
  const directory = new ModelDescriptionDirectory(clientAnswering([new Error("offline")]), () => undefined);
  assert.equal(directory.descriptions.size, 0);
  await directory.refresh();
  assert.equal(directory.descriptions.size, 0);
});
