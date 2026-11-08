import assert from "node:assert/strict";
import { test } from "node:test";
import { IDLE_DOWNLOAD_STATUS, parseLibraryDownloadStatus, refusalCodeFromErrorDocument } from "../src/LibraryDownloadStatus";

test("should_read_a_running_download", () => {
  const downloadText = JSON.stringify({ state: "downloading", huggingface_id: "example-org/model", bytes_completed: 40, bytes_total: 100, error_code: null });
  assert.deepEqual(parseLibraryDownloadStatus(downloadText), {
    state: "downloading", huggingfaceID: "example-org/model", bytesCompleted: 40, bytesTotal: 100, errorCode: null,
  });
});

test("should_read_a_failed_download_with_its_code", () => {
  const downloadText = JSON.stringify({ state: "failed", huggingface_id: "example-org/model", bytes_completed: 0, bytes_total: 0, error_code: "insufficient_disk" });
  assert.equal(parseLibraryDownloadStatus(downloadText).errorCode, "insufficient_disk");
});

test("should_treat_unreadable_or_wrongly_typed_documents_as_idle", () => {
  assert.deepEqual(parseLibraryDownloadStatus("not json"), IDLE_DOWNLOAD_STATUS);
  assert.deepEqual(parseLibraryDownloadStatus("null"), IDLE_DOWNLOAD_STATUS);
  const wronglyTyped = parseLibraryDownloadStatus(JSON.stringify({ state: 3, bytes_completed: -1, bytes_total: "x" }));
  assert.deepEqual(wronglyTyped, IDLE_DOWNLOAD_STATUS);
});

test("should_read_the_refusal_code_from_an_error_document_or_none", () => {
  assert.equal(refusalCodeFromErrorDocument(JSON.stringify({ error: { code: "library_busy", message: "busy" } })), "library_busy");
  assert.equal(refusalCodeFromErrorDocument(JSON.stringify({ error: {} })), null);
  assert.equal(refusalCodeFromErrorDocument(JSON.stringify({})), null);
  assert.equal(refusalCodeFromErrorDocument("<html>"), null);
});
