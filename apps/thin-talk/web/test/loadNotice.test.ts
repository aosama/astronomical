import assert from "node:assert/strict";
import { test } from "node:test";
import { ChatFailure } from "../src/ChatFailure";
import { ChatFailureKind } from "../src/ChatFailureKind";
import { WAITING_FOR_ENGINE_NOTICE, describeLoadFailure, loadNoticeAsksForAction, loadNoticeText } from "../src/LoadNotice";
import { OTHER_COPY_RUNNING_NOTICE } from "../src/OtherCopyAnsweredError";

test("should_say_the_window_is_looking_while_idle_or_loading", () => {
  assert.equal(loadNoticeText("idle", ""), "Looking for a chat model on this Mac.");
  assert.equal(loadNoticeText("loading", ""), "Looking for a chat model on this Mac.");
});

test("should_point_to_the_models_page_when_no_chat_model_is_installed", () => {
  assert.match(loadNoticeText("empty", "")!, /Models page/);
});

test("should_show_no_notice_once_a_chat_model_is_ready", () => {
  assert.equal(loadNoticeText("ready", "ignored"), null);
});

test("should_show_the_failure_in_its_own_words_after_a_failed_load", () => {
  assert.equal(loadNoticeText("failed", "The engine refused the request."), "The engine refused the request.");
});

test("should_describe_an_engine_that_is_not_answering_as_waiting", () => {
  assert.equal(describeLoadFailure(new ChatFailure(ChatFailureKind.WORKER_UNAVAILABLE)), WAITING_FOR_ENGINE_NOTICE);
});

test("should_keep_any_other_failure_message_and_stringify_non_errors", () => {
  assert.equal(describeLoadFailure(new Error("disk full")), "disk full");
  assert.equal(describeLoadFailure("plain text"), "plain text");
});

test("should_ask_the_person_to_act_only_when_another_copy_holds_the_address", () => {
  assert.equal(loadNoticeAsksForAction("failed", OTHER_COPY_RUNNING_NOTICE), true);
  assert.equal(loadNoticeAsksForAction("failed", WAITING_FOR_ENGINE_NOTICE), false, "a wait that ends by itself is not theirs to act on");
  assert.equal(loadNoticeAsksForAction("failed", ""), false);
  assert.equal(loadNoticeAsksForAction("ready", OTHER_COPY_RUNNING_NOTICE), false, "a stale message never outlives a load that succeeded");
});
