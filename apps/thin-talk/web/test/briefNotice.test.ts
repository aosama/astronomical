import assert from "node:assert/strict";
import { test } from "node:test";
import { BRIEF_NOTICE_MILLISECONDS, BriefNotice } from "../src/BriefNotice";

test("should_show_a_notice_and_hide_it_after_its_moment", (testContext) => {
  testContext.mock.timers.enable({ apis: ["setTimeout"] });
  const noticeElement = document.createElement("p");
  noticeElement.hidden = true;
  const notice = new BriefNotice(noticeElement);

  notice.show("Saved “Mitosis.pdf”");
  assert.equal(noticeElement.hidden, false);
  assert.equal(noticeElement.textContent, "Saved “Mitosis.pdf”");
  testContext.mock.timers.tick(BRIEF_NOTICE_MILLISECONDS - 1);
  assert.equal(noticeElement.hidden, false);
  testContext.mock.timers.tick(1);
  assert.equal(noticeElement.hidden, true);
});

test("should_restart_the_moment_when_a_second_notice_follows_the_first", (testContext) => {
  testContext.mock.timers.enable({ apis: ["setTimeout"] });
  const noticeElement = document.createElement("p");
  const notice = new BriefNotice(noticeElement);

  notice.show("first");
  testContext.mock.timers.tick(BRIEF_NOTICE_MILLISECONDS - 10);
  notice.show("second");
  testContext.mock.timers.tick(20);

  assert.equal(noticeElement.hidden, false, "the first notice's timer must not hide the second");
  assert.equal(noticeElement.textContent, "second");
});
