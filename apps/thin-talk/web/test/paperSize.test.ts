import assert from "node:assert/strict";
import { test } from "node:test";
import { AppConfig } from "../src/AppConfig";
import { A4_PAPER, LETTER_PAPER, paperSizeFromHost, textHeightInches, textWidthInches } from "../src/export/PaperSize";

test("should_read_the_hosts_paper_in_points_as_inches", () => {
  assert.deepEqual(paperSizeFromHost({ widthPoints: 612, heightPoints: 792 }), { widthInches: 8.5, heightInches: 11 });
  const a4Paper = paperSizeFromHost({ widthPoints: 595, heightPoints: 842 });
  assert.ok(Math.abs(a4Paper.widthInches - A4_PAPER.widthInches) < 0.01);
  assert.ok(Math.abs(a4Paper.heightInches - A4_PAPER.heightInches) < 0.01);
});

test("should_fall_back_to_letter_when_the_host_names_no_usable_paper", () => {
  const unusablePapers: unknown[] = [undefined, null, "A4", {}, { widthPoints: 612 }, { widthPoints: "612", heightPoints: 792 }, { widthPoints: Number.NaN, heightPoints: 792 }, { widthPoints: 100, heightPoints: 792 }, { widthPoints: 612, heightPoints: 5000 }];
  for (const unusablePaper of unusablePapers) {
    assert.deepEqual(paperSizeFromHost(unusablePaper), LETTER_PAPER, JSON.stringify(unusablePaper));
  }
});

test("should_accept_paper_at_the_edges_of_the_usable_range", () => {
  assert.deepEqual(paperSizeFromHost({ widthPoints: 360, heightPoints: 1440 }), { widthInches: 5, heightInches: 20 });
  assert.deepEqual(paperSizeFromHost({ widthPoints: 359, heightPoints: 1440 }), LETTER_PAPER);
  assert.deepEqual(paperSizeFromHost({ widthPoints: 360, heightPoints: 1441 }), LETTER_PAPER);
});

test("should_leave_one_inch_margins_around_the_text", () => {
  assert.equal(textWidthInches(LETTER_PAPER), 6.5);
  assert.equal(textHeightInches(LETTER_PAPER), 9);
  assert.ok(Math.abs(textWidthInches(A4_PAPER) - 6.27) < 0.001);
  assert.ok(Math.abs(textHeightInches(A4_PAPER) - 9.69) < 0.001);
});

test("should_carry_the_hosts_paper_in_the_page_configuration", () => {
  const hostConfiguration = { daemonBaseURL: "http://127.0.0.1:6733", expectedChannel: "development", expectedStateDirectory: "state" };

  assert.deepEqual(AppConfig.fromWindow(hostConfiguration).printedPaper, LETTER_PAPER);
  assert.deepEqual(AppConfig.fromWindow({ ...hostConfiguration, printedPaper: { widthPoints: 612, heightPoints: 792 } }).printedPaper, LETTER_PAPER);
  assert.ok(Math.abs(AppConfig.fromWindow({ ...hostConfiguration, printedPaper: { widthPoints: 595, heightPoints: 842 } }).printedPaper.heightInches - 11.69) < 0.01);
});
