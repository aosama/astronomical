import assert from "node:assert/strict";
import { test } from "node:test";
import { pagedTextHeightPixels } from "../src/export/PageEstimate";

const PAGE_HEIGHT_PIXELS = 864;

// The chat that showed the miscount: text about 1.9 pages tall whose
// flowchart started near the foot of page 2 and printed on page 3.
test("should_count_the_gap_a_diagram_leaves_when_it_moves_to_the_next_page", () => {
  const pagedHeightPixels = pagedTextHeightPixels(1650, [{ topPixels: 1300, heightPixels: 520 }], PAGE_HEIGHT_PIXELS);

  assert.equal(pagedHeightPixels, 1650 + (1728 - 1300));
  assert.equal(Math.ceil(pagedHeightPixels / PAGE_HEIGHT_PIXELS), 3);
});

test("should_add_nothing_for_a_block_that_fits_on_its_page", () => {
  assert.equal(pagedTextHeightPixels(1650, [{ topPixels: 100, heightPixels: 400 }], PAGE_HEIGHT_PIXELS), 1650);
});

test("should_add_nothing_for_a_block_that_ends_exactly_at_the_page_end", () => {
  assert.equal(pagedTextHeightPixels(1650, [{ topPixels: 464, heightPixels: 400 }], PAGE_HEIGHT_PIXELS), 1650);
  assert.equal(pagedTextHeightPixels(1650, [{ topPixels: 465, heightPixels: 400 }], PAGE_HEIGHT_PIXELS), 1650 + 399);
});

test("should_not_move_a_block_taller_than_a_page", () => {
  assert.equal(pagedTextHeightPixels(2000, [{ topPixels: 300, heightPixels: 900 }], PAGE_HEIGHT_PIXELS), 2000);
});

test("should_not_move_a_block_already_at_the_top_of_a_page", () => {
  assert.equal(pagedTextHeightPixels(2000, [{ topPixels: 864, heightPixels: 864 }], PAGE_HEIGHT_PIXELS), 2000);
});

test("should_carry_one_blocks_gap_into_where_the_next_block_lands", () => {
  // The first block moves down 64, which pushes the second across a page end.
  const wholeBlocks = [{ topPixels: 1500, heightPixels: 200 }, { topPixels: 800, heightPixels: 100 }];

  assert.equal(pagedTextHeightPixels(1800, wholeBlocks, PAGE_HEIGHT_PIXELS), 1800 + 64 + (1728 - 1564));
});

test("should_return_the_flowing_height_when_there_is_no_page_to_cut_to", () => {
  assert.equal(pagedTextHeightPixels(500, [{ topPixels: 0, heightPixels: 100 }], 0), 500);
  assert.equal(pagedTextHeightPixels(500, [], PAGE_HEIGHT_PIXELS), 500);
});
