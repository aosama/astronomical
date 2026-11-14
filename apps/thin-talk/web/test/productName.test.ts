import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { SearchConversationsTool } from "../src/SearchConversationsTool";
import { scriptedBridge } from "./support/scriptedBridge";

/**
 * "Thin Talk" is this module's name in code. The person is only ever told
 * about Astronomical and Astro, so a window, a field or a model reply that
 * says "Thin Talk" reads as some other program (issue 2204).
 */
const MODULE_NAME = "Thin Talk";

function pageMarkupWithoutComments(): string {
  const indexPath = process.env["THINTALK_INDEX_HTML"];
  assert.ok(indexPath, "THINTALK_INDEX_HTML must point at the real index.html");
  return readFileSync(indexPath, "utf8").replace(/<!--[\s\S]*?-->/g, "");
}

test("should_name_the_product_in_the_page_title_and_the_message_field", () => {
  const pageMarkup = pageMarkupWithoutComments();
  assert.ok(pageMarkup.includes("<title>Astronomical</title>"));
  assert.ok(pageMarkup.includes('placeholder="Message Astronomical"'));
});

test("should_never_show_the_module_name_in_the_page_markup", () => {
  assert.equal(pageMarkupWithoutComments().includes(MODULE_NAME), false);
});

test("should_keep_the_module_name_out_of_what_the_model_is_told_about_saved_conversations", () => {
  const searchTool = new SearchConversationsTool(scriptedBridge({}));
  assert.equal(searchTool.description.includes(MODULE_NAME), false, "a model repeats the names it is given");
  assert.ok(searchTool.description.includes("Astronomical"));
});
