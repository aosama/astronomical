import assert from "node:assert/strict";
import { test } from "node:test";
import { CurrentTimeTool } from "../src/CurrentTimeTool";
import { ToolCatalog } from "../src/ToolCatalog";

test("should_offer_tools_only_to_a_tool_capable_model_with_the_switch_on", () => {
  const catalog = new ToolCatalog([new CurrentTimeTool()]);
  assert.equal(catalog.toolsForTurn({ modelSupportsToolCalls: true, toolsEnabled: true }).length, 1);
  assert.equal(catalog.toolsForTurn({ modelSupportsToolCalls: false, toolsEnabled: true }).length, 0);
  assert.equal(catalog.toolsForTurn({ modelSupportsToolCalls: true, toolsEnabled: false }).length, 0);
});

test("should_offer_nothing_from_an_empty_catalog", () => {
  assert.equal(new ToolCatalog([]).toolsForTurn({ modelSupportsToolCalls: true, toolsEnabled: true }).length, 0);
});

test("should_refuse_two_tools_with_the_same_name", () => {
  assert.throws(() => new ToolCatalog([new CurrentTimeTool(), new CurrentTimeTool()]), /share the name get_current_time/);
});

test("should_label_known_tools_and_fall_back_to_the_name_for_unknown_ones", () => {
  const catalog = new ToolCatalog([new CurrentTimeTool()]);
  assert.equal(catalog.labelFor("get_current_time"), "Current time");
  assert.equal(catalog.labelFor("retired_tool"), "retired_tool");
});
