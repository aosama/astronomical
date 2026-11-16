// Models page selection guard tests: a rebuild waits while text on the page is
// selected or being selected, and not otherwise.

const assert = require("node:assert/strict");
const test = require("node:test");
const vm = require("node:vm");

const { createDownloadModelsContext } = require("./download-models-test-support.js");

// A page container holding `insideNode`, and a selection that is either
// collapsed or spans `selectedNode`.
function contextWithSelection(selectedNode, isCollapsed) {
    const scriptContext = createDownloadModelsContext();
    const insideNode = { name: "inside" };
    scriptContext.catalogContainer = { contains: (candidateNode) => candidateNode === insideNode };
    scriptContext.window = {
        getSelection: () => ({
            isCollapsed,
            rangeCount: 1,
            getRangeAt: () => ({ startContainer: selectedNode(insideNode), endContainer: selectedNode(insideNode) })
        })
    };
    return scriptContext;
}

test("should_hold_the_rebuild_while_text_inside_the_page_is_selected", () => {
    const scriptContext = contextWithSelection((insideNode) => insideNode, false);

    assert.equal(vm.runInContext("isDownloadModelsSelectionHeld(catalogContainer)", scriptContext), true);
});

test("should_not_hold_the_rebuild_for_an_empty_selection", () => {
    const scriptContext = contextWithSelection((insideNode) => insideNode, true);

    assert.equal(vm.runInContext("isDownloadModelsSelectionHeld(catalogContainer)", scriptContext), false);
});

test("should_not_hold_the_rebuild_for_a_selection_elsewhere_on_the_screen", () => {
    const scriptContext = contextWithSelection(() => ({ name: "header" }), false);

    assert.equal(vm.runInContext("isDownloadModelsSelectionHeld(catalogContainer)", scriptContext), false);
});

test("should_hold_the_rebuild_while_the_mouse_button_is_down_for_a_drag", () => {
    const scriptContext = contextWithSelection((insideNode) => insideNode, true);

    vm.runInContext("isDownloadModelsPointerSelecting = true", scriptContext);

    assert.equal(vm.runInContext("isDownloadModelsSelectionHeld(catalogContainer)", scriptContext), true);
});

test("should_not_hold_the_rebuild_where_the_page_offers_no_selection_at_all", () => {
    const scriptContext = createDownloadModelsContext();
    scriptContext.catalogContainer = { contains: () => true };
    scriptContext.window = {};

    assert.equal(vm.runInContext("isDownloadModelsSelectionHeld(catalogContainer)", scriptContext), false);
});
