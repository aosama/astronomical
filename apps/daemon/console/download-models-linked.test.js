// Models page tests for a model opened from a link: which links are honoured,
// and what the page puts at its top for a model in the catalog, a model
// already on this Mac, and a model the catalog does not have.

const {
    assert,
    createDownloadModelsContext,
    createDownloadModelsDocument,
    findRenderedElementsByClass,
    test,
    validCatalogEntry,
    vm
} = require("./download-models-test-support.js");

function linkedModelIdFrom(locationSearch) {
    const scriptContext = createDownloadModelsContext();
    scriptContext.locationSearch = locationSearch;
    return vm.runInContext("readDownloadModelsLinkedModelId(locationSearch)", scriptContext);
}

function renderModelsPageOpenedOn(linkedModelId, catalogEntries) {
    const scriptContext = createDownloadModelsContext();
    const downloadModelsDocument = createDownloadModelsDocument();
    scriptContext.document = downloadModelsDocument;
    scriptContext.catalogDocument = { schema_version: 5, entries: catalogEntries };
    scriptContext.linkedModelId = linkedModelId;
    vm.runInContext(
        `downloadModelsCatalogDocument = catalogDocument;
         downloadModelsLinkedModelId = linkedModelId;
         downloadModelsServingCeilingBytes = 32 * 1073741824;
         renderDownloadModelsCatalogDocument(catalogDocument)`,
        scriptContext
    );
    return downloadModelsDocument.catalogContainer;
}

function textsOfClass(rootElement, className) {
    return findRenderedElementsByClass(rootElement, className).map((element) => element.textContent);
}

test("should_read_an_owner_and_name_from_the_page_address", () => {
    assert.equal(linkedModelIdFrom("?model=example-org/Example.Model_4bit"), "example-org/Example.Model_4bit");
    assert.equal(linkedModelIdFrom("?other=1&model=example-org%2Fexample"), "example-org/example");
});

test("should_ignore_an_address_without_a_usable_model_name", () => {
    assert.equal(linkedModelIdFrom(""), null);
    assert.equal(linkedModelIdFrom("?other=1"), null);
    assert.equal(linkedModelIdFrom("?model="), null);
    assert.equal(linkedModelIdFrom("?model=no-owner"), null);
    assert.equal(linkedModelIdFrom("?model=a/b/c"), null);
    assert.equal(linkedModelIdFrom("?model=example-org/<script>"), null);
    assert.equal(linkedModelIdFrom("?model=example-org/" + "a".repeat(97)), null);
    assert.equal(linkedModelIdFrom(undefined), null);
});

test("should_put_a_linked_catalog_model_at_the_top_with_its_own_get_button", () => {
    const catalogContainer = renderModelsPageOpenedOn("Example-Org/Linked-9B-4bit", [
        validCatalogEntry({ huggingface_id: "example-org/other-2b" }),
        validCatalogEntry({
            huggingface_id: "example-org/linked-9b-4bit",
            summary: "A quick everyday assistant."
        })
    ]);

    assert.deepEqual(textsOfClass(catalogContainer, "download-models-linked-title"), ["You opened linked-9b-4bit"]);
    assert.deepEqual(
        textsOfClass(catalogContainer, "download-models-linked-lede"),
        ["A quick everyday assistant. It downloads once, then runs entirely on this Mac."]
    );
    const linkedSection = findRenderedElementsByClass(catalogContainer, "download-models-linked")[0];
    const getButtons = findRenderedElementsByClass(linkedSection, "download-models-download-button");
    assert.equal(getButtons.length, 1, "the linked model offers one Get button");
    assert.equal(catalogContainer.children[1], linkedSection, "the linked model comes before every other section");
});

test("should_say_a_linked_model_is_already_on_this_mac_and_offer_no_download", () => {
    const catalogContainer = renderModelsPageOpenedOn("example-org/linked-9b-4bit", [
        validCatalogEntry({
            huggingface_id: "example-org/linked-9b-4bit",
            summary: "A quick everyday assistant.",
            ready_on_this_mac: true
        })
    ]);

    assert.deepEqual(
        textsOfClass(catalogContainer, "download-models-linked-lede"),
        ["A quick everyday assistant. It is already on this Mac."]
    );
    const linkedSection = findRenderedElementsByClass(catalogContainer, "download-models-linked")[0];
    assert.equal(findRenderedElementsByClass(linkedSection, "download-models-download-button").length, 0);
});

test("should_say_plainly_when_a_linked_model_is_not_in_the_catalog", () => {
    const catalogContainer = renderModelsPageOpenedOn("someone/unlisted-model", [
        validCatalogEntry({ huggingface_id: "example-org/other-2b" })
    ]);

    assert.deepEqual(
        textsOfClass(catalogContainer, "download-models-linked-title"),
        ["That model is not in Astronomical's list"]
    );
    assert.deepEqual(
        textsOfClass(catalogContainer, "download-models-linked-lede"),
        ["someone/unlisted-model is not one of the models Astronomical has checked. The models below are."]
    );
    const linkedSection = findRenderedElementsByClass(catalogContainer, "download-models-linked")[0];
    assert.equal(findRenderedElementsByClass(linkedSection, "download-models-download-button").length, 0);
});

test("should_show_no_linked_section_when_the_page_was_opened_without_a_link", () => {
    const catalogContainer = renderModelsPageOpenedOn(null, [
        validCatalogEntry({ huggingface_id: "example-org/other-2b" })
    ]);

    assert.equal(findRenderedElementsByClass(catalogContainer, "download-models-linked").length, 0);
});
