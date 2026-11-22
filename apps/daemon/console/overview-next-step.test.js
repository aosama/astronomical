// Home's next step: the engine's first suggestion, joined to its catalog size.

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const nextStepScriptPath = path.join(__dirname, "overview-next-step.js");

function nextStepFor(catalogDocument) {
    const scriptContext = vm.createContext({});
    vm.runInContext(fs.readFileSync(nextStepScriptPath, "utf8"), scriptContext, { filename: nextStepScriptPath });
    scriptContext.catalogDocument = catalogDocument;
    return JSON.parse(JSON.stringify(vm.runInContext("overviewNextStepSuggestion(catalogDocument)", scriptContext)));
}

const ENTRIES = [{ huggingface_id: "example-org/Large", approximate_size_bytes: 23_514_891_878 }];

test("should_take_the_engines_first_suggestion_with_its_download_size", () => {
    const nextStep = nextStepFor({
        suggestions: [{ kind: "best_fit", huggingface_id: "example-org/Large", model_id: "Large", tag: "Best fit for this Mac", reason: "Strongest that fits." }],
        entries: ENTRIES
    });

    assert.equal(nextStep.model_id, "Large");
    assert.equal(nextStep.tag, "Best fit for this Mac");
    assert.equal(nextStep.approximate_size_bytes, 23_514_891_878);
});

test("should_skip_a_suggestion_without_words_and_answer_null_without_a_listed_model", () => {
    assert.equal(nextStepFor({ suggestions: [], entries: ENTRIES }), null);
    assert.equal(nextStepFor({ entries: ENTRIES }), null);
    assert.equal(nextStepFor(null), null);
    assert.equal(
        nextStepFor({ suggestions: [{ huggingface_id: "example-org/Missing", reason: "Not listed." }], entries: ENTRIES }),
        null
    );
    assert.equal(
        nextStepFor({
            suggestions: [{ huggingface_id: "example-org/Large" }, { huggingface_id: "example-org/Large", reason: "Has words." }],
            entries: ENTRIES
        }).reason,
        "Has words."
    );
});
