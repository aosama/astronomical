const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const optimizerScriptPath = path.join(__dirname, "optimizer.js");
const optimizerScript = fs.readFileSync(optimizerScriptPath, "utf8");

// Loads only the optimizer domain so these tests remain focused on the
// context-range and decision transformations used by the user journey.
function createOptimizerContext() {
    const optimizerContext = vm.createContext({});
    vm.runInContext(optimizerScript, optimizerContext, { filename: optimizerScriptPath });
    return optimizerContext;
}

test("locates one optimizer evidence range within the full model context window", () => {
    const optimizerContext = createOptimizerContext();
    optimizerContext.latestChunkOutcome = {
        measurement_context: {
            chunk_start_token_position: 49152,
            position_range_start_token_position: 32768,
            position_range_end_token_position_exclusive: 65536
        },
        candidate_measurement_summaries: [
            {
                candidate_chunk_size_tokens: 1024,
                measurement_source: "execution_profile",
                measurement_count: 2
            },
            {
                candidate_chunk_size_tokens: 2048,
                measurement_source: "execution_profile",
                measurement_count: 1
            },
            {
                candidate_chunk_size_tokens: 4096,
                measurement_source: "no_measurements_available",
                measurement_count: 0
            }
        ]
    };

    const contextScope = vm.runInContext(
        "optimizerContextScope(latestChunkOutcome, 131072)",
        optimizerContext
    );

    assert.deepEqual(JSON.parse(JSON.stringify(contextScope)), {
        rangeStartTokenPosition: 32768,
        rangeEndTokenPositionExclusive: 65536,
        chunkStartTokenPosition: 49152,
        contextWindowTokenCount: 131072,
        rangeStartPercentage: 25,
        rangeWidthPercentage: 25,
        chunkPositionPercentage: 37.5,
        candidateCount: 3,
        measuredProfileCandidateCount: 2,
        unmeasuredCandidateCount: 1
    });
});

test("never lets an observed range exceed the available context track", () => {
    const optimizerContext = createOptimizerContext();
    optimizerContext.latestChunkOutcome = {
        measurement_context: {
            chunk_start_token_position: 70000,
            position_range_start_token_position: 65536,
            position_range_end_token_position_exclusive: 98304
        },
        candidate_measurement_summaries: []
    };

    const contextScope = vm.runInContext(
        "optimizerContextScope(latestChunkOutcome, 65536)",
        optimizerContext
    );

    assert.equal(contextScope.contextWindowTokenCount, 98304);
    assert.ok(Math.abs(contextScope.rangeStartPercentage - 66.6667) < 0.001);
    assert.ok(Math.abs(contextScope.rangeWidthPercentage - 33.3333) < 0.001);
    assert.ok(Math.abs(contextScope.chunkPositionPercentage - 71.2077) < 0.001);
});
