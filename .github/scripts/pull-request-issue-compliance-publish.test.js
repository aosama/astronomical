const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const repositoryRoot = path.resolve(__dirname, "../..");
const publisherWorkflow = fs.readFileSync(
    path.join(repositoryRoot, ".github/workflows/pull-request-issue-compliance-publish.yml"),
    "utf8",
);
const scriptMarker = "          script: |\n";
assert.ok(publisherWorkflow.includes(scriptMarker));
const publisherScript = publisherWorkflow.split(scriptMarker)[1]
    .split("\n").map((line) => line.slice(12)).join("\n");
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const executePublisher = new AsyncFunction("require", "process", "context", "github", "core", publisherScript);

async function publishForBody(body, author = "contributor", eventName = "workflow_dispatch") {
    const statuses = [];
    const failures = [];
    const context = {
        eventName,
        runId: 123,
        repo: { owner: "example", repo: "astronomical" },
        payload: {
            inputs: { pull_request_number: "42" },
            workflow_run: { head_sha: "collector-head" },
        },
    };
    const github = {
        rest: {
            pulls: {
                get: async () => ({
                    data: { head: { sha: "current-head" }, body, user: { login: author } },
                }),
            },
            issues: {
                get: async () => ({
                    data: {
                        number: 224,
                        state: "open",
                        html_url: "https://github.com/example/astronomical/issues/224",
                    },
                }),
            },
            repos: {
                createCommitStatus: async (status) => { statuses.push(status); },
                listPullRequestsAssociatedWithCommit: async () => ({
                    data: [{
                        number: 42,
                        state: "open",
                        base: { repo: { full_name: "example/astronomical" } },
                    }],
                }),
            },
        },
    };
    await executePublisher(require, {
        env: { GITHUB_WORKSPACE: repositoryRoot, GITHUB_SERVER_URL: "https://github.com" },
    }, context, github, {
        info: () => {},
        setFailed: (message) => { failures.push(message); },
    });
    return { statuses, failures };
}

test("publishes current-head commit statuses after a body failure is corrected", { timeout: 60_000 }, async () => {
    const rejected = await publishForBody("## Linked issue\n\nNone");
    assert.equal(rejected.statuses[0].state, "failure");
    assert.equal(rejected.failures.length, 1);

    const corrected = await publishForBody("## Linked issue\n\nCloses #224", "contributor", "workflow_run");
    assert.deepEqual(corrected.failures, []);
    assert.deepEqual(corrected.statuses, [{
        owner: "example",
        repo: "astronomical",
        sha: "current-head",
        context: "PR issue compliance",
        state: "success",
        description: "Pull request links issue #224",
        target_url: "https://github.com/example/astronomical/actions/runs/123",
    }]);
});

test("preserves the automated-author exemption through commit statuses", { timeout: 60_000 }, async () => {
    const automated = await publishForBody("", "dependabot[bot]");
    assert.equal(automated.statuses.length, 1);
    assert.equal(automated.statuses[0].state, "success");
    assert.deepEqual(automated.failures, []);
});

test("publisher grants status publication without synthetic check-write permission", { timeout: 60_000 }, () => {
    assert.match(publisherWorkflow, /statuses: write/);
    assert.doesNotMatch(publisherWorkflow, /checks: write|github\.rest\.checks/);
    assert.match(publisherWorkflow, /ref: \$\{\{ github\.event\.repository\.default_branch \}\}/);
});
