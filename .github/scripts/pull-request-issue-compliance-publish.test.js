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

async function publishForBody(
    body,
    author = "contributor",
    eventName = "workflow_dispatch",
    issueOverrides = {},
    issuesGetError = null,
) {
    const statuses = [];
    const failures = [];
    const warnings = [];
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
                get: async () => {
                    if (issuesGetError !== null) {
                        throw issuesGetError;
                    }
                    return {
                        data: {
                            number: 224,
                            state: "open",
                            html_url: "https://github.com/example/astronomical/issues/224",
                            labels: [],
                            ...issueOverrides,
                        },
                    };
                },
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
        warning: (message) => { warnings.push(message); },
        error: () => {},
        setFailed: (message) => { failures.push(message); },
    });
    return { statuses, failures, warnings };
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

test("accepts an umbrella context reference without a closing keyword on it", { timeout: 60_000 }, async () => {
    const accepted = await publishForBody("## Linked issue\n\nResolves #1426 (part of the #1428 umbrella)");
    assert.equal(accepted.statuses[0].state, "success");
    assert.equal(accepted.statuses[0].description, "Pull request links issue #1426");
    assert.deepEqual(accepted.failures, []);
    assert.deepEqual(accepted.warnings, []);
});

test("surfaces stale closing-keyword warnings through core.warning without failing", { timeout: 60_000 }, async () => {
    const warned = await publishForBody(
        "## Linked issue\n\nFixes #224",
        "contributor",
        "workflow_dispatch",
        { state: "closed" },
    );
    assert.equal(warned.statuses[0].state, "success");
    assert.deepEqual(warned.failures, []);
    assert.equal(warned.warnings.length, 1);
    assert.match(warned.warnings[0], /already closed/);
});

test("defers instead of publishing failure when GitHub answers 5xx mid-validation", { timeout: 60_000 }, async () => {
    const serviceUnavailableError = new Error("GitHub service unavailable");
    serviceUnavailableError.status = 500;
    const deferred = await publishForBody(
        "## Linked issue\n\nFixes #224",
        "contributor",
        "workflow_dispatch",
        {},
        serviceUnavailableError,
    );
    assert.deepEqual(deferred.statuses, []);
    assert.equal(deferred.failures.length, 1);
    assert.match(deferred.failures[0], /Transient GitHub API error \(HTTP 500\)/);
});

test("defers on a 403 secondary rate limit the step retries never attempt", { timeout: 60_000 }, async () => {
    const rateLimitError = new Error("Secondary rate limit reached");
    rateLimitError.status = 403;
    const deferred = await publishForBody(
        "## Linked issue\n\nFixes #224",
        "contributor",
        "workflow_dispatch",
        {},
        rateLimitError,
    );
    assert.deepEqual(deferred.statuses, []);
    assert.equal(deferred.failures.length, 1);
    assert.match(deferred.failures[0], /Transient GitHub API error \(HTTP 403\)/);
});

test("publisher grants status publication without synthetic check-write permission", { timeout: 60_000 }, () => {
    assert.match(publisherWorkflow, /statuses: write/);
    assert.doesNotMatch(publisherWorkflow, /checks: write|github\.rest\.checks/);
    assert.match(publisherWorkflow, /ref: \$\{\{ github\.event\.repository\.default_branch \}\}/);
    assert.match(publisherWorkflow, /retries: 3/);
});

// A collector run for commit "lower-head" belongs to the lower pull request of a
// stack, while the association endpoint also lists the stacked pull request.
async function publishForStackedHeadCommit(associatedPullRequests) {
    const requestedPullRequestNumbers = [];
    const context = {
        eventName: "workflow_run",
        runId: 123,
        repo: { owner: "example", repo: "astronomical" },
        payload: { workflow_run: { head_sha: "lower-head" } },
    };
    const github = {
        rest: {
            pulls: {
                get: async ({ pull_number }) => {
                    requestedPullRequestNumbers.push(pull_number);
                    return {
                        data: { head: { sha: "lower-head" }, body: "## Linked issue\n\nRefs #224\n", user: { login: "contributor" } },
                    };
                },
            },
            issues: {
                get: async () => ({
                    data: {
                        number: 224,
                        state: "open",
                        html_url: "https://github.com/example/astronomical/issues/224",
                        labels: [],
                    },
                }),
            },
            repos: {
                createCommitStatus: async () => {},
                listPullRequestsAssociatedWithCommit: async () => ({ data: associatedPullRequests }),
            },
        },
    };
    await executePublisher(require, {
        env: { GITHUB_WORKSPACE: repositoryRoot, GITHUB_SERVER_URL: "https://github.com" },
    }, context, github, { info: () => {}, warning: () => {}, error: () => {}, setFailed: () => {} });
    return requestedPullRequestNumbers;
}

function openPullRequest(number, headSha) {
    return {
        number,
        state: "open",
        head: { sha: headSha },
        base: { repo: { full_name: "example/astronomical" } },
    };
}

test("publishes for the pull request whose head is the checked commit, not one stacked on it", { timeout: 60_000 }, async () => {
    const requested = await publishForStackedHeadCommit([
        openPullRequest(43, "stacked-head"),
        openPullRequest(42, "lower-head"),
    ]);
    assert.deepEqual(requested, [42]);
});

test("falls back to the first open pull request when none has the checked commit as its head", { timeout: 60_000 }, async () => {
    const requested = await publishForStackedHeadCommit([
        { ...openPullRequest(41, "older-head"), state: "closed" },
        openPullRequest(43, "stacked-head"),
    ]);
    assert.deepEqual(requested, [43]);
});

test("skips without reading a pull request when none is open in this repository", { timeout: 60_000 }, async () => {
    const requested = await publishForStackedHeadCommit([
        { ...openPullRequest(41, "lower-head"), state: "closed" },
    ]);
    assert.deepEqual(requested, []);
});

test("never picks a fork's pull request even when its head is the checked commit", { timeout: 60_000 }, async () => {
    const forkPullRequest = {
        ...openPullRequest(44, "lower-head"),
        base: { repo: { full_name: "someone-else/astronomical" } },
    };
    const requested = await publishForStackedHeadCommit([
        forkPullRequest,
        openPullRequest(42, "other-head"),
    ]);
    assert.deepEqual(requested, [42]);
});
