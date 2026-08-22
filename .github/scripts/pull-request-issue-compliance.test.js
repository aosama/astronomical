// These journey-level contracts keep pull-request enforcement actionable without
// depending on mutable GitHub state.

const assert = require("node:assert/strict");
const test = require("node:test");

const {
    validatePullRequestIssue,
} = require("./pull-request-issue-compliance.js");

const COMPLETE_ISSUE_BODY = `
### Goal

Make the contribution policy enforceable.

### Evidence

The repository does not currently verify linked issues.

### Scope

Validate pull request metadata.

### Constraints

Use read-only issue access.

### Acceptance criteria

Every pull request identifies an explanatory issue.
`;

function createIssue(overrides = {}) {
    return {
        number: 224,
        state: "open",
        body: COMPLETE_ISSUE_BODY,
        html_url: "https://github.com/example/astronomical/issues/224",
        ...overrides,
    };
}

test("should validate the complete contributor journey", async () => {
    const loadedIssueNumbers = [];

    const compliance = await validatePullRequestIssue({
        pullRequestBody: "## Linked issue\n\nFixes #224\n\n## Change\n\nAdd enforcement.",
        loadIssue: async (issueNumber) => {
            loadedIssueNumbers.push(issueNumber);
            return createIssue();
        },
    });

    assert.deepEqual(loadedIssueNumbers, [224]);
    assert.deepEqual(compliance, {
        issueNumber: 224,
        relationship: "Fixes",
        issueUrl: "https://github.com/example/astronomical/issues/224",
    });
});

test("should accept each documented relationship keyword without case sensitivity", async () => {
    for (const relationship of ["fixes", "CLOSES", "Resolves", "Refs"]) {
        const compliance = await validatePullRequestIssue({
            pullRequestBody: `## Linked issue\n${relationship} #224`,
            loadIssue: async () => createIssue(),
        });

        assert.equal(compliance.issueNumber, 224);
    }
});

test("should reject a missing Linked issue section", async () => {
    await assert.rejects(
        validatePullRequestIssue({
            pullRequestBody: "## Change\n\nAdd enforcement.",
            loadIssue: async () => createIssue(),
        }),
        /Add a `## Linked issue` section/,
    );
});

test("should reject malformed and cross-repository references", async () => {
    for (const reference of ["#224", "Fixes example/other#224", "Fixes #0", "Fixes #224 and #225"]) {
        await assert.rejects(
            validatePullRequestIssue({
                pullRequestBody: `## Linked issue\n\n${reference}`,
                loadIssue: async () => createIssue(),
            }),
            /exactly one same-repository reference/,
        );
    }
});

test("should reject a pull request presented as an issue", async () => {
    await assert.rejects(
        validatePullRequestIssue({
            pullRequestBody: "## Linked issue\n\nRefs #224",
            loadIssue: async () => createIssue({ pull_request: { url: "https://api.github.com/pulls/224" } }),
        }),
        /identifies a pull request, not an issue/,
    );
});

test("should reject a closed issue", async () => {
    await assert.rejects(
        validatePullRequestIssue({
            pullRequestBody: "## Linked issue\n\nRefs #224",
            loadIssue: async () => createIssue({ state: "closed" }),
        }),
        /must remain open until this pull request merges/,
    );
});

test("should reject a nonexistent issue with actionable guidance", async () => {
    const notFoundError = new Error("Not Found");
    notFoundError.status = 404;

    await assert.rejects(
        validatePullRequestIssue({
            pullRequestBody: "## Linked issue\n\nRefs #999",
            loadIssue: async () => {
                throw notFoundError;
            },
        }),
        /Issue #999 does not exist in this repository/,
    );
});

test("should preserve unexpected GitHub failures", async () => {
    const serviceError = new Error("GitHub service unavailable");
    serviceError.status = 503;

    await assert.rejects(
        validatePullRequestIssue({
            pullRequestBody: "## Linked issue\n\nRefs #224",
            loadIssue: async () => {
                throw serviceError;
            },
        }),
        serviceError,
    );
});

test("should reject an issue missing explanatory content", async () => {
    const incompleteIssueBody = COMPLETE_ISSUE_BODY.replace(
        "Every pull request identifies an explanatory issue.",
        "<!-- Add acceptance criteria here. -->",
    );

    await assert.rejects(
        validatePullRequestIssue({
            pullRequestBody: "## Linked issue\n\nRefs #224",
            loadIssue: async () => createIssue({ body: incompleteIssueBody }),
        }),
        /complete these issue sections: Acceptance criteria/,
    );
});
