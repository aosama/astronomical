// These journey-level contracts keep pull-request enforcement actionable without
// depending on mutable GitHub state.

const assert = require("node:assert/strict");
const test = require("node:test");

const {
    isAutomatedPullRequest,
    validatePullRequestIssue,
} = require("./pull-request-issue-compliance.js");

function createIssue(overrides = {}) {
    return {
        number: 224,
        state: "open",
        html_url: "https://github.com/example/astronomical/issues/224",
        ...overrides,
    };
}

test("should treat bot-authored pull requests as exempt from issue linkage", () => {
    for (const authorLogin of ["dependabot[bot]", "renovate[bot]", "app/dependabot[bot]"]) {
        assert.equal(isAutomatedPullRequest(authorLogin), true, authorLogin);
    }
    for (const authorLogin of ["aosama", "maintenance-contributor", "", undefined, null]) {
        assert.equal(isAutomatedPullRequest(authorLogin), false, String(authorLogin));
    }
});

test("should validate linked issue provenance for the complete pull request journey", async () => {
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

test("should accept an existing issue independently of its state and body format", async () => {
    const validIssueVariants = [
        createIssue({ body: "" }),
        createIssue({ body: "A free-form issue description without prescribed headings." }),
        createIssue({ state: "closed", body: null }),
    ];

    for (const linkedIssue of validIssueVariants) {
        const compliance = await validatePullRequestIssue({
            pullRequestBody: "## Linked issue\n\nRefs #224",
            loadIssue: async () => linkedIssue,
        });

        assert.equal(compliance.issueNumber, 224);
    }
});

test("should accept a bare keyword reference when the heading is forgotten", async () => {
    const compliance = await validatePullRequestIssue({
        pullRequestBody: "Fixes #224\n\n## Change\n\nAdd enforcement.",
        loadIssue: async () => createIssue(),
    });

    assert.equal(compliance.issueNumber, 224);
    assert.equal(compliance.relationship, "Fixes");
});

test("should accept the Linked issue heading without the ## marker", async () => {
    for (const heading of ["Linked issue", "linked issue:", "# Linked issue", "### Linked issue"]) {
        const compliance = await validatePullRequestIssue({
            pullRequestBody: `${heading}\n\nRefs #224`,
            loadIssue: async () => createIssue(),
        });

        assert.equal(compliance.issueNumber, 224);
    }
});

test("should accept trailing prose after the reference in the Linked issue section", async () => {
    const compliance = await validatePullRequestIssue({
        pullRequestBody: "## Linked issue\n\nFixes #224 — the gate instrumentation work.",
        loadIssue: async () => createIssue(),
    });

    assert.equal(compliance.issueNumber, 224);
});

test("should reject a keyword-less issue mention when no heading exists", async () => {
    await assert.rejects(
        validatePullRequestIssue({
            pullRequestBody: "Addresses #224 without any heading or closing keyword.",
            loadIssue: async () => createIssue(),
        }),
        /Add a `## Linked issue` section/,
    );
});

test("should accept several canonical references without a heading, first as primary", async () => {
    const compliance = await validatePullRequestIssue({
        pullRequestBody: "Fixes #224 and Closes #225 in one pass.",
        loadIssue: async (issueNumber) => createIssue({ number: issueNumber }),
    });

    assert.equal(compliance.issueNumber, 224);
    assert.equal(compliance.relationship, "Fixes");
});

test("should reject a heading-less body mixing a keyword reference with a bare mention", async () => {
    await assert.rejects(
        validatePullRequestIssue({
            pullRequestBody: "Fixes #224 and #225 in one pass.",
            loadIssue: async () => createIssue(),
        }),
        /exactly one same-repository issue reference/,
    );
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

test("should accept several canonical references with the first as the primary linked issue", async () => {
    const loadedIssueNumbers = [];

    const compliance = await validatePullRequestIssue({
        pullRequestBody: "## Linked issue\n\nRefs #983\nRefs #1081\n\n## Change\n\nThe slice work.",
        loadIssue: async (issueNumber) => {
            loadedIssueNumbers.push(issueNumber);
            return createIssue({ number: issueNumber, html_url: `https://github.com/example/astronomical/issues/${issueNumber}` });
        },
    });

    assert.deepEqual(loadedIssueNumbers, [983, 1081]);
    assert.deepEqual(compliance, {
        issueNumber: 983,
        relationship: "Refs",
        issueUrl: "https://github.com/example/astronomical/issues/983",
    });
});

test("should reject a bare issue mention inside the Linked issue section", async () => {
    await assert.rejects(
        validatePullRequestIssue({
            pullRequestBody: "## Linked issue\n\nFixes #224 and #225 in one pass.",
            loadIssue: async () => createIssue(),
        }),
        /must reference at least one issue/,
    );
});

test("should reject a nonexistent secondary reference with its own number", async () => {
    const notFoundError = new Error("Not Found");
    notFoundError.status = 404;

    await assert.rejects(
        validatePullRequestIssue({
            pullRequestBody: "## Linked issue\n\nRefs #224\nRefs #999",
            loadIssue: async (issueNumber) => {
                if (issueNumber === 999) {
                    throw notFoundError;
                }
                return createIssue();
            },
        }),
        /Issue #999 does not exist in this repository/,
    );
});

test("should reject a pull request presented as a secondary reference", async () => {
    await assert.rejects(
        validatePullRequestIssue({
            pullRequestBody: "## Linked issue\n\nRefs #224\nCloses #225",
            loadIssue: async (issueNumber) => {
                if (issueNumber === 225) {
                    return createIssue({ number: 225, pull_request: { url: "https://api.github.com/pulls/225" } });
                }
                return createIssue();
            },
        }),
        /#225 identifies a pull request, not an issue/,
    );
});

test("should reject malformed and cross-repository references", async () => {
    for (const reference of ["#224", "Fixes example/other#224", "Fixes #0", "Fixes #224 and #225"]) {
        await assert.rejects(
            validatePullRequestIssue({
                pullRequestBody: `## Linked issue\n\n${reference}`,
                loadIssue: async () => createIssue(),
            }),
            /must reference at least one issue/,
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
