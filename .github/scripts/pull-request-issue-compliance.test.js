// These journey-level contracts keep pull-request enforcement actionable without
// depending on mutable GitHub state.

const assert = require("node:assert/strict");
const test = require("node:test");

const {
    isAutomatedPullRequest,
    validatePullRequestIssue,
} = require("./pull-request-issue-compliance.js");

const REPOSITORY_FULL_NAME = "example/astronomical";

function createIssue(overrides = {}) {
    return {
        number: 224,
        state: "open",
        html_url: "https://github.com/example/astronomical/issues/224",
        ...overrides,
    };
}

function issueUrlFor(issueNumber) {
    return `https://github.com/${REPOSITORY_FULL_NAME}/issues/${issueNumber}`;
}

async function validateBody(pullRequestBody, overrides = {}) {
    const defaultLoadIssue = async (issueNumber) => createIssue({
        number: issueNumber,
        html_url: issueUrlFor(issueNumber),
    });
    const loadIssue = overrides.loadIssue != null ? overrides.loadIssue : defaultLoadIssue;
    return validatePullRequestIssue({
        pullRequestBody,
        repositoryFullName: REPOSITORY_FULL_NAME,
        loadIssue,
    });
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
        repositoryFullName: REPOSITORY_FULL_NAME,
    });

    assert.deepEqual(loadedIssueNumbers, [224]);
    assert.deepEqual(compliance, {
        issueNumber: 224,
        relationship: "Fixes",
        issueUrl: issueUrlFor(224),
        linkedIssues: [
            { issueNumber: 224, relationship: "Fixes", kind: "closing", issueUrl: issueUrlFor(224) },
        ],
        warnings: [],
    });
});

test("should accept each documented relationship keyword without case sensitivity", async () => {
    for (const relationship of ["fixes", "CLOSES", "Resolves", "Refs"]) {
        const compliance = await validateBody(`## Linked issue\n${relationship} #224`);
        assert.equal(compliance.issueNumber, 224);
    }
});

test("should accept the full GitHub closing-keyword set", async () => {
    for (const keyword of ["close", "Closed", "FIX", "fixed", "resolve", "RESOLVED", "Resolve"]) {
        const compliance = await validateBody(`## Linked issue\n${keyword} #224`);
        assert.equal(compliance.issueNumber, 224, keyword);
    }
});

test("should accept a colon between the keyword and the issue like GitHub does", async () => {
    const compliance = await validateBody("## Linked issue\n\nResolves: #224");
    assert.equal(compliance.issueNumber, 224);
    assert.equal(compliance.relationship, "Resolves");
});

test("should accept an existing issue independently of its state and body format", async () => {
    const validIssueVariants = [
        createIssue({ body: "" }),
        createIssue({ body: "A free-form issue description without prescribed headings." }),
        createIssue({ state: "closed", body: null }),
    ];

    for (const linkedIssue of validIssueVariants) {
        const compliance = await validateBody("## Linked issue\n\nRefs #224", {
            loadIssue: async () => linkedIssue,
        });

        assert.equal(compliance.issueNumber, 224);
    }
});

test("should accept a bare keyword reference when the heading is forgotten", async () => {
    const compliance = await validateBody("Fixes #224\n\n## Change\n\nAdd enforcement.");

    assert.equal(compliance.issueNumber, 224);
    assert.equal(compliance.relationship, "Fixes");
});

test("should accept the Linked issue heading without the ## marker", async () => {
    for (const heading of ["Linked issue", "linked issue:", "# Linked issue", "### Linked issue"]) {
        const compliance = await validateBody(`${heading}\n\nRefs #224`);
        assert.equal(compliance.issueNumber, 224);
    }
});

test("should accept trailing prose after the reference in the Linked issue section", async () => {
    const compliance = await validateBody("## Linked issue\n\nFixes #224 — the gate instrumentation work.");
    assert.equal(compliance.issueNumber, 224);
});

test("should reject a keyword-less issue mention when no heading exists", async () => {
    await assert.rejects(
        validateBody("Addresses #224 without any heading or closing keyword."),
        /Add a `## Linked issue` section/,
    );
});

test("should accept several canonical references without a heading, first as primary", async () => {
    const compliance = await validateBody("Fixes #224 and Closes #225 in one pass.");

    assert.equal(compliance.issueNumber, 224);
    assert.equal(compliance.relationship, "Fixes");
});

test("should link every issue in a grouped and-list", async () => {
    const compliance = await validateBody("## Linked issue\n\nFixes #224 and #225 in one pass.");

    assert.deepEqual(compliance.linkedIssues.map((linked) => linked.issueNumber), [224, 225]);
    assert.deepEqual(compliance.linkedIssues.map((linked) => linked.relationship), ["Fixes", "Fixes"]);
    assert.equal(compliance.issueNumber, 224);
});

test("should link every issue in a grouped comma-list", async () => {
    const compliance = await validateBody("## Linked issue\n\nCloses #224, #225");
    assert.deepEqual(compliance.linkedIssues.map((linked) => linked.issueNumber), [224, 225]);
    assert.deepEqual(compliance.linkedIssues.map((linked) => linked.relationship), ["Closes", "Closes"]);
});

test("should link every issue in a grouped ampersand-list", async () => {
    const compliance = await validateBody("## Linked issue\n\nRefs #224 & #225");
    assert.deepEqual(compliance.linkedIssues.map((linked) => linked.issueNumber), [224, 225]);
});

test("should link every issue in a comma-and-list", async () => {
    const compliance = await validateBody("## Linked issue\n\nFixes #224, and #225");
    assert.deepEqual(compliance.linkedIssues.map((linked) => linked.issueNumber), [224, 225]);
});

test("should combine grouped lists with standalone references and keep order as primary order", async () => {
    const compliance = await validateBody("## Linked issue\n\nFixes #224 and #225\nCloses #300");

    assert.deepEqual(compliance.linkedIssues.map((linked) => linked.issueNumber), [224, 225, 300]);
    assert.deepEqual(compliance.linkedIssues.map((linked) => linked.relationship), ["Fixes", "Fixes", "Closes"]);
    assert.equal(compliance.issueNumber, 224);
});

test("should accept the full closing-keyword set inside grouped lists", async () => {
    const compliance = await validateBody("## Linked issue\n\nfix #224 and resolved #225");
    assert.deepEqual(compliance.linkedIssues.map((linked) => linked.issueNumber), [224, 225]);
});

test("should accept a same-repository issue URL as a reference", async () => {
    const compliance = await validateBody(
        "## Linked issue\n\nFixes https://github.com/example/astronomical/issues/224",
    );

    assert.equal(compliance.issueNumber, 224);
    assert.equal(compliance.relationship, "Fixes");
});

test("should accept mixed number and URL designators in one grouped list", async () => {
    const compliance = await validateBody(
        "## Linked issue\n\nFixes #224 and https://github.com/example/astronomical/issues/225",
    );

    assert.deepEqual(compliance.linkedIssues.map((linked) => linked.issueNumber), [224, 225]);
});

test("should ignore foreign-repository issue URLs instead of treating them as provenance", async () => {
    const compliance = await validateBody(
        "## Linked issue\n\nFixes #224, related context at https://github.com/other-owner/other-repo/issues/225",
    );

    assert.deepEqual(compliance.linkedIssues.map((linked) => linked.issueNumber), [224]);
});

test("should deduplicate repeated references to the same issue and load it once", async () => {
    const loadedIssueNumbers = [];

    const compliance = await validateBody("## Linked issue\n\nFixes #224 and Fixes #224", {
        loadIssue: async (issueNumber) => {
            loadedIssueNumbers.push(issueNumber);
            return createIssue({ number: issueNumber });
        },
    });

    assert.deepEqual(loadedIssueNumbers, [224]);
    assert.equal(compliance.linkedIssues.length, 1);
});

test("should reject a bare issue mention with the exact line and a suggested replacement", async () => {
    await assert.rejects(
        validateBody("## Linked issue\n\nFixes #224 while #225 remains open."),
        (error) => {
            assert.match(error.message, /#225 on line 3 has no keyword/);
            assert.match(error.message, /did you mean "Refs #225"\?/);
            return true;
        },
    );
});

test("should reject a heading-less body mixing a keyword reference with a bare mention", async () => {
    await assert.rejects(
        validateBody("Fixes #224 while #225 remains open."),
        /#225 on line 1 has no keyword/,
    );
});

test("should reject a missing Linked issue section", async () => {
    await assert.rejects(
        validateBody("## Change\n\nAdd enforcement."),
        /Add a `## Linked issue` section/,
    );
});

test("should name the missing issue when a grouped list contains a nonexistent member", async () => {
    const notFoundError = new Error("Not Found");
    notFoundError.status = 404;

    await assert.rejects(
        validateBody("## Linked issue\n\nFixes #224 and #999", {
            loadIssue: async (issueNumber) => {
                if (issueNumber === 999) {
                    throw notFoundError;
                }
                return createIssue({ number: issueNumber });
            },
        }),
        /Issue #999 does not exist in this repository/,
    );
});

test("should deduplicate an issue referenced as both a number and a same-repository URL", async () => {
    const loadedIssueNumbers = [];

    const compliance = await validateBody(
        "## Linked issue\n\nFixes #224 and https://github.com/example/astronomical/issues/224",
        {
            loadIssue: async (issueNumber) => {
                loadedIssueNumbers.push(issueNumber);
                return createIssue({ number: issueNumber });
            },
        },
    );

    assert.deepEqual(loadedIssueNumbers, [224]);
    assert.equal(compliance.linkedIssues.length, 1);
});

test("should accept a grouped list when the heading is forgotten", async () => {
    const compliance = await validateBody("Fixes #224 and #225\n\n## Change\n\nAdd enforcement.");

    assert.deepEqual(compliance.linkedIssues.map((linked) => linked.issueNumber), [224, 225]);
});

test("should accept several canonical references with the first as the primary linked issue", async () => {
    const loadedIssueNumbers = [];

    const compliance = await validateBody("## Linked issue\n\nRefs #983\nRefs #1081\n\n## Change\n\nThe slice work.", {
        loadIssue: async (issueNumber) => {
            loadedIssueNumbers.push(issueNumber);
            return createIssue({ number: issueNumber, html_url: issueUrlFor(issueNumber) });
        },
    });

    assert.deepEqual(loadedIssueNumbers, [983, 1081]);
    assert.deepEqual(compliance, {
        issueNumber: 983,
        relationship: "Refs",
        issueUrl: issueUrlFor(983),
        linkedIssues: [
            { issueNumber: 983, relationship: "Refs", kind: "reference", issueUrl: issueUrlFor(983) },
            { issueNumber: 1081, relationship: "Refs", kind: "reference", issueUrl: issueUrlFor(1081) },
        ],
        warnings: [],
    });
});

test("should reject a missing Linked issue section when only a foreign URL is given", async () => {
    await assert.rejects(
        validateBody("## Linked issue\n\nSee https://github.com/other-owner/other-repo/issues/225"),
        /must reference at least one issue/,
    );
});

test("should reject malformed and cross-repository references", async () => {
    for (const reference of ["#224", "Fixes example/other#224", "Fixes #0"]) {
        await assert.rejects(
            validateBody(`## Linked issue\n\n${reference}`),
            /must reference at least one issue/,
        );
    }
});

test("should ignore a pull-request reference when a genuine issue link exists", async () => {
    const compliance = await validateBody("## Linked issue\n\nRefs #224\nCloses #225", {
        loadIssue: async (issueNumber) => {
            if (issueNumber === 225) {
                return createIssue({ number: 225, pull_request: { url: "https://api.github.com/pulls/225" } });
            }
            return createIssue();
        },
    });
    assert.equal(compliance.issueNumber, 224);
    assert.equal(compliance.linkedIssues.length, 1);
    assert.equal(compliance.warnings.length, 1);
    assert.match(compliance.warnings[0], /#225 .* references a pull request, not an issue/);
});

test("should reject a body whose every reference points at a pull request", async () => {
    await assert.rejects(
        validateBody("## Linked issue\n\nRefs #224", {
            loadIssue: async () => createIssue({ pull_request: { url: "https://api.github.com/pulls/224" } }),
        }),
        /Every reference in the Linked issue section points at a pull request \(#224\)/,
    );
});

test("should ignore a bare pull-request number in stacking prose", async () => {
    // No `## Linked issue` heading, so the whole body is scanned and the
    // stacking note's `#1458` reaches the bare-mention classifier.
    const compliance = await validateBody(
        "Fixes #224 in the model-serving crate.\n\nOriginally stacked on #1458; that pull request merged first.",
        {
            loadIssue: async (issueNumber) => {
                if (issueNumber === 1458) {
                    return createIssue({
                        number: 1458,
                        pull_request: { url: "https://api.github.com/pulls/1458" },
                    });
                }
                return createIssue();
            },
        },
    );
    assert.equal(compliance.issueNumber, 224);
    assert.deepEqual(compliance.warnings, []);
});

test("should reject a nonexistent issue with actionable guidance", async () => {
    const notFoundError = new Error("Not Found");
    notFoundError.status = 404;

    await assert.rejects(
        validateBody("## Linked issue\n\nRefs #999", {
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
        validateBody("## Linked issue\n\nRefs #224", {
            loadIssue: async () => {
                throw serviceError;
            },
        }),
        serviceError,
    );
});

test("should accept an umbrella context mention alongside a closing reference", async () => {
    const compliance = await validateBody(
        "## Linked issue\n\nResolves #1426 (part of the #1428 umbrella)",
        {
            loadIssue: async (issueNumber) => createIssue({
                number: issueNumber,
                html_url: issueUrlFor(issueNumber),
            }),
        },
    );

    assert.deepEqual(compliance.linkedIssues.map((linked) => linked.issueNumber), [1426, 1428]);
    assert.deepEqual(compliance.linkedIssues.map((linked) => linked.kind), ["closing", "context"]);
    assert.deepEqual(
        compliance.linkedIssues.map((linked) => linked.relationship),
        ["Resolves", "part of"],
    );
    assert.equal(compliance.issueNumber, 1426);
    assert.deepEqual(compliance.warnings, []);
});

test("should accept a context-only linked issue section", async () => {
    const compliance = await validateBody("## Linked issue\n\npart of #1428", {
        loadIssue: async (issueNumber) => createIssue({
            number: issueNumber,
            html_url: issueUrlFor(issueNumber),
        }),
    });

    assert.equal(compliance.issueNumber, 1428);
    assert.equal(compliance.relationship, "part of");
    assert.equal(compliance.linkedIssues[0].kind, "context");
});

test("should link every issue in a grouped context list", async () => {
    const compliance = await validateBody("## Linked issue\n\npart of #1428 and #1429", {
        loadIssue: async (issueNumber) => createIssue({
            number: issueNumber,
            html_url: issueUrlFor(issueNumber),
        }),
    });

    assert.deepEqual(compliance.linkedIssues.map((linked) => linked.issueNumber), [1428, 1429]);
    assert.deepEqual(compliance.linkedIssues.map((linked) => linked.kind), ["context", "context"]);
});

test("should accept each documented context phrase without case sensitivity", async () => {
    for (const phrase of ["part of", "Tracked by", "works toward", "UMBRELLA", "epic", "See also"]) {
        const compliance = await validateBody(`## Linked issue\n${phrase} #224`, {
            loadIssue: async (issueNumber) => createIssue({
                number: issueNumber,
                html_url: issueUrlFor(issueNumber),
            }),
        });
        assert.equal(compliance.linkedIssues[0].kind, "context", phrase);
    }
});

test("should accept a bare mention of an umbrella-labeled issue", async () => {
    const compliance = await validateBody("## Linked issue\n\nFixes #224 while #225 coordinates the work.", {
        loadIssue: async (issueNumber) => createIssue({
            number: issueNumber,
            html_url: issueUrlFor(issueNumber),
            labels: issueNumber === 225 ? [{ name: "umbrella" }] : [],
        }),
    });

    assert.deepEqual(compliance.linkedIssues.map((linked) => linked.issueNumber), [224, 225]);
    assert.equal(compliance.linkedIssues[1].kind, "context");
    assert.equal(compliance.linkedIssues[1].relationship, "umbrella");
    assert.equal(compliance.issueNumber, 224);
});

test("should accept a bare mention of an epic-labeled issue given as a plain label string", async () => {
    const compliance = await validateBody("## Linked issue\n\nFixes #224 while #225 coordinates the work.", {
        loadIssue: async (issueNumber) => createIssue({
            number: issueNumber,
            html_url: issueUrlFor(issueNumber),
            labels: issueNumber === 225 ? ["epic"] : [],
        }),
    });

    assert.deepEqual(compliance.linkedIssues.map((linked) => linked.issueNumber), [224, 225]);
});

test("should reject a bare mention of a plain issue even when other references exist", async () => {
    await assert.rejects(
        validateBody("## Linked issue\n\nFixes #224 while #225 remains open.", {
            loadIssue: async (issueNumber) => createIssue({
                number: issueNumber,
                html_url: issueUrlFor(issueNumber),
                labels: [],
            }),
        }),
        /#225 on line 3 has no keyword/,
    );
});

test("should prefer the first closing reference as primary even when a context reference comes first", async () => {
    const compliance = await validateBody("## Linked issue\n\npart of #1428\nFixes #224", {
        loadIssue: async (issueNumber) => createIssue({
            number: issueNumber,
            html_url: issueUrlFor(issueNumber),
        }),
    });

    assert.equal(compliance.issueNumber, 224);
    assert.equal(compliance.relationship, "Fixes");
});

test("should warn without failing when a closing keyword targets an already-closed issue", async () => {
    const compliance = await validateBody("## Linked issue\n\nFixes #224", {
        loadIssue: async () => createIssue({ state: "closed" }),
    });

    assert.equal(compliance.issueNumber, 224);
    assert.equal(compliance.warnings.length, 1);
    assert.match(compliance.warnings[0], /#224 on line 3/);
    assert.match(compliance.warnings[0], /already closed/);
});

test("should not warn for context references to already-closed issues", async () => {
    const compliance = await validateBody("## Linked issue\n\npart of #1428", {
        loadIssue: async () => createIssue({ state: "closed" }),
    });

    assert.deepEqual(compliance.warnings, []);
});

test("should not warn when the non-closing Refs keyword targets an already-closed issue", async () => {
    const compliance = await validateBody("## Linked issue\n\nRefs #224", {
        loadIssue: async () => createIssue({ state: "closed" }),
    });

    assert.equal(compliance.linkedIssues[0].kind, "reference");
    assert.deepEqual(compliance.warnings, []);
});

test("should reject an article between a closing keyword and its issue", async () => {
    // GitHub's closing syntax requires the designator to follow the keyword
    // directly; accepting `Fixes the #224` would publish success while the
    // issue never auto-closes at merge time.
    await assert.rejects(
        validateBody("## Linked issue\n\nFixes the #224 regression"),
        /must reference at least one issue/,
    );
    await assert.rejects(
        validateBody("## Linked issue\n\nFixes that #224 regression"),
        /must reference at least one issue/,
    );
});

test("should report body-absolute line numbers across a stripped multi-line comment", async () => {
    await assert.rejects(
        validateBody("## Linked issue\n\n<!-- multi\nline\ncomment -->\nFixes #224 while #225 remains open"),
        (error) => {
            assert.match(error.message, /#225 on line 6 has no keyword/);
            return true;
        },
    );
});

test("should report the designator's line when a grouped list wraps", async () => {
    const compliance = await validateBody("## Linked issue\n\nFixes #224,\n#225", {
        loadIssue: async (issueNumber) => createIssue({
            number: issueNumber,
            html_url: issueUrlFor(issueNumber),
            state: "closed",
        }),
    });

    assert.deepEqual(compliance.linkedIssues.map((linked) => linked.issueNumber), [224, 225]);
    assert.match(compliance.warnings[0], /#224 on line 3/);
    assert.match(compliance.warnings[1], /#225 on line 4/);
});
