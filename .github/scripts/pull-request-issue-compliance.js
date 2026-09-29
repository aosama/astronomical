// This validator owns the repository's pull-request-to-issue contract. Keeping
// the policy pure makes every contributor-facing failure reproducible locally.

// Dependency and workflow bots open pull requests programmatically and cannot
// author issue-linked bodies, so the provenance contract is enforced on human
// pull requests only. The suffix match covers Dependabot, Renovate, and any
// future automation GitHub names with the [bot] marker.
function isAutomatedPullRequest(pullRequestAuthorLogin) {
    return /\[bot\]$/i.test(String(pullRequestAuthorLogin ?? ""));
}

function removeHtmlComments(markdown) {
    return markdown.replace(/<!--[\s\S]*?-->/g, "");
}

// A provenance reference is a GitHub closing keyword plus a same-repository
// issue number. The leading word boundary keeps words like "Prefixes" from
// matching, and the [1-9] start keeps "#0"-style placeholders out.
const ISSUE_REFERENCE_PATTERN = /\b(Fixes|Closes|Resolves|Refs)\s+#([1-9]\d*)\b/gi;

const SECTION_REFERENCE_FAILURE =
    "The `## Linked issue` section must contain exactly one same-repository reference: `Fixes #N`, `Closes #N`, `Resolves #N`, or `Refs #N`.";
const MISSING_SECTION_FAILURE =
    "Add a `## Linked issue` section containing `Fixes #N` for implementation work or `Refs #N` for documentation, CI, and maintenance work; a bare `Fixes #N` line anywhere in the body is also accepted.";
const AMBIGUOUS_BODY_FAILURE =
    "Keep exactly one same-repository issue reference in the pull request body: `Fixes #N`, `Closes #N`, `Resolves #N`, or `Refs #N`.";

function findIssueReferences(text) {
    return [...String(text ?? "").matchAll(ISSUE_REFERENCE_PATTERN)].map((match) => ({
        relationship: match[1],
        issueNumber: Number(match[2]),
    }));
}

function findMentionedIssueNumbers(text) {
    return [...new Set(
        [...String(text ?? "").matchAll(/#([1-9]\d*)\b/g)].map((match) => Number(match[1])),
    )];
}

function extractLinkedIssue(pullRequestBody) {
    const bodyLines = String(pullRequestBody ?? "").split(/\r?\n/);
    const linkedIssueHeadingIndex = bodyLines.findIndex((line) =>
        /^#{0,6}\s*linked issue\s*:?\s*$/i.test(line.trim()),
    );

    // Contributors reliably forget the `##` on the heading, or skip the section
    // and write `Fixes #N` straight into the summary, so a missing heading falls
    // back to scanning the whole body. A present heading stays authoritative:
    // only its section is searched, preserving the stricter behavior.
    let referenceSearchText;
    if (linkedIssueHeadingIndex === -1) {
        referenceSearchText = bodyLines.join("\n");
    } else {
        const linkedIssueSectionLines = [];
        for (let lineIndex = linkedIssueHeadingIndex + 1; lineIndex < bodyLines.length; lineIndex += 1) {
            if (/^##\s+/.test(bodyLines[lineIndex].trim())) {
                break;
            }
            linkedIssueSectionLines.push(bodyLines[lineIndex]);
        }
        referenceSearchText = linkedIssueSectionLines.join("\n");
    }

    const commentsRemoved = removeHtmlComments(referenceSearchText);
    const references = findIssueReferences(commentsRemoved);
    const referencedIssueNumbers = [...new Set(references.map((reference) => reference.issueNumber))];
    // Every issue number mentioned in the searched text counts toward the
    // single-reference rule, keyword or not, so `Fixes #224 and #225` cannot
    // smuggle a second unlinked issue past the provenance contract.
    const mentionedIssueNumbers = findMentionedIssueNumbers(commentsRemoved);
    if (referencedIssueNumbers.length === 1 && mentionedIssueNumbers.length === 1) {
        return {
            relationship: references[0].relationship,
            issueNumber: referencedIssueNumbers[0],
        };
    }
    if (linkedIssueHeadingIndex !== -1) {
        throw new Error(SECTION_REFERENCE_FAILURE);
    }
    throw new Error(references.length === 0 ? MISSING_SECTION_FAILURE : AMBIGUOUS_BODY_FAILURE);
}

async function validatePullRequestIssue({ pullRequestBody, loadIssue }) {
    const { relationship, issueNumber } = extractLinkedIssue(pullRequestBody);

    let linkedIssue;
    try {
        linkedIssue = await loadIssue(issueNumber);
    } catch (error) {
        if (error?.status === 404) {
            throw new Error(`Issue #${issueNumber} does not exist in this repository.`);
        }
        throw error;
    }

    if (linkedIssue.pull_request !== undefined) {
        throw new Error(`#${issueNumber} identifies a pull request, not an issue.`);
    }
    return {
        issueNumber,
        relationship,
        issueUrl: linkedIssue.html_url,
    };
}

module.exports = {
    isAutomatedPullRequest,
    validatePullRequestIssue,
};
