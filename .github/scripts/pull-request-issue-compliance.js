// This validator owns the repository's pull-request-to-issue contract. Keeping
// the policy pure makes every contributor-facing failure reproducible locally.

function removeHtmlComments(markdown) {
    return markdown.replace(/<!--[\s\S]*?-->/g, "");
}

function extractLinkedIssue(pullRequestBody) {
    const bodyLines = String(pullRequestBody ?? "").split(/\r?\n/);
    const linkedIssueHeadingIndex = bodyLines.findIndex((line) =>
        /^##\s+Linked issue\s*#*\s*$/i.test(line.trim()),
    );

    if (linkedIssueHeadingIndex === -1) {
        throw new Error(
            "Add a `## Linked issue` section containing `Fixes #N` for implementation work or `Refs #N` for documentation, CI, and maintenance work.",
        );
    }

    const linkedIssueSectionLines = [];
    for (let lineIndex = linkedIssueHeadingIndex + 1; lineIndex < bodyLines.length; lineIndex += 1) {
        if (/^##\s+/.test(bodyLines[lineIndex].trim())) {
            break;
        }
        linkedIssueSectionLines.push(bodyLines[lineIndex]);
    }

    const linkedIssueReference = removeHtmlComments(linkedIssueSectionLines.join("\n")).trim();
    const referenceMatch = /^(Fixes|Closes|Resolves|Refs)\s+#([1-9]\d*)$/i.exec(linkedIssueReference);
    if (referenceMatch === null) {
        throw new Error(
            "The `## Linked issue` section must contain exactly one same-repository reference: `Fixes #N`, `Closes #N`, `Resolves #N`, or `Refs #N`.",
        );
    }

    return {
        relationship: referenceMatch[1],
        issueNumber: Number(referenceMatch[2]),
    };
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
    validatePullRequestIssue,
};
