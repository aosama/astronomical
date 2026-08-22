// This validator owns the repository's pull-request-to-issue contract. Keeping
// the policy pure makes every contributor-facing failure reproducible locally.

const REQUIRED_ISSUE_SECTIONS = [
    "Goal",
    "Evidence",
    "Scope",
    "Constraints",
    "Acceptance criteria",
];

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

function findIncompleteIssueSections(issueBody) {
    const issueBodyWithoutComments = removeHtmlComments(String(issueBody ?? ""));
    const sectionContentByNormalizedHeading = new Map();
    let activeHeading = null;

    for (const line of issueBodyWithoutComments.split(/\r?\n/)) {
        const headingMatch = /^#{2,6}\s+(.+?)\s*#*\s*$/.exec(line.trim());
        if (headingMatch !== null) {
            activeHeading = headingMatch[1].trim().toLocaleLowerCase("en-US");
            if (!sectionContentByNormalizedHeading.has(activeHeading)) {
                sectionContentByNormalizedHeading.set(activeHeading, []);
            }
            continue;
        }

        if (activeHeading !== null) {
            sectionContentByNormalizedHeading.get(activeHeading).push(line);
        }
    }

    return REQUIRED_ISSUE_SECTIONS.filter((requiredHeading) => {
        const sectionLines = sectionContentByNormalizedHeading.get(
            requiredHeading.toLocaleLowerCase("en-US"),
        );
        if (sectionLines === undefined) {
            return true;
        }

        const sectionContent = sectionLines.join("\n").trim();
        return sectionContent.length === 0 || sectionContent === "_No response_";
    });
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
    if (linkedIssue.state !== "open") {
        throw new Error(`Issue #${issueNumber} must remain open until this pull request merges.`);
    }

    const incompleteSections = findIncompleteIssueSections(linkedIssue.body);
    if (incompleteSections.length > 0) {
        throw new Error(
            `Please complete these issue sections: ${incompleteSections.join(", ")}.`,
        );
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
