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

function escapeRegExp(text) {
    return String(text).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// GitHub's documented closing-keyword set (case-insensitive, optional colon,
// as in `Resolves: #42`) plus this repository's `Refs` for work that must
// stay open after the pull request merges.
const CLOSING_KEYWORD_PATTERN =
    "close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved|refs";

// A designator is either `#N` or a same-repository issue URL, because GitHub
// closes the issue through the URL exactly like the shorthand. Foreign-repo
// URLs never match, so related-context links stay plain prose.
function issueDesignatorPattern(repositoryFullName) {
    const sameRepositoryIssueUrlPrefix =
        escapeRegExp(`https://github.com/${repositoryFullName}/issues/`);
    return `#([1-9]\\d*)\\b|${sameRepositoryIssueUrlPrefix}([1-9]\\d*)`;
}

function createIssueReferencePattern(repositoryFullName) {
    return new RegExp(
        `\\b(${CLOSING_KEYWORD_PATTERN}):?[ \\t]+(?:${issueDesignatorPattern(repositoryFullName)})`,
        "gi",
    );
}

// A grouped list (`Fixes #224 and #225`) shares one keyword: designators
// joined only by `,`, `and`, or `&` carry the introducing keyword's
// relationship.
function createGroupedContinuationPattern(repositoryFullName) {
    return new RegExp(
        `(?:,\\s*(?:and\\s+)?|\\s+and\\s+|\\s*&\\s*)(?:${issueDesignatorPattern(repositoryFullName)})`,
        "gi",
    );
}

function findIssueReferences(searchText, repositoryFullName) {
    const referencePattern = createIssueReferencePattern(repositoryFullName);
    const groupedContinuationPattern = createGroupedContinuationPattern(repositoryFullName);
    const references = [];
    let searchStartIndex = 0;

    while (searchStartIndex < searchText.length) {
        referencePattern.lastIndex = searchStartIndex;
        const referenceMatch = referencePattern.exec(searchText);
        if (referenceMatch === null) {
            break;
        }

        const relationship = referenceMatch[1];
        references.push({
            relationship,
            issueNumber: Number(referenceMatch[2] ?? referenceMatch[3]),
        });
        let groupEndIndex = referencePattern.lastIndex;

        while (groupEndIndex < searchText.length) {
            groupedContinuationPattern.lastIndex = groupEndIndex;
            const continuationMatch = groupedContinuationPattern.exec(searchText);
            // A continuation must start exactly where the previous designator
            // ended; intervening prose such as "while #225 remains open"
            // otherwise leaves #225 an unlinked bare mention that slips past
            // the contract.
            if (continuationMatch === null || continuationMatch.index !== groupEndIndex) {
                break;
            }
            references.push({
                relationship,
                issueNumber: Number(continuationMatch[1] ?? continuationMatch[2]),
            });
            groupEndIndex = groupedContinuationPattern.lastIndex;
        }

        searchStartIndex = groupEndIndex;
    }
    return references;
}

// The same issue referenced several times is provenance once: it is loaded
// once and reported once, keeping the published check free of duplicates.
function dedupeIssueReferences(references) {
    const firstReferenceByIssueNumber = new Map();
    for (const reference of references) {
        if (!firstReferenceByIssueNumber.has(reference.issueNumber)) {
            firstReferenceByIssueNumber.set(reference.issueNumber, reference);
        }
    }
    return [...firstReferenceByIssueNumber.values()];
}

function findMentionedIssueNumbers(text) {
    return [...new Set(
        [...String(text ?? "").matchAll(/#([1-9]\d*)\b/g)].map((match) => Number(match[1])),
    )];
}

const SECTION_REFERENCE_FAILURE =
    "The `## Linked issue` section must reference at least one issue, and every `#N` in the section must use a canonical keyword: `Fixes #N`, `Closes #N`, `Resolves #N`, or `Refs #N`. Grouped forms such as `Fixes #N and #M` are accepted.";
const MISSING_SECTION_FAILURE =
    "Add a `## Linked issue` section containing `Fixes #N` for implementation work or `Refs #N` for documentation, CI, and maintenance work; a bare `Fixes #N` line anywhere in the body is also accepted.";
const AMBIGUOUS_BODY_FAILURE =
    "Link every issue number mentioned in the pull request body with a canonical keyword: `Fixes #N`, `Closes #N`, `Resolves #N`, or `Refs #N`; grouped forms such as `Fixes #N and #M` are accepted.";

function extractLinkedIssue(pullRequestBody, repositoryFullName) {
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
    const linkedReferences =
        dedupeIssueReferences(findIssueReferences(commentsRemoved, repositoryFullName));
    const referencedIssueNumbers = linkedReferences.map((reference) => reference.issueNumber);
    // Every issue number mentioned in the searched text must carry a
    // canonical keyword, so `Fixes #224 while #225 remains open` cannot
    // smuggle an unlinked issue past the provenance contract. A slice
    // legitimately touches several issues, so one or more canonical
    // references are accepted and the first one is the primary linked issue.
    const mentionedIssueNumbers = findMentionedIssueNumbers(commentsRemoved);
    const everyMentionIsReferenced = mentionedIssueNumbers
        .every((issueNumber) => referencedIssueNumbers.includes(issueNumber));
    if (linkedReferences.length >= 1 && everyMentionIsReferenced) {
        return linkedReferences;
    }
    if (linkedIssueHeadingIndex !== -1) {
        throw new Error(SECTION_REFERENCE_FAILURE);
    }
    throw new Error(linkedReferences.length === 0 ? MISSING_SECTION_FAILURE : AMBIGUOUS_BODY_FAILURE);
}

async function validatePullRequestIssue({ pullRequestBody, repositoryFullName, loadIssue }) {
    const linkedReferences = extractLinkedIssue(pullRequestBody, repositoryFullName);

    const issueUrlByIssueNumber = new Map();
    for (const linkedReference of linkedReferences) {
        let linkedIssue;
        try {
            linkedIssue = await loadIssue(linkedReference.issueNumber);
        } catch (error) {
            if (error?.status === 404) {
                throw new Error(`Issue #${linkedReference.issueNumber} does not exist in this repository.`);
            }
            throw error;
        }

        if (linkedIssue.pull_request !== undefined) {
            throw new Error(`#${linkedReference.issueNumber} identifies a pull request, not an issue.`);
        }
        issueUrlByIssueNumber.set(linkedReference.issueNumber, linkedIssue.html_url);
    }

    const linkedIssues = linkedReferences.map((linkedReference) => ({
        issueNumber: linkedReference.issueNumber,
        relationship: linkedReference.relationship,
        issueUrl: issueUrlByIssueNumber.get(linkedReference.issueNumber),
    }));
    return {
        issueNumber: linkedIssues[0].issueNumber,
        relationship: linkedIssues[0].relationship,
        issueUrl: linkedIssues[0].issueUrl,
        linkedIssues,
    };
}

module.exports = {
    isAutomatedPullRequest,
    validatePullRequestIssue,
};
