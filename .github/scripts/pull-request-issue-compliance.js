// This validator owns the repository's pull-request-to-issue contract. Keeping
// the policy pure makes every contributor-facing failure reproducible locally.

// Dependency and workflow bots open pull requests programmatically and cannot
// author issue-linked bodies, so the provenance contract is enforced on human
// pull requests only. The suffix match covers Dependabot, Renovate, and any
// future automation GitHub names with the [bot] marker.
function isAutomatedPullRequest(pullRequestAuthorLogin) {
    const authorLoginText =
        pullRequestAuthorLogin != null ? String(pullRequestAuthorLogin) : "";
    return /\[bot\]$/i.test(authorLoginText);
}

// Blank out HTML comments while preserving every newline, so reported line
// numbers stay body-absolute even when a comment spans several lines, and
// commented-out references can never be scanned as prose.
function removeHtmlComments(markdown) {
    return markdown.replace(/<!--[\s\S]*?-->/g, (comment) => comment.replace(/[^\n]/g, ""));
}

function escapeRegExp(text) {
    return String(text).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// GitHub's documented closing-keyword set (case-insensitive, optional colon,
// as in `Resolves: #42`). Only these keywords make GitHub close an issue at
// merge time, so only they claim closing semantics.
const GITHUB_CLOSING_KEYWORD_PATTERN =
    "close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved";
// This repository additionally accepts `Refs` for work that must stay open
// after the pull request merges. It shares the acceptance alternation but is
// classified as its own `reference` kind below, so it never triggers the
// stale-closing warning and never implies merge-time auto-close.
const CLOSING_KEYWORD_PATTERN = `${GITHUB_CLOSING_KEYWORD_PATTERN}|refs`;
const GITHUB_CLOSING_KEYWORD_ONLY_PATTERN =
    new RegExp(`^(?:${GITHUB_CLOSING_KEYWORD_PATTERN})$`, "i");

// Context phrases name relationships that are real but not closing: an
// umbrella or epic that stays open after this pull request merges, a parent
// that tracks this work, related context. GitHub only closes an issue through
// a closing keyword, so requiring one on every `#N` forced writers to
// rephrase natural sentences like `Resolves #1426 (part of the #1428
// umbrella)`; a context phrase satisfies the linkage contract instead.
const CONTEXT_KEYWORD_PATTERN =
    "part of|tracked by|works toward|contributes to|belongs to|relates to|parent issue|umbrella|epic|see also";

// A short article may sit between a context phrase and its designator, as in
// `part of the #1428 umbrella`. The filler belongs to the context branch of
// the reference pattern only: GitHub's closing syntax requires the designator
// to follow the keyword directly, so widening `Fixes #N` with articles would
// accept bodies GitHub will not auto-close.
const CONTEXT_ARTICLE_FILLER_PATTERN = "(?:[ \\t]+(?:the|this|that|these|those))?";
// A captured context relationship may carry the trailing article; trimming
// keeps reported relationships as the bare phrases.
const CONTEXT_ARTICLE_FILLER_TRIM_PATTERN = /[ \t]+(?:the|this|that|these|those)$/i;

// Issues labeled `umbrella` or `epic` are coordination containers by
// definition, so a bare mention of one already states the relationship and is
// accepted as a context reference without any phrase.
const UMBRELLA_LABEL_PATTERN = /^(?:umbrella|epic)$/i;

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
        `\\b(${CLOSING_KEYWORD_PATTERN}|(?:${CONTEXT_KEYWORD_PATTERN})${CONTEXT_ARTICLE_FILLER_PATTERN}):?[ \\t]+(?:${issueDesignatorPattern(repositoryFullName)})`,
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

// Line numbers are reported against the comment-stripped search text so a
// failure can point at the exact offending reference instead of a whole-body
// verdict (issue #1450).
function createLineNumberLookup(searchText) {
    const newlineIndexes = [];
    for (let characterIndex = 0; characterIndex < searchText.length; characterIndex += 1) {
        if (searchText[characterIndex] === "\n") {
            newlineIndexes.push(characterIndex);
        }
    }
    return function lineNumberAt(textIndex) {
        let lowBound = 0;
        let highBound = newlineIndexes.length;
        while (lowBound < highBound) {
            const middleIndex = Math.floor((lowBound + highBound) / 2);
            if (newlineIndexes[middleIndex] < textIndex) {
                lowBound = middleIndex + 1;
            } else {
                highBound = middleIndex;
            }
        }
        return lowBound + 1;
    };
}

function relationshipKind(relationshipText) {
    if (GITHUB_CLOSING_KEYWORD_ONLY_PATTERN.test(relationshipText)) {
        return "closing";
    }
    return relationshipText.toLowerCase() === "refs" ? "reference" : "context";
}

function findIssueReferences(searchText, repositoryFullName) {
    const referencePattern = createIssueReferencePattern(repositoryFullName);
    const groupedContinuationPattern = createGroupedContinuationPattern(repositoryFullName);
    const lineNumberAt = createLineNumberLookup(searchText);
    const references = [];
    let searchStartIndex = 0;

    while (searchStartIndex < searchText.length) {
        referencePattern.lastIndex = searchStartIndex;
        const referenceMatch = referencePattern.exec(searchText);
        if (referenceMatch === null) {
            break;
        }

        const rawRelationship = referenceMatch[1];
        // The designator pattern is an alternation, so exactly one of the two
        // capture groups matched.
        const referenceDesignatorText =
            referenceMatch[2] != null ? referenceMatch[2] : referenceMatch[3];
        const relationship = rawRelationship.replace(CONTEXT_ARTICLE_FILLER_TRIM_PATTERN, "");
        const referenceKind = relationshipKind(relationship);
        references.push({
            relationship,
            kind: referenceKind,
            issueNumber: Number(referenceDesignatorText),
            lineNumber: lineNumberAt(referenceMatch.index),
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
            const continuationDesignatorText =
                continuationMatch[1] != null ? continuationMatch[1] : continuationMatch[2];
            // The match starts at the `,`/`and` joiner, which can sit on the
            // previous line when a grouped list wraps; the designator's own
            // index is what the reported line number must point at.
            const continuationDesignatorIndex =
                continuationMatch.index + continuationMatch[0].length - continuationDesignatorText.length;
            references.push({
                relationship,
                kind: referenceKind,
                issueNumber: Number(continuationDesignatorText),
                lineNumber: lineNumberAt(continuationDesignatorIndex),
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
    const searchableText = text != null ? String(text) : "";
    const lineNumberAt = createLineNumberLookup(searchableText);
    return [...searchableText.matchAll(/#([1-9]\d*)\b/g)].map((match) => ({
        issueNumber: Number(match[1]),
        lineNumber: lineNumberAt(match.index),
    }));
}

const SECTION_REFERENCE_FAILURE =
    "The `## Linked issue` section must reference at least one issue, and every `#N` in the section must use a canonical closing keyword (`Fixes #N`, `Closes #N`, `Resolves #N`, `Refs #N`) or a context phrase (`part of #N`, `tracked by #N`, `umbrella #N`, `epic #N`). Grouped forms such as `Fixes #N and #M` are accepted.";
const MISSING_SECTION_FAILURE =
    "Add a `## Linked issue` section containing `Fixes #N` for implementation work or `Refs #N` for documentation, CI, and maintenance work; context phrases such as `part of #N` are accepted when the issue stays open after this pull request merges; a bare `Fixes #N` line anywhere in the body is also accepted.";

function extractLinkedIssue(pullRequestBody, repositoryFullName) {
    const bodyText = pullRequestBody != null ? String(pullRequestBody) : "";
    const bodyLines = bodyText.split(/\r?\n/);
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
    const linkedReferences = findIssueReferences(commentsRemoved, repositoryFullName);
    if (linkedReferences.length === 0) {
        if (linkedIssueHeadingIndex !== -1) {
            throw new Error(SECTION_REFERENCE_FAILURE);
        }
        throw new Error(MISSING_SECTION_FAILURE);
    }
    return {
        references: linkedReferences,
        mentionedMentions: findMentionedIssueNumbers(commentsRemoved),
        // Reported line numbers are body-absolute: the searched text starts
        // one line after the heading, or at the body's first line when the
        // heading is missing.
        bodyLineOffset: linkedIssueHeadingIndex === -1 ? 0 : linkedIssueHeadingIndex + 1,
    };
}

function issueLabelNames(loadedIssue) {
    const labels = Array.isArray(loadedIssue?.labels) ? loadedIssue.labels : [];
    return labels
        .map((label) => (typeof label === "string" ? label : label?.name))
        .filter((labelName) => typeof labelName === "string");
}

function isUmbrellaLikeIssue(loadedIssue) {
    return issueLabelNames(loadedIssue).some((labelName) => UMBRELLA_LABEL_PATTERN.test(labelName));
}

function bareMentionFailure(bareMention, bodyLineOffset) {
    const bodyLineNumber = bareMention.lineNumber + bodyLineOffset;
    return [
        `#${bareMention.issueNumber} on line ${bodyLineNumber} has no keyword — did you mean "Refs #${bareMention.issueNumber}"?`,
        `A context phrase such as "part of #${bareMention.issueNumber}" also satisfies the contract when the issue stays open after this pull request merges.`,
    ].join(" ");
}

async function validatePullRequestIssue({ pullRequestBody, repositoryFullName, loadIssue }) {
    const extraction = extractLinkedIssue(pullRequestBody, repositoryFullName);
    const references = dedupeIssueReferences(extraction.references);

    const loadedIssueByIssueNumber = new Map();
    async function loadIssueOnce(issueNumber) {
        if (loadedIssueByIssueNumber.has(issueNumber)) {
            return loadedIssueByIssueNumber.get(issueNumber);
        }
        let loadedIssue;
        try {
            loadedIssue = await loadIssue(issueNumber);
        } catch (error) {
            if (error?.status === 404) {
                throw new Error(`Issue #${issueNumber} does not exist in this repository.`);
            }
            throw error;
        }
        loadedIssueByIssueNumber.set(issueNumber, loadedIssue);
        return loadedIssue;
    }

    const warnings = [];
    // A `#N` that resolves to a pull request is a GitHub cross-reference to
    // sibling work, not a claim of issue provenance. Stacking notes such as
    // "stacked on #1458" used to fail the whole check even when the body also
    // carried a perfectly valid issue link, so pull-request-shaped references
    // are skipped here (with an advisory warning) and the contract falls back
    // to the remaining genuine issue references.
    const issueReferences = [];
    for (const reference of references) {
        const referencedIssue = await loadIssueOnce(reference.issueNumber);
        if (referencedIssue.pull_request !== undefined) {
            warnings.push(
                `#${reference.issueNumber} on line ${reference.lineNumber + extraction.bodyLineOffset} references a pull request, not an issue; pull-request mentions do not count toward the issue-linkage contract.`,
            );
            continue;
        }
        issueReferences.push(reference);
    }
    if (issueReferences.length === 0 && references.length > 0) {
        const pullRequestNumbers = references.map((reference) => `#${reference.issueNumber}`).join(", ");
        throw new Error(
            `Every reference in the Linked issue section points at a pull request (${pullRequestNumbers}); the contract needs at least one issue reference such as \`Fixes #N\`.`,
        );
    }

    // Closing references state merge-time behavior; context references and
    // umbrella-labeled bare mentions are meaningful relationships that must
    // not force a closing keyword onto a naturally written sentence
    // (issue #1450).
    const referencedIssueNumbers = new Set(issueReferences.map((reference) => reference.issueNumber));
    const classifiedBareIssueNumbers = new Set();
    const promotedUmbrellaReferences = [];
    for (const bareMention of extraction.mentionedMentions) {
        if (referencedIssueNumbers.has(bareMention.issueNumber)
            || classifiedBareIssueNumbers.has(bareMention.issueNumber)) {
            continue;
        }
        classifiedBareIssueNumbers.add(bareMention.issueNumber);
        const mentionedIssue = await loadIssueOnce(bareMention.issueNumber);
        // A bare pull-request number is ordinary GitHub prose (a
        // cross-reference to sibling work), so it is ignored exactly like a
        // keyworded pull-request reference above.
        if (mentionedIssue.pull_request !== undefined) {
            continue;
        }
        if (isUmbrellaLikeIssue(mentionedIssue)) {
            promotedUmbrellaReferences.push({
                relationship: "umbrella",
                kind: "context",
                issueNumber: bareMention.issueNumber,
                lineNumber: bareMention.lineNumber,
            });
            continue;
        }
        throw new Error(bareMentionFailure(bareMention, extraction.bodyLineOffset));
    }

    // A closing keyword pointed at an already-closed issue cannot close
    // anything at merge time, which usually signals a stale template or a
    // copy-pasted body; that is a warning, not a rejection, because the
    // linkage itself is still well-formed.
    for (const reference of issueReferences) {
        if (reference.kind !== "closing") {
            continue;
        }
        const referencedIssue = loadedIssueByIssueNumber.get(reference.issueNumber);
        if (referencedIssue.state === "closed") {
            warnings.push(
                `#${reference.issueNumber} on line ${reference.lineNumber + extraction.bodyLineOffset} uses the closing keyword \`${reference.relationship}\`, but issue #${reference.issueNumber} is already closed; if this pull request does not close it, a context phrase such as "part of #${reference.issueNumber}" states the relationship without the stale closing promise.`,
            );
        }
    }

    const linkedIssues = [...issueReferences, ...promotedUmbrellaReferences].map((reference) => ({
        issueNumber: reference.issueNumber,
        relationship: reference.relationship,
        kind: reference.kind,
        issueUrl: loadedIssueByIssueNumber.get(reference.issueNumber).html_url,
    }));
    // The primary linked issue is the first closing reference when one
    // exists, because it names the issue this pull request actually closes;
    // a `Refs` reference is next, and a context-only body still links its
    // first issue for provenance.
    const primaryLinkedIssue =
        linkedIssues.find((linked) => linked.kind === "closing")
        ?? linkedIssues.find((linked) => linked.kind === "reference")
        ?? linkedIssues[0];
    return {
        issueNumber: primaryLinkedIssue.issueNumber,
        relationship: primaryLinkedIssue.relationship,
        issueUrl: primaryLinkedIssue.issueUrl,
        linkedIssues,
        warnings,
    };
}

module.exports = {
    isAutomatedPullRequest,
    validatePullRequestIssue,
};
