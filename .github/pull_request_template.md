## Linked issue

<!-- The pull-request issue-compliance check (.github/scripts/pull-request-issue-compliance.js) fails this pull request until this section references at least one real issue of this repository, and every #N written here must carry a keyword. The numbers in the examples below are fictitious; replace them with your real issue numbers. An example left in place fails loudly, because the referenced issue does not exist. -->

<!-- Closes the issue when this pull request merges (GitHub's closing keywords, case-insensitive, optional colon): `Fixes #987654`, `Closes #987654`, `Resolves #987654`. -->

<!-- Keeps the issue open: `Refs #987654`, or a context phrase: `part of #987654`, `tracked by #987654`, `part of the #987654 umbrella`, `epic #987654`, `see also #987654`. An issue labeled `umbrella` or `epic` may also be mentioned bare, without any keyword. -->

<!-- One keyword covers a group joined by `,`, `and`, or `&`: `Fixes #987654 and #987655`, `Fixes #987654, #987655, and #987656`, `Fixes #987654 & #987655`. The same-repository issue URL (`https://github.com/aosama/astronomical/issues/987654`) is accepted wherever a `#N` shorthand is. -->

<!-- Worked example — adapt, and replace the numbers with your real issues:
Fixes #987654 and #987655
Refs #987656
part of the #987657 umbrella
-->

<!-- Rejected by the check: a bare `#987654` with no keyword (the failure message suggests `Refs #987654`); a reference whose number is another pull request (pull-request references never count toward issue linkage); any issue number that does not exist in this repository. -->

## Github Issues

List the applicable github issues whether pre-existing in the backlog or you created them specific for this pull request.

<!-- Only the Linked issue section above is scanned for the linkage contract, so issues mentioned here as plain prose need no keywords. Keep the `## Linked issue` heading in place: without it the whole body is scanned, and every #N anywhere in the body must then carry a keyword. -->

## Problem

Describe the concrete problem and why it belongs in Astronomical.

## Change

Describe the implemented solution and any removed complexity.

## Verification

List the exact focused and repository checks that passed. State any hardware-dependent acceptance that could not run.

## Safety

- [ ] All unit tests passed locally before I opened this PR.
- [ ] All integration tests passed locally before I opened this PR.
- [ ] No credentials, personal paths, prompts, model inventories, proprietary artifacts, or machine logs are included.
- [ ] Model precision and output semantics are preserved or explicitly justified.
- [ ] Performance and memory effects are measured when you changed code in the critical path for inference.
- [ ] Source reuse and third-party licensing are accurately attributed.
