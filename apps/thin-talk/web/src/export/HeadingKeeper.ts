/**
 * Keeps a heading on the same printed page as what it introduces.
 *
 * The stylesheet already asks for this with `break-after: avoid`, but the
 * WebKit that prints the PDF does not follow that hint: a heading was left
 * alone at the foot of a page with its diagram on the next. WebKit does
 * follow `break-inside: avoid`, so the heading and the block after it are
 * put in one group that carries that rule instead.
 */

/** The group a heading and the block it introduces are printed in. */
export const KEPT_TOGETHER_CLASS = "export-keep";

const HEADING_TAGS = new Set(["H1", "H2", "H3", "H4"]);
/** A list can run for most of a page. Kept whole with its heading it would
 * be pushed to the next page and leave the first one half empty, which looks
 * worse than the stranded heading it was meant to prevent. */
const LONG_RUNNING_TAGS = new Set(["UL", "OL"]);

function isHeading(block: Element): boolean {
  return HEADING_TAGS.has(block.tagName);
}

/** The headings that follow one another from `firstHeading`, then the one
 * block they introduce, or only the headings when that block is a list. */
function groupStartingAt(firstHeading: Element): Element[] {
  const groupedBlocks: Element[] = [firstHeading];
  let followingBlock = firstHeading.nextElementSibling;
  while (followingBlock !== null && isHeading(followingBlock)) {
    groupedBlocks.push(followingBlock);
    followingBlock = followingBlock.nextElementSibling;
  }
  if (followingBlock !== null && !LONG_RUNNING_TAGS.has(followingBlock.tagName)) {
    groupedBlocks.push(followingBlock);
  }
  return groupedBlocks;
}

/**
 * The body's markup with each heading grouped with the block after it.
 * The body itself is left as it was: the Word writer reads the same body
 * and keeps headings with their text through Word's own heading styles.
 */
export function markupKeepingHeadingsWithTheirBlocks(body: HTMLElement): string {
  const pagedBody = body.cloneNode(true) as HTMLElement;
  let block = pagedBody.firstElementChild;
  while (block !== null) {
    if (!isHeading(block)) {
      block = block.nextElementSibling;
      continue;
    }
    const groupedBlocks = groupStartingAt(block);
    if (groupedBlocks.length > 1) {
      const keptGroup = pagedBody.ownerDocument.createElement("div");
      keptGroup.className = KEPT_TOGETHER_CLASS;
      block.before(keptGroup);
      keptGroup.append(...groupedBlocks);
      block = keptGroup.nextElementSibling;
    } else {
      block = block.nextElementSibling;
    }
  }
  return pagedBody.innerHTML;
}
