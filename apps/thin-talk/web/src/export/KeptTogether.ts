/**
 * Keeps on one printed page the things a reader expects together: a heading
 * and what it introduces, a sentence ending in a colon and the formula, list
 * or table it leads into, and a short list as a whole.
 *
 * The stylesheet already asks for the first with `break-after: avoid`, but
 * the WebKit that prints the PDF does not follow that hint: a heading was
 * left alone at the foot of a page with its diagram on the next, and a
 * formula was printed on the page after "…packed into a given volume:".
 * WebKit does follow `break-inside: avoid`, so each such run of blocks is
 * put in one group that carries that rule instead.
 */

/** The group whose blocks are printed on one page. */
export const KEPT_TOGETHER_CLASS = "export-keep";

const HEADING_TAGS = new Set(["H1", "H2", "H3", "H4"]);
const LIST_TAGS = new Set(["UL", "OL"]);
/** A list this short is a checklist or a few steps, and losing its last
 * item to the next page reads as a mistake. A longer one can run for most
 * of a page: kept whole it would move to the next page and leave the first
 * half empty, which looks worse than a list that continues overleaf. */
const SHORT_LIST_MAXIMUM_ITEMS = 6;
const SHORT_LIST_MAXIMUM_CHARACTERS = 600;

function isHeading(block: Element): boolean {
  return HEADING_TAGS.has(block.tagName);
}

function isLongList(block: Element): boolean {
  if (!LIST_TAGS.has(block.tagName)) {
    return false;
  }
  const itemCount = block.querySelectorAll("li").length;
  return itemCount > SHORT_LIST_MAXIMUM_ITEMS || (block.textContent ?? "").length > SHORT_LIST_MAXIMUM_CHARACTERS;
}

function isShortList(block: Element): boolean {
  return LIST_TAGS.has(block.tagName) && !isLongList(block);
}

/** A paragraph that leads into the next block: "The formula is:". */
function isLeadIn(block: Element): boolean {
  return block.tagName === "P" && /[:：]\s*$/.test(block.textContent ?? "");
}

/** The block after `introducer` when it can be held to it: anything but a
 * long list or a further heading. */
function introducedBlock(introducer: Element): Element | null {
  const followingBlock = introducer.nextElementSibling;
  return followingBlock !== null && !isLongList(followingBlock) && !isHeading(followingBlock) ? followingBlock : null;
}

/**
 * The blocks that belong on one page starting at `firstBlock`: headings that
 * follow one another, the block they introduce, and, when that block (or
 * `firstBlock` itself) is a lead-in sentence, the block it leads into. One
 * block alone when nothing is held to it.
 */
function groupStartingAt(firstBlock: Element): Element[] {
  const groupedBlocks: Element[] = [firstBlock];
  let lastBlock = firstBlock;
  while (isHeading(lastBlock) && lastBlock.nextElementSibling !== null && isHeading(lastBlock.nextElementSibling)) {
    lastBlock = lastBlock.nextElementSibling;
    groupedBlocks.push(lastBlock);
  }
  if (isHeading(lastBlock)) {
    const blockUnderHeading = introducedBlock(lastBlock);
    if (blockUnderHeading === null) {
      return groupedBlocks;
    }
    lastBlock = blockUnderHeading;
    groupedBlocks.push(lastBlock);
  }
  if (isLeadIn(lastBlock)) {
    const blockLedInto = introducedBlock(lastBlock);
    if (blockLedInto !== null) {
      groupedBlocks.push(blockLedInto);
    }
  }
  return groupedBlocks;
}

function startsAGroup(block: Element): boolean {
  return isHeading(block) || isLeadIn(block);
}

/**
 * The body's markup with each run of blocks that belongs on one page put in
 * a group, and each short list on its own marked to stay whole. The body
 * itself is left as it was: the Word writer reads the same body and keeps
 * headings with their text through Word's own heading styles.
 */
export function markupKeepingBlocksTogether(body: HTMLElement): string {
  const pagedBody = body.cloneNode(true) as HTMLElement;
  let block = pagedBody.firstElementChild;
  while (block !== null) {
    const groupedBlocks = startsAGroup(block) ? groupStartingAt(block) : [block];
    if (groupedBlocks.length > 1) {
      const keptGroup = pagedBody.ownerDocument.createElement("div");
      keptGroup.className = KEPT_TOGETHER_CLASS;
      block.before(keptGroup);
      keptGroup.append(...groupedBlocks);
      block = keptGroup.nextElementSibling;
      continue;
    }
    if (isShortList(block)) {
      block.classList.add(KEPT_TOGETHER_CLASS);
    }
    block = block.nextElementSibling;
  }
  return pagedBody.innerHTML;
}
