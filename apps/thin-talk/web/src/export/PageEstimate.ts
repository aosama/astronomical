/**
 * How tall the document's text becomes once it is cut into pages.
 *
 * Dividing the text's height by a page's height undercounts: a diagram, a
 * table or a heading with its block that would straddle a page boundary is
 * moved whole to the next page, and the gap it leaves is paper too. A chat
 * whose text was just under two pages tall printed as three, and the sheet
 * said "About 2 pages".
 */

/** A block the printed page never cuts, measured from the top of the text. */
export interface WholeBlock {
  readonly topPixels: number;
  readonly heightPixels: number;
}

function gapLeftAbove(wholeBlock: WholeBlock, pushedDownPixels: number, pageHeightPixels: number): number {
  const pagedTopPixels = wholeBlock.topPixels + pushedDownPixels;
  const pageStartPixels = Math.floor(pagedTopPixels / pageHeightPixels) * pageHeightPixels;
  const nextPageStartPixels = pageStartPixels + pageHeightPixels;
  const crossesPageEnd = pagedTopPixels + wholeBlock.heightPixels > nextPageStartPixels;
  // A block taller than a page is cut wherever it starts, and one already at
  // the top of a page has nowhere better to go.
  const canMoveWhole = wholeBlock.heightPixels <= pageHeightPixels && pagedTopPixels > pageStartPixels;
  return crossesPageEnd && canMoveWhole ? nextPageStartPixels - pagedTopPixels : 0;
}

/** The text's height with the gaps that whole blocks leave when they move to
 * the next page. Blocks may arrive in any order. */
export function pagedTextHeightPixels(flowingHeightPixels: number, wholeBlocks: readonly WholeBlock[], pageHeightPixels: number): number {
  if (pageHeightPixels <= 0) {
    return flowingHeightPixels;
  }
  const blocksFromTheTop = [...wholeBlocks].sort((earlierBlock, laterBlock) => earlierBlock.topPixels - laterBlock.topPixels);
  let pushedDownPixels = 0;
  for (const wholeBlock of blocksFromTheTop) {
    pushedDownPixels += gapLeftAbove(wholeBlock, pushedDownPixels, pageHeightPixels);
  }
  return flowingHeightPixels + pushedDownPixels;
}
