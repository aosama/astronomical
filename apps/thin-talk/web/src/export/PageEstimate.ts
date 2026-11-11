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

/**
 * Where an article's uncut blocks sit in its text at full size. Read from
 * drawn rectangles, which the preview's transform has scaled, so each is
 * divided by the scale the article was drawn at. `textInsetPixels` is the
 * page margin above the text. A block inside another uncut block moves with
 * it and is left out.
 */
export function wholeBlocksIn(article: HTMLElement, wholeBlockSelector: string, textInsetPixels: number): WholeBlock[] {
  const articleRectangle = article.getBoundingClientRect();
  const drawnScale = article.offsetWidth > 0 ? articleRectangle.width / article.offsetWidth : 0;
  if (drawnScale <= 0) {
    return [];
  }
  const wholeBlocks: WholeBlock[] = [];
  for (const blockElement of article.querySelectorAll<HTMLElement>(wholeBlockSelector)) {
    if (blockElement.parentElement?.closest(wholeBlockSelector) != null) {
      continue;
    }
    const blockRectangle = blockElement.getBoundingClientRect();
    wholeBlocks.push({
      topPixels: (blockRectangle.top - articleRectangle.top) / drawnScale - textInsetPixels,
      heightPixels: blockRectangle.height / drawnScale,
    });
  }
  return wholeBlocks;
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
