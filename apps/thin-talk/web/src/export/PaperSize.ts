/**
 * The sheet of paper an export is laid out for.
 *
 * Most of the world prints on A4 and North America on Letter, and the two
 * differ enough to move every page break: A4 is a quarter of an inch narrower
 * and seven tenths of an inch taller. The host says which one the Mac is set
 * to, and the preview, the PDF and the Word file all follow it, so the page
 * the sheet shows is the page that prints.
 */
export interface PaperSize {
  readonly widthInches: number;
  readonly heightInches: number;
}

export const LETTER_PAPER: PaperSize = { widthInches: 8.5, heightInches: 11 };
export const A4_PAPER: PaperSize = { widthInches: 8.27, heightInches: 11.69 };

/** The margin on every side of the page, which the host's print setup and
 * the Word file's page setup both use. */
export const PRINTED_MARGIN_INCHES = 1;

const POINTS_PER_INCH = 72;
/** Sizes outside this range are not paper anyone writes a report on (a
 * label, a banner), and a preview drawn for them would be unreadable. */
const SMALLEST_PAPER_SIDE_INCHES = 5;
const LARGEST_PAPER_SIDE_INCHES = 20;

function validSideInches(sidePoints: unknown): number | null {
  if (typeof sidePoints !== "number" || !Number.isFinite(sidePoints)) {
    return null;
  }
  const sideInches = sidePoints / POINTS_PER_INCH;
  return sideInches >= SMALLEST_PAPER_SIDE_INCHES && sideInches <= LARGEST_PAPER_SIDE_INCHES ? sideInches : null;
}

/** The paper the host named, in points as macOS measures paper, or Letter
 * when the host named none or something that is not report paper. */
export function paperSizeFromHost(candidate: unknown): PaperSize {
  if (typeof candidate !== "object" || candidate === null) {
    return LETTER_PAPER;
  }
  const paperRecord = candidate as Record<string, unknown>;
  const widthInches = validSideInches(paperRecord["widthPoints"]);
  const heightInches = validSideInches(paperRecord["heightPoints"]);
  return widthInches !== null && heightInches !== null ? { widthInches, heightInches } : LETTER_PAPER;
}

/** The width of the text column: the paper less its two side margins. */
export function textWidthInches(paper: PaperSize): number {
  return paper.widthInches - 2 * PRINTED_MARGIN_INCHES;
}

/** The height of the text on one page: the paper less top and bottom margins. */
export function textHeightInches(paper: PaperSize): number {
  return paper.heightInches - 2 * PRINTED_MARGIN_INCHES;
}
