/** The three kinds of file a conversation can be exported as. The values are
 * the file extensions, which the host also reads to choose how to write. */
export enum ExportFormat {
  PDF = "pdf",
  WORD = "docx",
  WEB_PAGE = "html",
}

/** How one format is offered in the export sheet. The hint says what the file
 * is good for, since "PDF" and "Word" mean little to a ten-year-old and the
 * choice is really between handing in, editing, and opening anywhere. */
export interface ExportFormatOffer {
  readonly format: ExportFormat;
  readonly label: string;
  readonly hint: string;
  readonly saveLabel: string;
}

export const EXPORT_FORMAT_OFFERS: readonly ExportFormatOffer[] = [
  { format: ExportFormat.PDF, label: "PDF", hint: "Hand in or print", saveLabel: "Save as PDF…" },
  { format: ExportFormat.WORD, label: "Word", hint: "Keep editing", saveLabel: "Save as Word…" },
  { format: ExportFormat.WEB_PAGE, label: "Web page", hint: "Open anywhere", saveLabel: "Save as web page…" },
];

export function exportFormatOffer(format: ExportFormat): ExportFormatOffer {
  const matchingOffer = EXPORT_FORMAT_OFFERS.find((offer) => offer.format === format);
  if (matchingOffer === undefined) {
    throw new Error(`no export offer for format "${format}"`);
  }
  return matchingOffer;
}
