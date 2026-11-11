import { standaloneExportPage } from "./ExportArticle";
import type { ExportContent } from "./ExportContentBuilder";
import { suggestedExportName } from "./ExportContentBuilder";
import type { ExportSaveRequest } from "./ExportEnvironment";
import { ExportFormat } from "./ExportFormat";
import type { DiagramRasteriser } from "./DiagramRasteriser";
import { writeWordDocument } from "./WordDocumentWriter";

/** Turns finished content into the file the host saves. The web page and the
 * PDF share one standalone page; only Word needs a file built here. */
export async function exportSaveRequest(content: ExportContent, format: ExportFormat, rasteriseDiagram: DiagramRasteriser): Promise<ExportSaveRequest> {
  const suggestedName = suggestedExportName(content.title);
  if (format === ExportFormat.WORD) {
    return { format, suggestedName, html: null, base64: await writeWordDocument(content, rasteriseDiagram) };
  }
  return { format, suggestedName, html: standaloneExportPage(content), base64: null };
}
