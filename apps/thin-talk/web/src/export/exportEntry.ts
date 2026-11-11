import type { ExportEnvironment, ExportFeature } from "./ExportEnvironment";
import type { ExportScope } from "./ExportOptions";
import { ExportSheet } from "./ExportSheet";

/**
 * The entry of the export script, which the chat page loads the first time
 * someone exports. It carries the Word library, about as large as the whole
 * chat client, so keeping it out of the page's own script keeps every launch
 * as quick as before for the many people who never export.
 */

/** The feature the script registers: one sheet per page, built on first use. */
export function createExportFeature(): ExportFeature {
  let openSheet: ExportSheet | null = null;
  return {
    open(environment: ExportEnvironment, scope: ExportScope): void {
      openSheet ??= new ExportSheet(environment);
      openSheet.open(scope);
    },
  };
}

window.__thintalkExport = createExportFeature();
