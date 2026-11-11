import { ExportFormat } from "./ExportFormat";

/** Which part of the conversation goes into the file. */
export type ExportScope =
  /** Every finished message. */
  | { readonly kind: "wholeChat" }
  /** One answer on its own, without the question that led to it. */
  | { readonly kind: "oneAnswer"; readonly messageID: string }
  /** The messages the person ticked, in conversation order. */
  | { readonly kind: "pickedMessages"; readonly messageIDs: readonly string[] };

/** Everything the export sheet lets a person choose. */
export interface ExportOptions {
  readonly scope: ExportScope;
  readonly format: ExportFormat;
  /** Their own messages. Off turns a chat into the answers alone, which is
   * what most homework wants. */
  readonly includesQuestions: boolean;
  /** Generated pictures. */
  readonly includesPictures: boolean;
  /** One line at the end saying an AI helped. On by default: schools
   * increasingly ask for it, and a default that hides it would make the app
   * the thing that got a student into trouble. */
  readonly includesAiNote: boolean;
}

export function defaultExportOptions(scope: ExportScope): ExportOptions {
  return {
    scope,
    format: ExportFormat.PDF,
    includesQuestions: scope.kind !== "oneAnswer",
    includesPictures: true,
    includesAiNote: true,
  };
}
