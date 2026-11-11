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

/** The choices worth keeping from one export to the next. Someone who hands
 * in Word files without pictures wants that again next week; whose messages
 * to include is not kept, because it follows what is being exported. */
export interface ExportChoices {
  readonly format: ExportFormat;
  readonly includesPictures: boolean;
  readonly includesAiNote: boolean;
}

/** Reads saved choices, or null when they are missing or not ones this
 * version offers, so a damaged or older file falls back to the defaults. */
export function parseExportChoices(plain: unknown): ExportChoices | null {
  if (typeof plain !== "object" || plain === null) {
    return null;
  }
  const record = plain as Record<string, unknown>;
  const format = Object.values(ExportFormat).find((knownFormat) => knownFormat === record["format"]);
  if (format === undefined || typeof record["includesPictures"] !== "boolean" || typeof record["includesAiNote"] !== "boolean") {
    return null;
  }
  return { format, includesPictures: record["includesPictures"], includesAiNote: record["includesAiNote"] };
}

export function defaultExportOptions(scope: ExportScope, rememberedChoices: ExportChoices | null = null): ExportOptions {
  return {
    scope,
    format: rememberedChoices?.format ?? ExportFormat.PDF,
    includesQuestions: scope.kind !== "oneAnswer",
    includesPictures: rememberedChoices?.includesPictures ?? true,
    includesAiNote: rememberedChoices?.includesAiNote ?? true,
  };
}
