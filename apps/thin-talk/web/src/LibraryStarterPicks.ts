/** One model the Library recommends to a newcomer. */
export interface LibraryStarterPick {
  readonly huggingfaceID: string;
  readonly approximateSizeBytes: number;
  /** Why this model suits this Mac, in the engine's words; null for the
   * catalog's fixed Get started pick, which carries no reason. */
  readonly reason: string | null;
}

/** The Library's get-started recommendations Astro offers: a chat model and
 * an image model. Either may be missing from a catalog. */
export interface LibraryStarterPicks {
  readonly chatModel: LibraryStarterPick | null;
  readonly imageModel: LibraryStarterPick | null;
}

const CHAT_ROLE = "chat";
const IMAGE_GENERATION_ROLE = "image_generation";
/** The engine's suggestions that answer "which chat model should this Mac
 * start with": the best fit for its memory, or a stronger one than a model it
 * has. The engine words the reason; Astro shows it as sent. */
const FIRST_CHAT_SUGGESTION_KINDS: readonly string[] = ["best_fit", "stronger"];

/**
 * Reads the get-started chat and image models out of a `/v1/library/catalog`
 * document. Astro never chooses a model itself: the chat model is the engine's
 * suggestion for this Mac's memory, the same one the Observatory's Models page
 * shows, so both doors lead to the same download. Before the engine knows the
 * Mac's memory it suggests nothing, and the catalog's fixed Get started chat
 * pick stands in. Returns null when the text is not a catalog.
 */
export function parseLibraryStarterPicks(catalogBodyText: string): LibraryStarterPicks | null {
  let catalogDocument: unknown;
  try {
    catalogDocument = JSON.parse(catalogBodyText);
  } catch {
    return null;
  }
  const rawEntries = typeof catalogDocument === "object" && catalogDocument !== null ? (catalogDocument as Record<string, unknown>)["entries"] : null;
  if (!Array.isArray(rawEntries)) {
    return null;
  }
  const rawSuggestions = (catalogDocument as Record<string, unknown>)["suggestions"];
  return {
    chatModel: suggestedChatPick(rawSuggestions, rawEntries) ?? firstStarterPick(rawEntries, CHAT_ROLE),
    imageModel: firstStarterPick(rawEntries, IMAGE_GENERATION_ROLE),
  };
}

/** Whether the catalog marks this model as installed and ready on this Mac. */
export function isLibraryModelReady(catalogBodyText: string, huggingfaceID: string): boolean {
  let catalogDocument: unknown;
  try {
    catalogDocument = JSON.parse(catalogBodyText);
  } catch {
    return false;
  }
  const rawEntries = typeof catalogDocument === "object" && catalogDocument !== null ? (catalogDocument as Record<string, unknown>)["entries"] : null;
  if (!Array.isArray(rawEntries)) {
    return false;
  }
  return rawEntries.some(
    (rawEntry) =>
      typeof rawEntry === "object" && rawEntry !== null &&
      (rawEntry as Record<string, unknown>)["huggingface_id"] === huggingfaceID &&
      (rawEntry as Record<string, unknown>)["ready_on_this_mac"] === true,
  );
}

/** The first best-fit or stronger suggestion whose model the catalog lists
 * with a size; null when the engine suggested none. */
function suggestedChatPick(rawSuggestions: unknown, rawEntries: readonly unknown[]): LibraryStarterPick | null {
  if (!Array.isArray(rawSuggestions)) {
    return null;
  }
  for (const rawSuggestion of rawSuggestions) {
    if (typeof rawSuggestion !== "object" || rawSuggestion === null) {
      continue;
    }
    const suggestionRecord = rawSuggestion as Record<string, unknown>;
    const huggingfaceID = suggestionRecord["huggingface_id"];
    const reason = suggestionRecord["reason"];
    if (!FIRST_CHAT_SUGGESTION_KINDS.includes(String(suggestionRecord["kind"])) || typeof huggingfaceID !== "string") {
      continue;
    }
    const approximateSizeBytes = catalogSizeBytes(rawEntries, huggingfaceID);
    if (approximateSizeBytes !== null) {
      return { huggingfaceID, approximateSizeBytes, reason: typeof reason === "string" ? reason : null };
    }
  }
  return null;
}

function catalogSizeBytes(rawEntries: readonly unknown[], huggingfaceID: string): number | null {
  for (const rawEntry of rawEntries) {
    if (typeof rawEntry !== "object" || rawEntry === null) {
      continue;
    }
    const entryRecord = rawEntry as Record<string, unknown>;
    if (entryRecord["huggingface_id"] === huggingfaceID && typeof entryRecord["approximate_size_bytes"] === "number") {
      return entryRecord["approximate_size_bytes"];
    }
  }
  return null;
}

function firstStarterPick(rawEntries: readonly unknown[], role: string): LibraryStarterPick | null {
  for (const rawEntry of rawEntries) {
    if (typeof rawEntry !== "object" || rawEntry === null) {
      continue;
    }
    const entryRecord = rawEntry as Record<string, unknown>;
    const huggingfaceID = entryRecord["huggingface_id"];
    const approximateSizeBytes = entryRecord["approximate_size_bytes"];
    if (entryRecord["role"] !== role || entryRecord["get_started"] !== true) {
      continue;
    }
    if (typeof huggingfaceID !== "string" || typeof approximateSizeBytes !== "number") {
      continue;
    }
    return { huggingfaceID, approximateSizeBytes, reason: null };
  }
  return null;
}
