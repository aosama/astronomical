/** One model the Library recommends to a newcomer. */
export interface LibraryStarterPick {
  readonly huggingfaceID: string;
  readonly approximateSizeBytes: number;
}

/** The Library's get-started recommendations Astro offers: a brain to chat
 * with and a sketchbook to draw with. Either may be missing from a catalog. */
export interface LibraryStarterPicks {
  readonly brain: LibraryStarterPick | null;
  readonly sketchbook: LibraryStarterPick | null;
}

const CHAT_ROLE = "chat";
const IMAGE_GENERATION_ROLE = "image_generation";

/**
 * Reads the get-started chat and image models out of a `/v1/library/catalog`
 * document. Astro never chooses a model itself: the Library's own pick is the
 * one the Observatory recommends, so both doors lead to the same download.
 * Returns null when the text is not a catalog.
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
  return {
    brain: firstStarterPick(rawEntries, CHAT_ROLE),
    sketchbook: firstStarterPick(rawEntries, IMAGE_GENERATION_ROLE),
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
    return { huggingfaceID, approximateSizeBytes };
  }
  return null;
}
