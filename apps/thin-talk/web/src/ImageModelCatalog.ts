import { ImageModel } from "./ImageModel";

/** The role the Library gives image generators. */
const IMAGE_GENERATION_ROLE = "image_generation";

/** What the Library says about one model Thin Talk can request. */
export interface LibraryImageEntry {
  readonly requestableModelID: string;
  readonly displayName: string;
  readonly isGetStartedPick: boolean;
}

/**
 * Reads the image generators out of a `/v1/library/catalog` document, or
 * returns null when the text is not a catalog at all. Only entries installed
 * on this Mac carry the id to request, so the rest are skipped; a malformed
 * entry is skipped too, because names are a courtesy and must never stop an
 * image from being made.
 */
export function parseLibraryImageEntries(catalogBodyText: string): LibraryImageEntry[] | null {
  let catalogDocument: unknown;
  try {
    catalogDocument = JSON.parse(catalogBodyText);
  } catch {
    return null;
  }
  const catalogRecord = typeof catalogDocument === "object" && catalogDocument !== null ? (catalogDocument as Record<string, unknown>) : {};
  const rawEntries = catalogRecord["entries"];
  if (!Array.isArray(rawEntries)) {
    return null;
  }
  const libraryImageEntries: LibraryImageEntry[] = [];
  for (const rawEntry of rawEntries) {
    const parsedImageEntry = parseLibraryImageEntry(rawEntry);
    if (parsedImageEntry != null) {
      libraryImageEntries.push(parsedImageEntry);
    }
  }
  return libraryImageEntries;
}

function parseLibraryImageEntry(rawEntry: unknown): LibraryImageEntry | null {
  if (typeof rawEntry !== "object" || rawEntry === null) {
    return null;
  }
  const entryRecord = rawEntry as Record<string, unknown>;
  const requestableModelID = entryRecord["requestable_model_id"];
  const displayName = entryRecord["display_name"];
  if (entryRecord["role"] !== IMAGE_GENERATION_ROLE || typeof requestableModelID !== "string" || typeof displayName !== "string") {
    return null;
  }
  return { requestableModelID, displayName, isGetStartedPick: entryRecord["get_started"] === true };
}

/**
 * The installed image models in the order the picker shows them: the Library's
 * Image Generation pick first, then the rest in the server's order. A model the Library
 * does not list keeps its id as its name.
 */
export function buildImageModels(imageModelIDs: readonly string[], libraryImageEntries: readonly LibraryImageEntry[]): ImageModel[] {
  const libraryEntryByModelID = new Map<string, LibraryImageEntry>();
  for (const libraryImageEntry of libraryImageEntries) {
    libraryEntryByModelID.set(libraryImageEntry.requestableModelID, libraryImageEntry);
  }
  const imageModels = imageModelIDs.map((imageModelID) => {
    const libraryImageEntry = libraryEntryByModelID.get(imageModelID);
    return new ImageModel({
      id: imageModelID,
      name: libraryImageEntry != null ? libraryImageEntry.displayName : imageModelID,
      isImageGenerationPick: libraryImageEntry != null && libraryImageEntry.isGetStartedPick,
    });
  });
  const imageGenerationPicks = imageModels.filter((imageModel) => imageModel.isImageGenerationPick);
  const otherImageModels = imageModels.filter((imageModel) => !imageModel.isImageGenerationPick);
  return [...imageGenerationPicks, ...otherImageModels];
}
