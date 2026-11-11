import { ImageModel } from "./ImageModel";

/** The role the Library gives image generators. */
const IMAGE_GENERATION_ROLE = "image_generation";

/** What the Library says about one model Thin Talk can request. */
export interface LibraryImageEntry {
  readonly requestableModelID: string;
  readonly isGetStartedPick: boolean;
}

/**
 * Reads the image generators out of a `/v1/library/catalog` document, or
 * returns null when the text is not a catalog at all. Only entries installed
 * on this Mac carry the id to request, so the rest are skipped; a malformed
 * entry is skipped too, because the Library's picks are a courtesy and must
 * never stop an image from being made.
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
  if (entryRecord["role"] !== IMAGE_GENERATION_ROLE || typeof requestableModelID !== "string") {
    return null;
  }
  return { requestableModelID, isGetStartedPick: entryRecord["get_started"] === true };
}

/**
 * The installed image models in the order the picker shows them: the Library's
 * Image Generation pick first, then the rest in the server's order.
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
      isImageGenerationPick: libraryImageEntry != null && libraryImageEntry.isGetStartedPick,
    });
  });
  const imageGenerationPicks = imageModels.filter((imageModel) => imageModel.isImageGenerationPick);
  const otherImageModels = imageModels.filter((imageModel) => !imageModel.isImageGenerationPick);
  return [...imageGenerationPicks, ...otherImageModels];
}
