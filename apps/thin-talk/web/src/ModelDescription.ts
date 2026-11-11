/**
 * What the Library says about one installed model beyond its name. A model has
 * one name, the id the engine answers to and the terminal accepts, so this
 * carries only the quieter facts shown under it: "27.1 GB · sees images". The
 * engine's model list decides what is installed; a model the Library does not
 * describe simply shows no facts.
 */
export class ModelDescription {
  public readonly sizeBytes: number | null;
  public readonly seesImages: boolean;

  public constructor(init: { sizeBytes?: number | null; seesImages?: boolean }) {
    this.sizeBytes = init.sizeBytes != null ? init.sizeBytes : null;
    this.seesImages = init.seesImages === true;
  }

  /** The quieter second line of a menu row: "27.1 GB · sees images". */
  public detailLine(): string {
    const detailParts: string[] = [];
    if (this.sizeBytes !== null) {
      detailParts.push(formatGigabytes(this.sizeBytes));
    }
    if (this.seesImages) {
      detailParts.push("sees images");
    }
    return detailParts.join(" · ");
  }
}

/** Gibibytes, labelled GB, to one decimal: the Observatory's Models page
 * prints sizes the same way, so the two never disagree about one model. */
const BYTES_PER_GIGABYTE = 1024 * 1024 * 1024;

export function formatGigabytes(sizeBytes: number): string {
  return `${(sizeBytes / BYTES_PER_GIGABYTE).toFixed(1)} GB`;
}

/**
 * Reads every installed model's description out of a `/v1/library/catalog`
 * document, keyed by the id the engine answers to. Returns null when the text
 * is not a catalog at all. Entries that are not installed carry no requestable
 * id and are skipped, as are malformed ones: these facts are a courtesy and
 * must never stop a chat.
 */
export function parseLibraryModelDescriptions(catalogBodyText: string): Map<string, ModelDescription> | null {
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
  const descriptionByModelID = new Map<string, ModelDescription>();
  for (const rawEntry of rawEntries) {
    const describedModel = parseDescribedModel(rawEntry);
    if (describedModel !== null) {
      descriptionByModelID.set(describedModel.modelID, describedModel.description);
    }
  }
  return descriptionByModelID;
}

function parseDescribedModel(rawEntry: unknown): { modelID: string; description: ModelDescription } | null {
  if (typeof rawEntry !== "object" || rawEntry === null) {
    return null;
  }
  const entryRecord = rawEntry as Record<string, unknown>;
  const requestableModelID = entryRecord["requestable_model_id"];
  if (typeof requestableModelID !== "string" || requestableModelID.length === 0) {
    return null;
  }
  const rawSize = entryRecord["approximate_size_bytes"];
  const capabilities = entryRecord["capabilities"];
  const seesImages = typeof capabilities === "object" && capabilities !== null && (capabilities as Record<string, unknown>)["supports_vision"] === true;
  return {
    modelID: requestableModelID,
    description: new ModelDescription({
      sizeBytes: typeof rawSize === "number" && Number.isFinite(rawSize) && rawSize > 0 ? rawSize : null,
      seesImages,
    }),
  };
}
