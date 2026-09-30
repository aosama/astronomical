/**
 * One decoded raw model entry from `/v1/models`. The supervisor publishes input
 * modalities and the endpoints each model supports, so a chat app can keep only
 * the models that can receive chat requests.
 */
export class ModelCatalogEntry {
  public readonly id: string;
  public readonly inputModalities: readonly string[];
  public readonly supportedEndpoints: readonly string[] | null;

  public constructor(init: {
    id: string;
    inputModalities: readonly string[];
    supportedEndpoints: readonly string[] | null;
  }) {
    this.id = init.id;
    this.inputModalities = init.inputModalities;
    this.supportedEndpoints = init.supportedEndpoints;
  }

  public get supportsChat(): boolean {
    const keepsText = this.inputModalities.length === 0 || this.inputModalities.includes("text");
    const keepsChat = this.supportedEndpoints === null || this.supportedEndpoints.some((endpoint) => endpoint.includes("chat"));
    return keepsText && keepsChat;
  }

  public static fromPlain(plain: unknown): ModelCatalogEntry | null {
    if (typeof plain !== "object" || plain === null) {
      return null;
    }
    const record = plain as Record<string, unknown>;
    if (typeof record["id"] !== "string") {
      return null;
    }
    const inputModalities = Array.isArray(record["input_modalities"])
      ? record["input_modalities"].filter((modality): modality is string => typeof modality === "string")
      : [];
    const supportedEndpoints = Array.isArray(record["supported_endpoints"])
      ? record["supported_endpoints"].filter((endpoint): endpoint is string => typeof endpoint === "string")
      : null;
    return new ModelCatalogEntry({ id: record["id"], inputModalities, supportedEndpoints });
  }
}
