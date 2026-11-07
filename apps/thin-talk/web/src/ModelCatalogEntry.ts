/**
 * One decoded raw model entry from `/v1/models`. The daemon publishes input
 * modalities, the endpoints each model supports and whether it can call tools,
 * so a chat app can keep only the models that can receive chat requests and
 * offer tools only to models that can use them.
 */
export class ModelCatalogEntry {
  public readonly id: string;
  public readonly inputModalities: readonly string[];
  public readonly supportedEndpoints: readonly string[] | null;
  public readonly supportsToolCalls: boolean;
  /** Whether the model makes images (an image-generation model). */
  public readonly generatesImages: boolean;

  public constructor(init: {
    id: string;
    inputModalities: readonly string[];
    supportedEndpoints: readonly string[] | null;
    supportsToolCalls?: boolean;
    generatesImages?: boolean;
  }) {
    this.id = init.id;
    this.inputModalities = init.inputModalities;
    this.supportedEndpoints = init.supportedEndpoints;
    this.supportsToolCalls = init.supportsToolCalls === true;
    this.generatesImages = init.generatesImages === true;
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
    // Only an explicit `true` counts: a model that does not say it can call
    // tools is offered none, so it never sees declarations it cannot follow.
    const supportsToolCalls = record["supports_tool_calls"] === true;
    const generatesImages = supportedEndpoints !== null && supportedEndpoints.includes("/v1/images/generations");
    return new ModelCatalogEntry({ id: record["id"], inputModalities, supportedEndpoints, supportsToolCalls, generatesImages });
  }
}
