import type { DaemonClient } from "./DaemonClient";
import { ModelDescription, parseLibraryModelDescriptions } from "./ModelDescription";

/**
 * Keeps the Library's plain names for the installed models. The last catalog
 * that could be read is kept, so a slow or failed read never swaps a model's
 * plain name back to its folder name halfway through a conversation. Whoever
 * draws the names is told after each successful read.
 */
export class ModelDescriptionDirectory {
  private readonly client: DaemonClient;
  private readonly onDescriptionsRead: () => void;
  private lastReadDescriptions: ReadonlyMap<string, ModelDescription> = new Map();

  public constructor(client: DaemonClient, onDescriptionsRead: () => void) {
    this.client = client;
    this.onDescriptionsRead = onDescriptionsRead;
  }

  public get descriptions(): ReadonlyMap<string, ModelDescription> {
    return this.lastReadDescriptions;
  }

  /** Reads the catalog again. Never throws: without it the ids are shown. */
  public async refresh(): Promise<void> {
    let catalogBodyText: string;
    try {
      catalogBodyText = await this.client.fetchLibraryCatalogText();
    } catch {
      return;
    }
    const parsedDescriptions = parseLibraryModelDescriptions(catalogBodyText);
    if (parsedDescriptions !== null) {
      this.lastReadDescriptions = parsedDescriptions;
      this.onDescriptionsRead();
    }
  }
}
