import type { DaemonClient } from "./DaemonClient";
import { ModelDescription, parseLibraryModelDescriptions } from "./ModelDescription";

/**
 * Keeps the Library's facts (size, whether it sees images) for the installed
 * models, which are named by their ids. The last catalog that could be read is
 * kept, so a slow or failed read never drops a model's facts halfway through a
 * conversation. Whoever draws them is told after each successful read.
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
