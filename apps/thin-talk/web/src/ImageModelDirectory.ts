import { DaemonClient } from "./DaemonClient";
import { ImageModel } from "./ImageModel";
import { buildImageModels, parseLibraryImageEntries } from "./ImageModelCatalog";
import type { LibraryImageEntry } from "./ImageModelCatalog";

/**
 * Lists the installed image models, named and ordered as the Library presents
 * them. The model list decides what is installed; the Library catalog only
 * adds names and the Image Generation pick.
 *
 * The last Library answer that could be read is kept, so a slow or failed
 * catalog read changes neither the names nor which model the selection rule
 * picks: without it, a timeout would quietly swap the Image Generation pick for the
 * first installed model.
 */
export class ImageModelDirectory {
  private readonly client: DaemonClient;
  private lastReadLibraryImageEntries: readonly LibraryImageEntry[] = [];

  public constructor(client: DaemonClient) {
    this.client = client;
  }

  /** The installed image models. Throws only when the model list itself
   * cannot be read, or when `signal` stops the read. */
  public async load(signal?: AbortSignal): Promise<ImageModel[]> {
    const imageModelIDs = await this.client.fetchImageModelIDs(signal);
    if (imageModelIDs.length === 0) {
      return [];
    }
    return buildImageModels(imageModelIDs, await this.libraryImageEntries(signal));
  }

  private async libraryImageEntries(signal?: AbortSignal): Promise<readonly LibraryImageEntry[]> {
    let catalogBodyText: string;
    try {
      catalogBodyText = await this.client.fetchLibraryCatalogText(signal);
    } catch (catalogFailure) {
      if (signal?.aborted === true) {
        throw catalogFailure;
      }
      return this.lastReadLibraryImageEntries;
    }
    const parsedImageEntries = parseLibraryImageEntries(catalogBodyText);
    if (parsedImageEntries != null) {
      this.lastReadLibraryImageEntries = parsedImageEntries;
    }
    return this.lastReadLibraryImageEntries;
  }
}
