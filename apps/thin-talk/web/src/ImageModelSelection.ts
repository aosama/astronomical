import { ImageModel } from "./ImageModel";

/**
 * Which image model makes the next picture, over the image models installed
 * as last listed. The controller sets the person's choice and the image tool
 * refreshes the installed list each time it draws, so the composer's pill and
 * the tool always read the same answer; listeners hear every change.
 *
 * One rule decides: the person's saved choice while it is installed, otherwise
 * the Models page Image Generation pick, otherwise the first installed image model. The
 * order of the folders a model was downloaded into never decides.
 */
export class ImageModelSelection {
  private preferredImageModelID: string | null;
  private installedImageModels: readonly ImageModel[] = [];
  private readonly changeListeners = new Set<() => void>();

  public constructor(preferredImageModelID: string | null = null) {
    this.preferredImageModelID = preferredImageModelID;
  }

  public get installed(): readonly ImageModel[] {
    return this.installedImageModels;
  }

  public prefer(imageModelID: string | null): void {
    this.preferredImageModelID = imageModelID;
    this.notifyChange();
  }

  public updateInstalled(installedImageModels: readonly ImageModel[]): void {
    this.installedImageModels = installedImageModels;
    this.notifyChange();
  }

  /** Calls `listener` after each change; returns the call that stops it. */
  public onChange(listener: () => void): () => void {
    this.changeListeners.add(listener);
    return () => {
      this.changeListeners.delete(listener);
    };
  }

  /** The model the next picture is drawn with, or null when none is installed. */
  public current(): ImageModel | null {
    const preferredImageModel = this.installedImageModels.find((imageModel) => imageModel.id === this.preferredImageModelID);
    if (preferredImageModel != null) {
      return preferredImageModel;
    }
    const imageGenerationPick = this.installedImageModels.find((imageModel) => imageModel.isImageGenerationPick);
    if (imageGenerationPick != null) {
      return imageGenerationPick;
    }
    const firstImageModel: ImageModel | undefined = this.installedImageModels[0];
    return firstImageModel != null ? firstImageModel : null;
  }

  private notifyChange(): void {
    for (const changeListener of this.changeListeners) {
      changeListener();
    }
  }
}
