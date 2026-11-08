import { ImageModel } from "./ImageModel";

/**
 * Which image model makes the next picture. The composer's picker sets it and
 * the image tool reads it, so the choice reaches the tool without travelling
 * through every layer of a turn.
 *
 * One rule decides: the person's saved choice while it is installed, otherwise
 * the Library's Pictures pick, otherwise the first installed image model. The
 * order of the folders a model was downloaded into never decides.
 */
export class ImageModelSelection {
  private preferredImageModelID: string | null;

  public constructor(preferredImageModelID: string | null = null) {
    this.preferredImageModelID = preferredImageModelID;
  }

  public get preferredID(): string | null {
    return this.preferredImageModelID;
  }

  public prefer(imageModelID: string | null): void {
    this.preferredImageModelID = imageModelID;
  }

  /** The model to use from `installedImageModels`, or null when none is installed. */
  public resolve(installedImageModels: readonly ImageModel[]): ImageModel | null {
    const preferredImageModel = installedImageModels.find((imageModel) => imageModel.id === this.preferredImageModelID);
    if (preferredImageModel != null) {
      return preferredImageModel;
    }
    const picturesPick = installedImageModels.find((imageModel) => imageModel.isPicturesPick);
    if (picturesPick != null) {
      return picturesPick;
    }
    const firstImageModel: ImageModel | undefined = installedImageModels[0];
    return firstImageModel != null ? firstImageModel : null;
  }
}
