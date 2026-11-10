import { lookFrameURL, LOOK_FRAME_COUNT } from "./AstroFigure";
import type { AstroStyle } from "./AstroStyle";
import { lookFrameForPointer } from "./LookDirection";
import { SkillCardKind, skillCard, starterPromptFor } from "./SkillCard";

/** The launcher buttons' data-skill values and the card each one stands for. */
const CARD_BY_LAUNCHER_SKILL: Readonly<Record<string, SkillCardKind>> = {
  pictures: SkillCardKind.PICTURES,
  look: SkillCardKind.LOOK,
  read: SkillCardKind.READ,
};

/** Facing the send button, at the right of the field. */
const LOOK_AT_SEND_FRAME = LOOK_FRAME_COUNT - 1;
const RESTING_FRAME = Math.floor(LOOK_FRAME_COUNT / 2);

/**
 * Astro's part of the message field: the skill buttons inside it and the small
 * Astro beside it, which watches the pointer and turns toward Send once there
 * is something to send. The buttons fill the field with the same starter
 * questions as the skill cards; nothing is sent until the person presses Send.
 */
export class AstroComposerChrome {
  private readonly launchers: HTMLElement;
  private readonly listeningAstroImage: HTMLImageElement;
  private readonly composerField: HTMLTextAreaElement;
  private readonly onStarterPrompt: (prompt: string) => void;
  private readonly hasImageModel: () => Promise<boolean>;
  private readonly onNeedsSketchbook: () => void;
  private style: AstroStyle | null = null;

  public constructor(init: {
    launchers: HTMLElement;
    listeningAstroImage: HTMLImageElement;
    composerField: HTMLTextAreaElement;
    watchedArea: HTMLElement;
    onStarterPrompt: (prompt: string) => void;
    /** Whether a picture model is installed, asked before Draw fills the field. */
    hasImageModel: () => Promise<boolean>;
    /** Draw was pressed with no picture model: offer the sketchbook instead. */
    onNeedsSketchbook: () => void;
  }) {
    this.launchers = init.launchers;
    this.listeningAstroImage = init.listeningAstroImage;
    this.composerField = init.composerField;
    this.onStarterPrompt = init.onStarterPrompt;
    this.hasImageModel = init.hasImageModel;
    this.onNeedsSketchbook = init.onNeedsSketchbook;
    this.launchers.addEventListener("click", (clickEvent) => void this.launch(clickEvent));
    init.composerField.addEventListener("input", () => this.lookAtDraft());
    init.watchedArea.addEventListener("pointermove", (pointerEvent) => this.watchPointer(init.watchedArea, pointerEvent));
    init.watchedArea.addEventListener("pointerleave", () => this.lookAtDraft());
  }

  public show(style: AstroStyle): void {
    this.style = style;
    this.launchers.hidden = false;
    this.lookAtDraft();
  }

  public hide(): void {
    this.style = null;
    this.launchers.hidden = true;
  }

  /** A drawing request with no picture model would only make the model say it
   * cannot draw, so Draw offers the sketchbook download instead. */
  private async launch(clickEvent: Event): Promise<void> {
    const style = this.style;
    const launcher = clickEvent.target instanceof Element ? clickEvent.target.closest<HTMLElement>("[data-skill]") : null;
    const cardKind = launcher !== null ? CARD_BY_LAUNCHER_SKILL[launcher.dataset.skill ?? ""] : undefined;
    if (style === null || cardKind === undefined) {
      return;
    }
    if (cardKind === SkillCardKind.PICTURES && !(await this.hasImageModel())) {
      this.onNeedsSketchbook();
      return;
    }
    const starterPrompt = starterPromptFor(skillCard(cardKind), style);
    if (starterPrompt !== null) {
      this.onStarterPrompt(starterPrompt);
    }
  }

  private lookAtDraft(): void {
    this.showFrame(this.composerField.value.trim().length > 0 ? LOOK_AT_SEND_FRAME : RESTING_FRAME);
  }

  private watchPointer(watchedArea: HTMLElement, pointerEvent: PointerEvent): void {
    // While Astro is hidden nobody sees this image, so there is nothing to turn.
    if (this.style === null) {
      return;
    }
    const areaBounds = watchedArea.getBoundingClientRect();
    if (areaBounds.width > 0) {
      this.showFrame(lookFrameForPointer((pointerEvent.clientX - areaBounds.left) / areaBounds.width, LOOK_FRAME_COUNT));
    }
  }

  private showFrame(lookFrame: number): void {
    const frameURL = lookFrameURL(lookFrame);
    if (this.listeningAstroImage.getAttribute("src") !== frameURL) {
      this.listeningAstroImage.src = frameURL;
    }
  }
}
