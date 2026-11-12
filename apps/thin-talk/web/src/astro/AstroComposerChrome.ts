import { SkillCardKind, skillCard } from "./SkillCard";

/** The launcher buttons' data-skill values and the card each one stands for. */
const CARD_BY_LAUNCHER_SKILL: Readonly<Record<string, SkillCardKind>> = {
  pictures: SkillCardKind.PICTURES,
  look: SkillCardKind.LOOK,
  read: SkillCardKind.READ,
};

/**
 * Astro's part of the message field: the skill buttons inside it. They fill
 * the field with the same starter questions as the skill cards; nothing is
 * sent until the person presses Send. Astro itself is not drawn here: the
 * field is where people type, and a figure beside it only crowded the words.
 */
export class AstroComposerChrome {
  private readonly launchers: HTMLElement;
  private readonly onStarterPrompt: (prompt: string) => void;
  private readonly hasImageModel: () => Promise<boolean>;
  private readonly onNeedsImageModel: () => void;
  private isShown = false;

  public constructor(init: {
    launchers: HTMLElement;
    onStarterPrompt: (prompt: string) => void;
    /** Whether an image model is installed, asked before Draw fills the field. */
    hasImageModel: () => Promise<boolean>;
    /** Draw was pressed with no image model: offer its download instead. */
    onNeedsImageModel: () => void;
  }) {
    this.launchers = init.launchers;
    this.onStarterPrompt = init.onStarterPrompt;
    this.hasImageModel = init.hasImageModel;
    this.onNeedsImageModel = init.onNeedsImageModel;
    this.launchers.addEventListener("click", (clickEvent) => void this.launch(clickEvent));
  }

  public show(): void {
    this.isShown = true;
    this.launchers.hidden = false;
  }

  public hide(): void {
    this.isShown = false;
    this.launchers.hidden = true;
  }

  /** A drawing request with no image model would only make the chat model
   * say it cannot draw, so Draw offers the image model's download instead. */
  private async launch(clickEvent: Event): Promise<void> {
    const launcher = clickEvent.target instanceof Element ? clickEvent.target.closest<HTMLElement>("[data-skill]") : null;
    const cardKind = launcher !== null ? CARD_BY_LAUNCHER_SKILL[launcher.dataset.skill ?? ""] : undefined;
    if (!this.isShown || cardKind === undefined) {
      return;
    }
    if (cardKind === SkillCardKind.PICTURES && !(await this.hasImageModel())) {
      this.onNeedsImageModel();
      return;
    }
    const starterPrompt = skillCard(cardKind).starterPrompt;
    if (starterPrompt !== null) {
      this.onStarterPrompt(starterPrompt);
    }
  }
}
