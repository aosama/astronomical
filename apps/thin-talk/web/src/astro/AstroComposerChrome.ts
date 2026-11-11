import { astroConstellationHtml } from "./AstroConstellation";
import type { AstroStyle } from "./AstroStyle";
import { SkillCardKind, skillCard, starterPromptFor } from "./SkillCard";

/** The launcher buttons' data-skill values and the card each one stands for. */
const CARD_BY_LAUNCHER_SKILL: Readonly<Record<string, SkillCardKind>> = {
  pictures: SkillCardKind.PICTURES,
  look: SkillCardKind.LOOK,
  read: SkillCardKind.READ,
};

/** Set on the small Astro while the field holds something to send. */
const READY_TO_SEND_CLASS = "composer__astro--ready";

/**
 * Astro's part of the message field: the skill buttons inside it and the small
 * Astro beside it, which brightens once there is something to send. The buttons fill the field with the same starter
 * questions as the skill cards; nothing is sent until the person presses Send.
 */
export class AstroComposerChrome {
  private readonly launchers: HTMLElement;
  private readonly composerAstro: HTMLElement;
  private readonly composerField: HTMLTextAreaElement;
  private readonly onStarterPrompt: (prompt: string) => void;
  private readonly hasImageModel: () => Promise<boolean>;
  private readonly onNeedsSketchbook: () => void;
  private style: AstroStyle | null = null;

  public constructor(init: {
    launchers: HTMLElement;
    composerAstro: HTMLElement;
    composerField: HTMLTextAreaElement;
    onStarterPrompt: (prompt: string) => void;
    /** Whether a picture model is installed, asked before Draw fills the field. */
    hasImageModel: () => Promise<boolean>;
    /** Draw was pressed with no picture model: offer the sketchbook instead. */
    onNeedsSketchbook: () => void;
  }) {
    this.launchers = init.launchers;
    this.composerAstro = init.composerAstro;
    this.composerAstro.innerHTML = astroConstellationHtml();
    this.composerField = init.composerField;
    this.onStarterPrompt = init.onStarterPrompt;
    this.hasImageModel = init.hasImageModel;
    this.onNeedsSketchbook = init.onNeedsSketchbook;
    this.launchers.addEventListener("click", (clickEvent) => void this.launch(clickEvent));
    init.composerField.addEventListener("input", () => this.reflectDraft());
  }

  public show(style: AstroStyle): void {
    this.style = style;
    this.launchers.hidden = false;
    this.reflectDraft();
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

  private reflectDraft(): void {
    this.composerAstro.classList.toggle(READY_TO_SEND_CLASS, this.composerField.value.trim().length > 0);
  }
}
