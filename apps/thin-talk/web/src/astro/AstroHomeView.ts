import { makeButton, makeElement } from "./AstroDom";
import { AstroFigure } from "./AstroFigure";
import type { AstroLibrary } from "./AstroLibrary";
import { AstroMoment, astroLine, homeIdeas } from "./AstroLines";
import type { AstroProfile } from "./AstroProfile";
import { AstroStyle } from "./AstroStyle";
import type { AstroTrail } from "./AstroTrail";
import { SketchbookOffer } from "./SketchbookOffer";
import { SkillCardKind, skillCard, starterPromptFor } from "./SkillCard";
import { buildSkillShelf } from "./SkillShelf";
import { WakeUpPanel } from "./WakeUpPanel";

/** Prefix for home's control ids, distinct from the first flight's "astro". */
const ID_PREFIX = "astro-home";

/**
 * Astro's home: what an empty conversation looks like once Astro is shown.
 * Astro greets the person by name, offers a few ideas and the skill cards, and
 * every one of them only fills the message field, so the person always sends.
 *
 * It lives inside Thin Talk's empty-state section, whose visibility the
 * transcript already manages; the classic title and subtitle are hidden while
 * Astro is shown and come back with the Focus switch.
 */
export class AstroHomeView {
  private readonly emptyState: HTMLElement;
  private readonly document: Document;
  private readonly library: AstroLibrary;
  private readonly onStarterPrompt: (prompt: string) => void;
  private readonly onTrailChosen: (trail: AstroTrail) => void;
  private readonly home: HTMLElement;
  private readonly figure: AstroFigure;
  private readonly greeting: HTMLElement;
  private readonly prompt: HTMLElement;
  private readonly ideas: HTMLElement;
  private readonly followUp: HTMLElement;
  private profile: AstroProfile | null = null;
  /** The download panel in the follow-up area, stopped when it is replaced. */
  private followUpPanel: { dispose(): void } | null = null;

  public constructor(init: {
    emptyState: HTMLElement;
    library: AstroLibrary;
    onStarterPrompt: (prompt: string) => void;
    onTrailChosen: (trail: AstroTrail) => void;
  }) {
    this.emptyState = init.emptyState;
    this.document = init.emptyState.ownerDocument;
    this.library = init.library;
    this.onStarterPrompt = init.onStarterPrompt;
    this.onTrailChosen = init.onTrailChosen;
    this.home = makeElement(this.document, "div", "astro-home");
    this.home.dataset.testid = "astro-home";
    this.home.hidden = true;
    this.figure = new AstroFigure(this.document, "astro-home__astro");
    this.greeting = makeElement(this.document, "h1", "astro-home__greeting");
    this.prompt = makeElement(this.document, "p", "astro-home__prompt");
    this.ideas = makeElement(this.document, "div", "astro-home__ideas");
    this.followUp = makeElement(this.document, "div", "astro-home__follow-up");
    this.followUp.setAttribute("role", "status");
    this.home.append(this.header(), this.ideas, this.shelf(), this.followUp);
    this.emptyState.prepend(this.home);
    this.figure.followPointerWithin(this.document.body);
  }

  public show(profile: AstroProfile): void {
    this.profile = profile;
    this.emptyState.classList.add("empty-state--astro");
    this.emptyState.classList.toggle("empty-state--pilot", profile.style === AstroStyle.PILOT);
    this.home.hidden = false;
    this.greeting.textContent = astroLine(AstroMoment.HOME_GREETING, profile.style, profile.nickname);
    this.prompt.textContent = astroLine(AstroMoment.HOME_PROMPT, profile.style, profile.nickname);
    this.replaceFollowUp(null);
    this.renderIdeas(profile.style);
    void this.offerBrainIfMissing(profile);
  }

  public hide(): void {
    this.profile = null;
    this.replaceFollowUp(null);
    this.emptyState.classList.remove("empty-state--astro", "empty-state--pilot");
    this.home.hidden = true;
  }

  /** Puts new content in the follow-up area, stopping the panel it replaces.
   * While home offers a brain itself, the classic "add one from the Models page"
   * notice under it is hidden (see astro.css), since it would say the same
   * thing in words a young person would not follow. */
  private replaceFollowUp(panel: { dispose(): void } | null, ...content: HTMLElement[]): void {
    this.followUpPanel?.dispose();
    this.followUpPanel = panel;
    this.followUp.replaceChildren(...content);
    this.emptyState.classList.toggle("empty-state--astro-offers-brain", panel instanceof WakeUpPanel);
  }

  private header(): HTMLElement {
    const header = makeElement(this.document, "div", "astro-home__header");
    const words = makeElement(this.document, "div", "astro-home__words");
    words.append(this.greeting, this.prompt);
    header.append(this.figure.element, words);
    return header;
  }

  private shelf(): HTMLElement {
    return buildSkillShelf(this.document, ID_PREFIX, {
      onChoose: (kind) => void this.chooseSkill(kind),
      isDropTarget: (clientX, clientY) => this.figure.containsPoint(clientX, clientY),
      onHoverTarget: (isOverTarget) => this.home.classList.toggle("astro-home--drop-armed", isOverTarget),
    });
  }

  private renderIdeas(style: AstroStyle): void {
    this.ideas.replaceChildren();
    for (const [ideaPosition, idea] of homeIdeas(style).entries()) {
      const ideaChip = makeButton(this.document, `${ID_PREFIX}-idea-${ideaPosition}`, idea, true);
      ideaChip.classList.add("astro-chip");
      ideaChip.addEventListener("click", () => this.onStarterPrompt(idea));
      this.ideas.append(ideaChip);
    }
  }

  /** Teaches Astro a skill as a card would: the Draw button uses it to offer
   * the sketchbook in a fresh chat when no picture model is installed. */
  public async chooseSkill(kind: SkillCardKind): Promise<void> {
    const profile = this.profile;
    if (profile === null) {
      return;
    }
    const card = skillCard(kind);
    this.figure.hop();
    if (kind === SkillCardKind.PICTURES && !(await this.library.hasImageModel())) {
      this.offerSketchbook(profile);
      return;
    }
    this.replaceFollowUp(null, makeElement(this.document, "p", "astro-home__reply", astroLine(card.reply, profile.style, profile.nickname)));
    const starterPrompt = starterPromptFor(card, profile.style);
    if (card.isAvailable && starterPrompt !== null) {
      this.onStarterPrompt(starterPrompt);
    }
  }

  /** A person who skipped the first flight on a Mac with no chat model still
   * needs a way to wake Astro, so home offers the brain itself. */
  private async offerBrainIfMissing(profile: AstroProfile): Promise<void> {
    if ((await this.library.hasChatModel()) !== false || this.profile !== profile) {
      return;
    }
    const reply = makeElement(this.document, "p", "astro-home__reply", astroLine(AstroMoment.NEEDS_BRAIN, profile.style, profile.nickname));
    const wakeUpPanel = new WakeUpPanel(this.document, ID_PREFIX, this.library, {
      figure: this.figure,
      trailChoice: { trail: profile.trail, onTrailChosen: (trail) => this.onTrailChosen(trail) },
      say: (moment) => {
        reply.textContent = astroLine(moment, profile.style, profile.nickname);
      },
      isShowing: () => !this.home.hidden,
      // The chat window learns of the new brain through AstroLibrary's install
      // notice, which also redraws this home; nothing more to do here.
      onBrainArrived: () => undefined,
      onContinue: () => this.replaceFollowUp(null),
    });
    this.replaceFollowUp(wakeUpPanel, reply, wakeUpPanel.element);
    await wakeUpPanel.loadOffer();
  }

  private offerSketchbook(profile: AstroProfile): void {
    const sketchbookOffer = new SketchbookOffer(this.document, ID_PREFIX, this.library, () => this.replaceFollowUp(null));
    const reply = makeElement(this.document, "p", "astro-home__reply", astroLine(AstroMoment.NEEDS_SKETCHBOOK, profile.style, profile.nickname));
    this.replaceFollowUp(sketchbookOffer, reply, sketchbookOffer.element);
    void sketchbookOffer.loadOffer();
  }
}
