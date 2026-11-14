import { makeButton, makeElement } from "./AstroDom";
import type { AstroLibrary } from "./AstroLibrary";
import { AstroMoment, HOME_IDEAS, astroLine } from "./AstroLines";
import type { HomeIdea } from "./AstroLines";
import type { AstroProfile } from "./AstroProfile";
import { ChatModelDownloadPanel } from "./ChatModelDownloadPanel";
import { ImageModelOffer } from "./ImageModelOffer";
import { SkillCardKind, skillCard } from "./SkillCard";
import { buildSkillShelf } from "./SkillShelf";

/** Prefix for home's control ids, distinct from the get-started journey's "astro". */
const ID_PREFIX = "astro-home";

/**
 * Astro's home: what an empty conversation looks like once Astro is shown.
 * It greets the person, offers a few ideas and the skill cards, and every
 * one of them only fills the message field, so the person always sends.
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
  private readonly home: HTMLElement;
  private readonly greeting: HTMLElement;
  private readonly prompt: HTMLElement;
  private readonly ideas: HTMLElement;
  private readonly followUp: HTMLElement;
  private profile: AstroProfile | null = null;
  /** The download panel in the follow-up area, stopped when it is replaced. */
  private followUpPanel: { dispose(): void } | null = null;

  public constructor(init: { emptyState: HTMLElement; library: AstroLibrary; onStarterPrompt: (prompt: string) => void }) {
    this.emptyState = init.emptyState;
    this.document = init.emptyState.ownerDocument;
    this.library = init.library;
    this.onStarterPrompt = init.onStarterPrompt;
    this.home = makeElement(this.document, "div", "astro-home");
    this.home.dataset.testid = "astro-home";
    this.home.hidden = true;
    this.greeting = makeElement(this.document, "h1", "astro-home__greeting");
    this.prompt = makeElement(this.document, "p", "astro-home__prompt");
    this.ideas = makeElement(this.document, "div", "astro-home__ideas");
    this.followUp = makeElement(this.document, "div", "astro-home__follow-up");
    this.followUp.setAttribute("role", "status");
    this.home.append(this.header(), this.ideas, this.shelf(), this.followUp);
    this.emptyState.prepend(this.home);
    this.renderIdeas();
  }

  public show(profile: AstroProfile): void {
    this.profile = profile;
    this.emptyState.classList.add("empty-state--astro");
    this.home.hidden = false;
    this.greeting.textContent = astroLine(AstroMoment.HOME_GREETING, profile.nickname);
    this.prompt.textContent = astroLine(AstroMoment.HOME_PROMPT, profile.nickname);
    this.replaceFollowUp(null);
    void this.offerChatModelIfMissing(profile);
  }

  public hide(): void {
    this.profile = null;
    this.replaceFollowUp(null);
    this.emptyState.classList.remove("empty-state--astro");
    this.home.hidden = true;
  }

  /** Puts new content in the follow-up area, stopping the panel it replaces.
   * While home offers a chat model itself, the classic "add one from the
   * Models page" notice under it is hidden (see astro.css), since it would
   * say the same thing twice. */
  private replaceFollowUp(panel: { dispose(): void } | null, ...content: HTMLElement[]): void {
    this.followUpPanel?.dispose();
    this.followUpPanel = panel;
    this.followUp.replaceChildren(...content);
    this.emptyState.classList.toggle("empty-state--astro-offers-chat-model", panel instanceof ChatModelDownloadPanel);
  }

  private header(): HTMLElement {
    const header = makeElement(this.document, "div", "astro-home__header");
    const words = makeElement(this.document, "div", "astro-home__words");
    words.append(this.greeting, this.prompt);
    header.append(words);
    return header;
  }

  private shelf(): HTMLElement {
    return buildSkillShelf(this.document, ID_PREFIX, {
      onChoose: (kind) => void this.chooseSkill(kind),
    });
  }

  private renderIdeas(): void {
    for (const [ideaPosition, idea] of HOME_IDEAS.entries()) {
      const ideaChip = makeButton(this.document, `${ID_PREFIX}-idea-${ideaPosition}`, idea.prompt, true);
      ideaChip.classList.add("astro-chip");
      ideaChip.addEventListener("click", () => void this.chooseIdea(idea));
      this.ideas.append(ideaChip);
    }
  }

  /** Fills the message field with the idea, unless it asks for a picture on
   * a Mac with no image model: sent as it is, that prompt ends in a failed
   * drawing, so the chip offers the image model the way the Pictures card
   * does. */
  private async chooseIdea(idea: HomeIdea): Promise<void> {
    const profile = this.profile;
    if (profile !== null && idea.needsImageModel && !(await this.library.hasImageModel())) {
      this.offerImageModel(profile, idea.prompt);
      return;
    }
    this.onStarterPrompt(idea.prompt);
  }

  /** Chooses a skill as its card would: the Draw button uses it to offer the
   * image model in a fresh chat when none is installed. */
  public async chooseSkill(kind: SkillCardKind): Promise<void> {
    const profile = this.profile;
    if (profile === null) {
      return;
    }
    const card = skillCard(kind);
    if (kind === SkillCardKind.PICTURES && !(await this.library.hasImageModel())) {
      this.offerImageModel(profile, card.starterPrompt);
      return;
    }
    this.replaceFollowUp(null, makeElement(this.document, "p", "astro-home__reply", astroLine(card.reply, profile.nickname)));
    if (card.isAvailable && card.starterPrompt !== null) {
      this.onStarterPrompt(card.starterPrompt);
    }
  }

  /** A person who skipped the get-started journey on a Mac with no chat
   * model still needs a way to get one, so home offers the download itself. */
  private async offerChatModelIfMissing(profile: AstroProfile): Promise<void> {
    if ((await this.library.hasChatModel()) !== false || this.profile !== profile) {
      return;
    }
    const reply = makeElement(this.document, "p", "astro-home__reply", astroLine(AstroMoment.NEEDS_CHAT_MODEL, profile.nickname));
    const downloadPanel = new ChatModelDownloadPanel(this.document, ID_PREFIX, this.library, {
      canSkip: false,
      openModelsPage: null,
      say: (moment) => {
        reply.textContent = astroLine(moment, profile.nickname);
      },
      isShowing: () => !this.home.hidden,
      // The chat window learns of the new model through AstroLibrary's install
      // notice, which also redraws this home; nothing more to do here.
      onChatModelArrived: () => undefined,
      onContinue: () => this.replaceFollowUp(null),
    });
    this.replaceFollowUp(downloadPanel, reply, downloadPanel.element);
    await downloadPanel.loadOffer();
  }

  /** `promptOnceReady` is what the person was about to ask when the offer
   * interrupted them. It goes into the message field when the image model
   * lands, so the download ends where they left off; without it they came
   * back to an empty field and had to pick the idea a second time. */
  private offerImageModel(profile: AstroProfile, promptOnceReady: string | null): void {
    const fillPromptOnceReady = (): void => {
      if (promptOnceReady !== null) {
        this.onStarterPrompt(promptOnceReady);
      }
    };
    const imageModelOffer = new ImageModelOffer(this.document, ID_PREFIX, this.library, () => this.replaceFollowUp(null), fillPromptOnceReady);
    const reply = makeElement(this.document, "p", "astro-home__reply", astroLine(AstroMoment.NEEDS_IMAGE_MODEL, profile.nickname));
    this.replaceFollowUp(imageModelOffer, reply, imageModelOffer.element);
    void imageModelOffer.loadOffer();
  }
}
