import type { SessionBridge } from "../SessionBridge";
import type { AstroLibrary } from "./AstroLibrary";
import { AstroHomeView } from "./AstroHomeView";
import { AstroProfile } from "./AstroProfile";
import { FirstFlightView } from "./FirstFlightView";
import type { AstroStyle } from "./AstroStyle";
import { SkillCardKind, skillCard, starterPromptFor } from "./SkillCard";
import { AstroComposerChrome } from "./AstroComposerChrome";
import { TranscriptPersona } from "../TranscriptPersona";

/**
 * Astro's place in Thin Talk: loads what Astro remembers, opens the first
 * flight for a newcomer, shows Astro's home in empty conversations, answers
 * the View menu's "Show or Hide Astro" and "Meet Astro Again", and fills the
 * message field when a card or idea is picked.
 *
 * Chat itself is untouched: Astro only ever prepares a message, and the
 * controller sends it exactly as if it had been typed.
 */
export class AstroCompanion {
  private readonly bridge: SessionBridge;
  private readonly library: AstroLibrary;
  private readonly composerField: HTMLTextAreaElement;
  private readonly applyPresentation: (persona: TranscriptPersona) => void;
  private readonly composerChrome: AstroComposerChrome;
  private readonly firstFlight: FirstFlightView;
  private readonly home: AstroHomeView;
  private profile = AstroProfile.newcomer();

  public constructor(init: {
    bridge: SessionBridge;
    library: AstroLibrary;
    firstFlightOverlay: HTMLElement;
    emptyState: HTMLElement;
    composerField: HTMLTextAreaElement;
    skillLaunchers: HTMLElement;
    composerAstro: HTMLImageElement;
    /** Shows the window classic or with Astro; see applyAstroPresentation. */
    applyPresentation: (persona: TranscriptPersona) => void;
    /** Opens an empty conversation, where Astro's home is shown. */
    startNewChat: () => Promise<void>;
    /** A model Astro downloaded has landed: the chat window re-reads its models. */
    onModelInstalled: () => void;
  }) {
    this.bridge = init.bridge;
    this.library = init.library;
    this.composerField = init.composerField;
    this.applyPresentation = init.applyPresentation;
    this.composerChrome = new AstroComposerChrome({
      launchers: init.skillLaunchers,
      listeningAstroImage: init.composerAstro,
      composerField: init.composerField,
      watchedArea: init.composerField.ownerDocument.body,
      onStarterPrompt: (prompt) => this.prepareMessage(prompt),
      hasImageModel: () => init.library.hasImageModel(),
      onNeedsSketchbook: () => void this.offerSketchbookInNewChat(init.startNewChat),
    });
    this.firstFlight = new FirstFlightView({
      overlay: init.firstFlightOverlay,
      library: init.library,
      callbacks: {
        onNicknameChosen: (typedNickname) => this.remember(this.profile.withNickname(typedNickname)),
        onStyleChosen: (style: AstroStyle) => this.remember(this.profile.withStyle(style)),
        onFinished: (chosenSkill) => this.finishFirstFlight(chosenSkill),
      },
    });
    this.home = new AstroHomeView({
      emptyState: init.emptyState,
      library: init.library,
      onStarterPrompt: (prompt) => this.prepareMessage(prompt),
    });
    init.library.onModelInstalled(() => this.welcomeNewModel(init.onModelInstalled));
  }

  /** A new brain or sketchbook changes what home offers, so home is redrawn
   * once the chat window has heard about the model. */
  private welcomeNewModel(onModelInstalled: () => void): void {
    onModelInstalled();
    if (this.profile.isAstroShown) {
      this.home.show(this.profile);
    }
  }

  /** Reads the saved profile; a newcomer starts the first flight. A profile
   * the host cannot read counts as a newcomer, which at worst replays the
   * welcome once. */
  public async start(): Promise<void> {
    try {
      this.profile = AstroProfile.fromPlain(await this.bridge.loadAstroProfile());
    } catch {
      this.profile = AstroProfile.newcomer();
    }
    this.presentProfile();
    if (!this.profile.hasFinishedFirstFlight) {
      await this.openFirstFlight();
    }
  }

  /** "Meet Astro Again": replays the welcome, keeping the saved name and style
   * as the starting answers, and brings Astro back if it was hidden. */
  public async meetAgain(): Promise<void> {
    if (this.firstFlight.isOpen) {
      return;
    }
    this.remember(this.profile.withAstroShown(true));
    this.presentProfile();
    await this.openFirstFlight();
  }

  /** The Focus switch. */
  public toggleShown(): void {
    this.remember(this.profile.withAstroShown(!this.profile.isAstroShown));
    this.presentProfile();
  }

  /** The sketchbook offer lives on Astro's home, which an empty chat shows. */
  private async offerSketchbookInNewChat(startNewChat: () => Promise<void>): Promise<void> {
    await startNewChat();
    await this.home.chooseSkill(SkillCardKind.PICTURES);
  }

  private async openFirstFlight(): Promise<void> {
    const hasChatModel = await this.library.hasChatModel();
    // An unreachable daemon (null) skips the wake-up screen: a download could
    // not start anyway, and the window's own notice explains the daemon.
    this.firstFlight.open(this.profile, hasChatModel !== false);
  }

  private finishFirstFlight(chosenSkill: SkillCardKind | null): void {
    this.remember(this.profile.withFirstFlightFinished());
    this.presentProfile();
    if (chosenSkill !== null) {
      const starterPrompt = starterPromptFor(skillCard(chosenSkill), this.profile.style);
      if (starterPrompt !== null) {
        this.prepareMessage(starterPrompt);
        return;
      }
    }
    this.composerField.focus();
  }

  private presentProfile(): void {
    this.applyPresentation(
      new TranscriptPersona({ isAstroShown: this.profile.isAstroShown, style: this.profile.style, nickname: this.profile.nickname }),
    );
    if (this.profile.isAstroShown) {
      this.home.show(this.profile);
      this.composerChrome.show(this.profile.style);
      return;
    }
    this.home.hide();
    this.composerChrome.hide();
  }

  /** Puts a starter question in the message field the way typing would (the
   * input event lets the composer enable Send), after anything already typed
   * rather than over it, and leaves the caret at the end, ready to edit. */
  private prepareMessage(prompt: string): void {
    const draft = this.composerField.value.trimEnd();
    this.composerField.value = draft.length > 0 ? `${draft} ${prompt}` : prompt;
    const pageWindow = this.composerField.ownerDocument.defaultView;
    if (pageWindow !== null) {
      this.composerField.dispatchEvent(new pageWindow.Event("input", { bubbles: true }));
    }
    this.composerField.focus();
    const caretPosition = this.composerField.value.length;
    this.composerField.setSelectionRange(caretPosition, caretPosition);
  }

  private remember(profile: AstroProfile): void {
    this.profile = profile;
    void this.bridge.saveAstroProfile(profile.toPlain()).catch(() => undefined);
  }
}
