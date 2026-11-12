import type { SessionBridge } from "../SessionBridge";
import type { AstroLibrary } from "./AstroLibrary";
import { AstroHomeView } from "./AstroHomeView";
import { AstroProfile } from "./AstroProfile";
import { FirstFlightView } from "./FirstFlightView";
import { EngineWatch } from "./EngineWatch";
import { YourAstroView } from "./YourAstroView";
import { SkillCardKind } from "./SkillCard";
import { AstroComposerChrome } from "./AstroComposerChrome";
import { TranscriptPersona } from "../TranscriptPersona";

/**
 * Astro's place in Thin Talk: loads what Astro remembers, opens the
 * get-started journey for a newcomer, shows Astro's home in empty
 * conversations, opens Settings (⌘,), answers the View menu's "Show or Hide
 * Astro", and fills the message field when a card or idea is picked.
 *
 * It is the one owner of the saved profile: every view asks it for changes,
 * and it redraws all of them from the one copy.
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
  private readonly yourAstro: YourAstroView;
  private readonly engineWatch: EngineWatch;
  private profile = AstroProfile.newcomer();
  /** False until start() has read the saved profile; see openYourAstro. */
  private hasLoadedProfile = false;
  /** Settings was asked for before the profile was read, as when the menu
   * bar's "Settings…" opens a new chat window; start() opens it then. */
  private opensYourAstroAfterLoad = false;

  public constructor(init: {
    bridge: SessionBridge;
    library: AstroLibrary;
    firstFlightOverlay: HTMLElement;
    yourAstroOverlay: HTMLElement;
    emptyState: HTMLElement;
    composerField: HTMLTextAreaElement;
    skillLaunchers: HTMLElement;
    /** Shows the window classic or with Astro; see applyAstroPresentation. */
    applyPresentation: (persona: TranscriptPersona) => void;
    /** Opens an empty conversation, where Astro's home is shown. */
    startNewChat: () => Promise<void>;
    /** A model Astro downloaded has landed: the chat window re-reads its models. */
    onModelInstalled: () => void;
    /** Asks the host to restart the engine; null where the host cannot. */
    restartEngine: (() => void) | null;
    /** Opens the Models page, where every downloadable model is listed. */
    openModelsPage: () => void;
  }) {
    this.bridge = init.bridge;
    this.library = init.library;
    this.composerField = init.composerField;
    this.applyPresentation = init.applyPresentation;
    this.engineWatch = new EngineWatch(init.library);
    this.composerChrome = new AstroComposerChrome({
      launchers: init.skillLaunchers,
      onStarterPrompt: (prompt) => this.prepareMessage(prompt),
      hasImageModel: () => init.library.hasImageModel(),
      onNeedsImageModel: () => void this.offerImageModelInNewChat(init.startNewChat),
    });
    this.firstFlight = new FirstFlightView({
      overlay: init.firstFlightOverlay,
      library: init.library,
      engineWatch: this.engineWatch,
      restartEngine: init.restartEngine,
      openModelsPage: init.openModelsPage,
      onFinished: () => this.finishFirstFlight(),
    });
    this.home = new AstroHomeView({
      emptyState: init.emptyState,
      library: init.library,
      onStarterPrompt: (prompt) => this.prepareMessage(prompt),
    });
    this.yourAstro = this.makeYourAstro(init.yourAstroOverlay, init.library);
    init.library.onModelInstalled(() => this.welcomeNewModel(init.onModelInstalled));
  }

  /** A new chat or image model changes what home offers, so home is redrawn
   * once the chat window has heard about the model. */
  private welcomeNewModel(onModelInstalled: () => void): void {
    onModelInstalled();
    if (this.profile.isAstroShown) {
      this.home.show(this.profile);
    }
    this.yourAstro.refreshModels();
  }

  /** Reads the saved profile; a newcomer starts the get-started journey. A profile
   * the host cannot read counts as a newcomer, which at worst replays the
   * welcome once. */
  public async start(): Promise<void> {
    try {
      this.profile = AstroProfile.fromPlain(await this.bridge.loadAstroProfile());
    } catch {
      this.profile = AstroProfile.newcomer();
    }
    this.presentProfile();
    void this.followEngineStart();
    if (!this.profile.hasFinishedFirstFlight) {
      await this.openFirstFlight();
    }
    // Only now: a newcomer's journey is open, so Settings stays ignored.
    this.hasLoadedProfile = true;
    if (this.opensYourAstroAfterLoad) {
      this.opensYourAstroAfterLoad = false;
      this.openYourAstro();
    }
  }

  /** Settings (⌘,). Ignored while the get-started journey is open, which
   * covers the window. Before the saved profile is read it waits, because
   * the page would show a newcomer's blank answers and a newcomer's journey
   * could then open underneath it. */
  public openYourAstro(): void {
    if (!this.hasLoadedProfile) {
      this.opensYourAstroAfterLoad = true;
      return;
    }
    if (this.firstFlight.isOpen || this.yourAstro.isOpen) {
      return;
    }
    this.yourAstro.open();
  }

  private makeYourAstro(overlay: HTMLElement, library: AstroLibrary): YourAstroView {
    return new YourAstroView({
      overlay,
      library,
      callbacks: {
        currentProfile: () => this.profile,
        saveNickname: (typedNickname) => this.remember(this.profile.withNickname(typedNickname)),
        toggleShown: () => this.rememberShown(!this.profile.isAstroShown),
        retrySave: () => this.remember(this.profile),
        replay: () => void this.replayFirstFlight(),
        onClosed: () => this.composerField.focus(),
      },
    });
  }

  /** Settings' last row: the welcome from the start, with Astro brought
   * back if hidden. */
  private async replayFirstFlight(): Promise<void> {
    void this.rememberShown(true);
    await this.openFirstFlight();
  }

  /** The Focus switch. */
  public toggleShown(): void {
    void this.rememberShown(!this.profile.isAstroShown);
  }

  private rememberShown(isAstroShown: boolean): Promise<boolean> {
    return this.remember(this.profile.withAstroShown(isAstroShown));
  }

  /** The image model's offer lives on Astro's home, which an empty chat shows. */
  private async offerImageModelInNewChat(startNewChat: () => Promise<void>): Promise<void> {
    await startNewChat();
    await this.home.chooseSkill(SkillCardKind.PICTURES);
  }

  private async openFirstFlight(): Promise<void> {
    this.firstFlight.open(await this.engineWatch.ask());
  }

  /** A window opened while the engine is still starting drew home and
   * Settings without knowing whether a chat model is installed. When the
   * engine first answers they are drawn again, so a Mac without one gets
   * its offer instead of silence. */
  private async followEngineStart(): Promise<void> {
    if ((await this.engineWatch.ask()) !== null) {
      return;
    }
    this.engineWatch.onceAnswered(() => {
      if (this.profile.isAstroShown) {
        this.home.show(this.profile);
      }
      this.yourAstro.refreshModels();
    });
  }

  /** The journey ends in an empty chat with the caret in the message field. */
  private finishFirstFlight(): void {
    void this.remember(this.profile.withFirstFlightFinished());
    this.composerField.focus();
  }

  private presentProfile(): void {
    this.yourAstro.refresh();
    this.applyPresentation(
      new TranscriptPersona({ isAstroShown: this.profile.isAstroShown, nickname: this.profile.nickname }),
    );
    if (this.profile.isAstroShown) {
      this.home.show(this.profile);
      this.composerChrome.show();
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

  /** Keeps a change and redraws every view from it at once, then answers
   * whether it reached the disk. A failed save keeps the change on screen
   * for this session; Settings says it was not kept and offers to retry. */
  private async remember(profile: AstroProfile): Promise<boolean> {
    this.profile = profile;
    this.presentProfile();
    try {
      await this.bridge.saveAstroProfile(profile.toPlain());
      return true;
    } catch {
      return false;
    }
  }
}
