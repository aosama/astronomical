/** Every moment Astro says something. */
export enum AstroMoment {
  WELCOME = "welcome",
  NEEDS_CHAT_MODEL = "needsChatModel",
  DOWNLOADING = "downloading",
  MODEL_READY = "modelReady",
  DOWNLOAD_FAILED = "downloadFailed",
  LIBRARY_BUSY = "libraryBusy",
  SKILL_CHAT = "skillChat",
  SKILL_PICTURES = "skillPictures",
  NEEDS_IMAGE_MODEL = "needsImageModel",
  SKILL_LOOK = "skillLook",
  SKILL_READ = "skillRead",
  SKILL_VOICE = "skillVoice",
  HOME_GREETING = "homeGreeting",
  HOME_PROMPT = "homePrompt",
  SETTINGS = "settings",
  ASK_NAME = "askName",
  PRIVACY = "privacy",
  SAVE_FAILED = "saveFailed",
  ENGINE_SILENT = "engineSilent",
  ENGINE_STARTING = "engineStarting",
  ENGINE_WONT_START = "engineWontStart",
}

/** "Hey, Maya" with a name, "Hey" without one: a name is optional, and
 * Astro must never say "Hey, " or invent a name. */
function addressed(opening: string, nickname: string): string {
  return nickname.length > 0 ? `${opening}, ${nickname}` : opening;
}

/**
 * Astro's words. Kept in one table, apart from any drawing, so the copy can
 * be read and reviewed as a script: short plain sentences, the words the rest
 * of the app uses ("chat model", "image model"), and never blaming the person
 * when something goes wrong.
 */
const LINES: Record<AstroMoment, (nickname: string) => string> = {
  [AstroMoment.WELCOME]: () => "Astronomical runs AI models privately on this Mac. Its assistant is called Astro.",
  [AstroMoment.NEEDS_CHAT_MODEL]: () => "Download a chat model to get started. It's a one-time download.",
  [AstroMoment.DOWNLOADING]: () => "Downloading. You can keep going while it runs.",
  // Shown under a headline that already says the model is ready, so it adds
  // what a newcomer does not know yet instead of repeating the headline.
  [AstroMoment.MODEL_READY]: () => "It is on this Mac now and works without the internet.",
  [AstroMoment.DOWNLOAD_FAILED]: () => "The download stopped. Try again?",
  [AstroMoment.LIBRARY_BUSY]: () => "Another download is paused. Resume or cancel it on the Models page, then try again.",
  [AstroMoment.SKILL_CHAT]: () => "Chat is ready. A question is in the message field.",
  [AstroMoment.SKILL_PICTURES]: () => "Pictures are ready. A prompt is in the message field.",
  [AstroMoment.NEEDS_IMAGE_MODEL]: () => "Pictures need an image model: a one-time download.",
  [AstroMoment.SKILL_LOOK]: () => "Pick a photo and Astro will describe it.",
  [AstroMoment.SKILL_READ]: () => "Pick a file and Astro will summarise or explain it.",
  [AstroMoment.SKILL_VOICE]: () => "Voice is coming soon.",
  [AstroMoment.HOME_GREETING]: (nickname) => addressed(timeOfDayOpening(new Date().getHours()), nickname),
  [AstroMoment.HOME_PROMPT]: () => "What can I help with?",
  [AstroMoment.SETTINGS]: () => "These settings are kept on this Mac.",
  [AstroMoment.ASK_NAME]: () => "What should Astro call you?",
  [AstroMoment.PRIVACY]: () => "What Astro does and what stays private.",
  [AstroMoment.SAVE_FAILED]: () => "That didn't save. Try again?",
  [AstroMoment.ENGINE_STARTING]: () => "One moment. Starting up.",
  [AstroMoment.ENGINE_WONT_START]: () => "The engine didn't start. A restart usually fixes it.",
  [AstroMoment.ENGINE_SILENT]: () => "The engine isn't answering. Check again in a moment.",
};

export function astroLine(moment: AstroMoment, nickname: string): string {
  return LINES[moment](nickname);
}

/** "Good morning" before noon, "Good afternoon" until six, "Good evening"
 * after that and through the night: a greeting that follows the clock reads
 * as the Mac noticing the person, where a fixed "Hey" read as a slogan. */
export function timeOfDayOpening(hourOfDay: number): string {
  if (hourOfDay >= 5 && hourOfDay < 12) {
    return "Good morning";
  }
  if (hourOfDay >= 12 && hourOfDay < 18) {
    return "Good afternoon";
  }
  return "Good evening";
}

/** What kind of help a home idea shows off. Each kind is drawn with the glyph
 * the message field's own buttons use, so the idea also teaches that button. */
export enum HomeIdeaKind {
  WRITE = "write",
  PICTURES = "pictures",
  LOOK = "look",
  READ = "read",
}

/** One idea offered as a chip on Astro's home. */
export interface HomeIdea {
  readonly kind: HomeIdeaKind;
  readonly prompt: string;
  /** True when the idea asks for a picture. Without an image model the chip
   * offers that download instead of a prompt that can only fail. */
  readonly needsImageModel: boolean;
}

/**
 * Everyday requests for the person Astronomical is for, an ordinary Mac
 * owner, rather than school exercises. One idea of each kind is shown, so the
 * four chips together show everything the window can do. The photo and file
 * ideas keep "I choose" in their words: that is what makes the model open the
 * picker instead of asking the person to paste something.
 */
const HOME_IDEA_CHOICES: Readonly<Record<HomeIdeaKind, readonly string[]>> = {
  [HomeIdeaKind.WRITE]: [
    "Write a polite reply to a late invoice",
    "Plan a week of easy dinners",
    "Help me write a thank-you note to a neighbour",
  ],
  [HomeIdeaKind.PICTURES]: [
    "Draw a birthday card for my mum",
    "Draw a poster for a garage sale",
    "Draw a cosy cabin in the snow",
  ],
  [HomeIdeaKind.LOOK]: [
    "Describe a photo I choose",
    "Read the text in a photo I choose",
  ],
  [HomeIdeaKind.READ]: [
    "Summarise a file I choose in five points",
    "Explain a document I choose in plain words",
  ],
};

const HOME_IDEA_ORDER: readonly HomeIdeaKind[] = [HomeIdeaKind.WRITE, HomeIdeaKind.PICTURES, HomeIdeaKind.LOOK, HomeIdeaKind.READ];
const MILLISECONDS_PER_DAY = 86_400_000;

/** The four ideas shown on `day`, one of each kind. They change once a day,
 * not on every visit, so an idea someone meant to try is still there when
 * they come back to an empty chat later the same day. */
export function homeIdeasFor(day: Date): readonly HomeIdea[] {
  const dayNumber = Math.floor((day.getTime() - day.getTimezoneOffset() * 60_000) / MILLISECONDS_PER_DAY);
  return HOME_IDEA_ORDER.map((kind) => ({ kind, prompt: ideaOfTheDay(kind, dayNumber), needsImageModel: kind === HomeIdeaKind.PICTURES }));
}

function ideaOfTheDay(kind: HomeIdeaKind, dayNumber: number): string {
  const choices = HOME_IDEA_CHOICES[kind];
  const chosenPrompt = choices[Math.abs(dayNumber) % choices.length];
  if (chosenPrompt === undefined) {
    throw new Error(`there are no home ideas of kind "${kind}"`);
  }
  return chosenPrompt;
}
