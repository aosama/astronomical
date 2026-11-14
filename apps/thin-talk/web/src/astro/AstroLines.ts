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
  [AstroMoment.LIBRARY_BUSY]: () => "Another download is running. Try again when it finishes.",
  [AstroMoment.SKILL_CHAT]: () => "Chat is ready. A question is in the message field.",
  [AstroMoment.SKILL_PICTURES]: () => "Pictures are ready. A prompt is in the message field.",
  [AstroMoment.NEEDS_IMAGE_MODEL]: () => "Pictures need an image model: a one-time download.",
  [AstroMoment.SKILL_LOOK]: () => "Pick a photo and Astro will describe it.",
  [AstroMoment.SKILL_READ]: () => "Pick a file and Astro will summarise or explain it.",
  [AstroMoment.SKILL_VOICE]: () => "Voice is coming soon.",
  [AstroMoment.HOME_GREETING]: (nickname) => addressed("Hey", nickname),
  [AstroMoment.HOME_PROMPT]: () => "What are we working on?",
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

/** One idea offered as a chip on Astro's home. */
export interface HomeIdea {
  readonly prompt: string;
  /** True when the idea asks for a picture. Without an image model the chip
   * offers that download instead of a prompt that can only fail. */
  readonly needsImageModel: boolean;
}

export const HOME_IDEAS: readonly HomeIdea[] = [
  { prompt: "Explain black holes in three sentences", needsImageModel: false },
  { prompt: "Quiz me on the French Revolution", needsImageModel: false },
  { prompt: "Help me plan a science project", needsImageModel: false },
  { prompt: "Draw a poster of a rocket launch", needsImageModel: true },
];
