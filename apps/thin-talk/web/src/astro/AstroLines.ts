import { AstroStyle } from "./AstroStyle";

/** Every moment Astro says something. */
export enum AstroMoment {
  HELLO = "hello",
  ASK_NAME = "askName",
  ASK_STYLE = "askStyle",
  GROWN_UP = "grownUp",
  NEEDS_BRAIN = "needsBrain",
  CHARGING = "charging",
  AWAKE = "awake",
  DOWNLOAD_FAILED = "downloadFailed",
  LIBRARY_BUSY = "libraryBusy",
  ASK_SKILL = "askSkill",
  SKILL_CHAT = "skillChat",
  SKILL_PICTURES = "skillPictures",
  NEEDS_SKETCHBOOK = "needsSketchbook",
  SKILL_LOOK = "skillLook",
  SKILL_READ = "skillRead",
  SKILL_VOICE = "skillVoice",
  HOME_GREETING = "homeGreeting",
  HOME_PROMPT = "homePrompt",
  YOUR_ASTRO = "yourAstro",
  ASK_NEW_NAME = "askNewName",
  STYLE_PREVIEW = "stylePreview",
  SAVE_FAILED = "saveFailed",
  ENGINE_SILENT = "engineSilent",
}

/** "Hi, Maya!" with a name, "Hi!" without one: a nickname is optional, and
 * Astro must never say "Hi, !" or invent a name. */
function addressed(opening: string, nickname: string, ending: string): string {
  return nickname.length > 0 ? `${opening}, ${nickname}${ending}` : `${opening}${ending}`;
}

type LinesForNickname = (nickname: string) => { explorer: string; pilot: string };

/**
 * Astro's words, in both voices. Kept in one table, apart from any drawing, so
 * the copy can be read and reviewed as a script: short sentences, no model
 * names, no jargon, and never blaming the person when something goes wrong.
 */
const LINES: Record<AstroMoment, LinesForNickname> = {
  [AstroMoment.HELLO]: () => ({ explorer: "Hi! I'm Astro. I live right here on your Mac!", pilot: "Hey, I'm Astro. I run right here on this Mac." }),
  [AstroMoment.ASK_NAME]: () => ({ explorer: "What should I call you?", pilot: "What should I call you?" }),
  [AstroMoment.ASK_STYLE]: (nickname) => ({
    explorer: addressed("Nice to meet you", nickname, "! How do you like your space?"),
    pilot: addressed("Good to meet you", nickname, ". Pick a style."),
  }),
  [AstroMoment.GROWN_UP]: () => ({ explorer: "Here's what grown-ups usually want to know.", pilot: "Here's what grown-ups usually want to know." }),
  [AstroMoment.NEEDS_BRAIN]: () => ({ explorer: "I need a brain to think with. It's a one-time download!", pilot: "First I need a brain: a one-time download." }),
  [AstroMoment.CHARGING]: () => ({ explorer: "Charging up… this tickles!", pilot: "Downloading. You can keep going while it runs." }),
  [AstroMoment.AWAKE]: () => ({ explorer: "I'm awake! Thank you!", pilot: "Online and ready." }),
  [AstroMoment.DOWNLOAD_FAILED]: () => ({ explorer: "Oops, my brain didn't arrive. Shall we try again?", pilot: "The download stopped. Try again?" }),
  [AstroMoment.LIBRARY_BUSY]: () => ({ explorer: "Something else is downloading right now. Let's try again when it's done!", pilot: "Another download is running. Try again when it finishes." }),
  [AstroMoment.ASK_SKILL]: () => ({ explorer: "Teach me something! Drop a card on me.", pilot: "Give me a skill: drop a card on me." }),
  [AstroMoment.SKILL_CHAT]: () => ({ explorer: "Yay! Ask me anything. I put a question in for you!", pilot: "Chat's ready. I've started a question for you." }),
  [AstroMoment.SKILL_PICTURES]: () => ({ explorer: "I love drawing! I put an idea in for you.", pilot: "Pictures are ready. I've started a prompt for you." }),
  [AstroMoment.NEEDS_SKETCHBOOK]: () => ({ explorer: "To draw I need my sketchbook. It's a one-time download!", pilot: "Pictures need my sketchbook: a one-time download." }),
  [AstroMoment.SKILL_LOOK]: () => ({ explorer: "Show me a photo and I'll tell you what I see!", pilot: "Pick a photo and I'll describe it." }),
  [AstroMoment.SKILL_READ]: () => ({ explorer: "Give me a file and I'll explain it, step by step!", pilot: "Hand me a file and I'll summarise or explain it." }),
  [AstroMoment.SKILL_VOICE]: () => ({ explorer: "Talking out loud is coming soon. I'm practising!", pilot: "Voice is coming soon." }),
  [AstroMoment.HOME_GREETING]: (nickname) => ({ explorer: addressed("Hi", nickname, "!"), pilot: addressed("Hey", nickname, "") }),
  [AstroMoment.HOME_PROMPT]: () => ({ explorer: "What should we do today?", pilot: "What are we working on?" }),
  [AstroMoment.YOUR_ASTRO]: (nickname) => ({
    explorer: addressed("Here's everything I remember about you", nickname, ". Change anything you like!"),
    pilot: "Here's what I remember. Change anything.",
  }),
  [AstroMoment.ASK_NEW_NAME]: () => ({ explorer: "Want a new name? What should I call you now?", pilot: "What should I call you?" }),
  // Each voice describes itself, so picking a card is a preview of how Astro
  // will sound before anything is saved.
  [AstroMoment.STYLE_PREVIEW]: () => ({ explorer: "Ooh, Explorer! Big buttons and lots of help. Is this the one?", pilot: "Pilot mode. Sleek and quick. Save it?" }),
  [AstroMoment.SAVE_FAILED]: () => ({ explorer: "Uh-oh, I couldn't remember that. Let's try again!", pilot: "That didn't save. Try again?" }),
  [AstroMoment.ENGINE_SILENT]: () => ({ explorer: "I can't hear my engine right now. Let's check again in a moment.", pilot: "My engine isn't answering. Check again in a moment." }),
};

export function astroLine(moment: AstroMoment, style: AstroStyle, nickname: string): string {
  const linePair = LINES[moment](nickname);
  return style === AstroStyle.PILOT ? linePair.pilot : linePair.explorer;
}

/** Ideas offered as one-click chips on Astro's home, per voice. */
export function homeIdeas(style: AstroStyle): readonly string[] {
  return style === AstroStyle.PILOT
    ? ["Explain black holes in three sentences", "Quiz me on the French Revolution", "Help me plan a science project", "Draw a poster of a rocket launch"]
    : ["Why is the sky blue?", "Draw a cat astronaut", "Help me with my maths", "Tell me a story about a brave robot"];
}
