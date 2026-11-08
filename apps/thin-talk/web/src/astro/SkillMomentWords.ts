import { ToolStepState } from "../ToolStep";
import { AstroStyle } from "./AstroStyle";

type MomentWords = Record<ToolStepState, { explorer: string; pilot: string }>;

/**
 * What Astro says it is doing while a tool runs, once it is done, and when it
 * could not, in both voices. Tools not listed here keep their own label, so a
 * new tool still shows up sensibly before it gets words of its own.
 */
const WORDS_BY_TOOL_NAME: Readonly<Record<string, MomentWords>> = {
  generate_image: {
    [ToolStepState.RUNNING]: { explorer: "Drawing in my sketchbook…", pilot: "Generating an image…" },
    [ToolStepState.SUCCEEDED]: { explorer: "Drew it in my sketchbook", pilot: "Image generated" },
    [ToolStepState.FAILED]: { explorer: "My sketchbook got stuck", pilot: "Image generation failed" },
  },
  describe_image: {
    [ToolStepState.RUNNING]: { explorer: "Looking at your photo…", pilot: "Describing the image…" },
    [ToolStepState.SUCCEEDED]: { explorer: "Looked at your photo", pilot: "Described the image" },
    [ToolStepState.FAILED]: { explorer: "I couldn't see that photo", pilot: "Couldn't describe the image" },
  },
  read_file: {
    [ToolStepState.RUNNING]: { explorer: "Reading your file…", pilot: "Reading the file…" },
    [ToolStepState.SUCCEEDED]: { explorer: "Read your file", pilot: "Read the file" },
    [ToolStepState.FAILED]: { explorer: "I couldn't read that file", pilot: "Couldn't read the file" },
  },
  search_conversations: {
    [ToolStepState.RUNNING]: { explorer: "Searching our chats…", pilot: "Searching past chats…" },
    [ToolStepState.SUCCEEDED]: { explorer: "Searched our chats", pilot: "Searched past chats" },
    [ToolStepState.FAILED]: { explorer: "I couldn't search our chats", pilot: "Search failed" },
  },
  get_current_time: {
    [ToolStepState.RUNNING]: { explorer: "Checking the clock…", pilot: "Checking the time…" },
    [ToolStepState.SUCCEEDED]: { explorer: "Checked the clock", pilot: "Checked the time" },
    [ToolStepState.FAILED]: { explorer: "I couldn't check the clock", pilot: "Couldn't check the time" },
  },
};

const FALLBACK_OUTCOME: Record<ToolStepState, string> = {
  [ToolStepState.RUNNING]: "working…",
  [ToolStepState.SUCCEEDED]: "done",
  [ToolStepState.FAILED]: "didn't work",
};

export function skillMomentWords(toolName: string, toolLabel: string, state: ToolStepState, style: AstroStyle): string {
  const knownWords = WORDS_BY_TOOL_NAME[toolName];
  if (knownWords === undefined) {
    return `${toolLabel}: ${FALLBACK_OUTCOME[state]}`;
  }
  return style === AstroStyle.PILOT ? knownWords[state].pilot : knownWords[state].explorer;
}
