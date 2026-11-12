import { ToolStepState } from "../ToolStep";

/**
 * What Astro says it is doing while a tool runs, once it is done, and when it
 * could not. Tools not listed here keep their own label, so a new tool still
 * shows up sensibly before it gets words of its own.
 */
const WORDS_BY_TOOL_NAME: Readonly<Record<string, Record<ToolStepState, string>>> = {
  generate_image: {
    [ToolStepState.RUNNING]: "Generating an image…",
    [ToolStepState.SUCCEEDED]: "Image generated",
    [ToolStepState.FAILED]: "Image generation failed",
  },
  describe_image: {
    [ToolStepState.RUNNING]: "Describing the image…",
    [ToolStepState.SUCCEEDED]: "Described the image",
    [ToolStepState.FAILED]: "Couldn't describe the image",
  },
  read_file: {
    [ToolStepState.RUNNING]: "Reading the file…",
    [ToolStepState.SUCCEEDED]: "Read the file",
    [ToolStepState.FAILED]: "Couldn't read the file",
  },
  search_conversations: {
    [ToolStepState.RUNNING]: "Searching past chats…",
    [ToolStepState.SUCCEEDED]: "Searched past chats",
    [ToolStepState.FAILED]: "Search failed",
  },
  get_current_time: {
    [ToolStepState.RUNNING]: "Checking the time…",
    [ToolStepState.SUCCEEDED]: "Checked the time",
    [ToolStepState.FAILED]: "Couldn't check the time",
  },
};

const FALLBACK_OUTCOME: Record<ToolStepState, string> = {
  [ToolStepState.RUNNING]: "working…",
  [ToolStepState.SUCCEEDED]: "done",
  [ToolStepState.FAILED]: "didn't work",
};

export function skillMomentWords(toolName: string, toolLabel: string, state: ToolStepState): string {
  const knownWords = WORDS_BY_TOOL_NAME[toolName];
  if (knownWords === undefined) {
    return `${toolLabel}: ${FALLBACK_OUTCOME[state]}`;
  }
  return knownWords[state];
}
