/**
 * What one answer, or a whole conversation, mostly did: chatted, drew, looked
 * at a photo, read a file, searched past chats or worked through maths. It is
 * derived from the tools an answer used, saved with each conversation, and
 * drawn by Astro's look as skill tags and sidebar colours. Chat is every answer
 * that used no other skill.
 */
export enum ConversationSkill {
  CHAT = "chat",
  PICTURES = "pictures",
  LOOK = "look",
  READ = "read",
  SEARCH = "search",
  MATHS = "maths",
}

export const CONVERSATION_SKILL_LABELS: Record<ConversationSkill, string> = {
  [ConversationSkill.CHAT]: "Chat",
  [ConversationSkill.PICTURES]: "Pictures",
  [ConversationSkill.LOOK]: "Look",
  [ConversationSkill.READ]: "Read",
  [ConversationSkill.SEARCH]: "Search",
  [ConversationSkill.MATHS]: "Maths",
};

/** The page's tool names (each tool's `name`) and the skill each one shows.
 * The clock is left out: telling the time is part of chatting. */
const SKILL_BY_TOOL_NAME: Readonly<Record<string, ConversationSkill>> = {
  generate_image: ConversationSkill.PICTURES,
  describe_image: ConversationSkill.LOOK,
  read_file: ConversationSkill.READ,
  search_conversations: ConversationSkill.SEARCH,
};

/** Display maths ($$…$$, \[…\]) or inline \(…\) in the answer. A lone dollar
 * is left out on purpose: prices ("$5") must not turn a chat into maths. */
const MATHS_PATTERN = /\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)/;

export function skillForToolName(toolName: string): ConversationSkill | null {
  const mappedSkill = SKILL_BY_TOOL_NAME[toolName];
  return mappedSkill !== undefined ? mappedSkill : null;
}

/** The skill one answer shows: the last skill-bearing tool it used, else maths
 * when it writes formulas, else chat. */
export function answerSkill(markdown: string, toolNames: readonly string[]): ConversationSkill {
  for (let toolPosition = toolNames.length - 1; toolPosition >= 0; toolPosition -= 1) {
    const toolSkill = skillForToolName(toolNames[toolPosition] ?? "");
    if (toolSkill !== null) {
      return toolSkill;
    }
  }
  return MATHS_PATTERN.test(markdown) ? ConversationSkill.MATHS : ConversationSkill.CHAT;
}

/**
 * The skill a whole chat used most, for its colour in the sidebar. Chat only
 * wins when nothing else was used: a conversation with one drawing among ten
 * questions is remembered as the one where Astro drew. Ties go to the skill
 * used most recently.
 */
export function dominantSkill(answers: readonly { markdown: string; toolNames: readonly string[] }[]): ConversationSkill {
  const useCountBySkill = new Map<ConversationSkill, number>();
  const lastUseBySkill = new Map<ConversationSkill, number>();
  answers.forEach((answer, answerPosition) => {
    const skill = answerSkill(answer.markdown, answer.toolNames);
    useCountBySkill.set(skill, (useCountBySkill.get(skill) ?? 0) + 1);
    lastUseBySkill.set(skill, answerPosition);
  });
  let winningSkill = ConversationSkill.CHAT;
  for (const [skill, useCount] of useCountBySkill) {
    if (skill === ConversationSkill.CHAT) {
      continue;
    }
    const winningCount = winningSkill === ConversationSkill.CHAT ? 0 : useCountBySkill.get(winningSkill) ?? 0;
    const isMoreRecentTie = useCount === winningCount && (lastUseBySkill.get(skill) ?? -1) > (lastUseBySkill.get(winningSkill) ?? -1);
    if (useCount > winningCount || isMoreRecentTie) {
      winningSkill = skill;
    }
  }
  return winningSkill;
}

/** Reads a stored skill name; anything unknown is chat. */
export function parseConversationSkill(storedSkill: unknown): ConversationSkill {
  const knownSkills: readonly string[] = Object.values(ConversationSkill);
  return typeof storedSkill === "string" && knownSkills.includes(storedSkill) ? (storedSkill as ConversationSkill) : ConversationSkill.CHAT;
}
