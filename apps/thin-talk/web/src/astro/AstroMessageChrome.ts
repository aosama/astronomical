import { ChatMessageState } from "../ChatMessageState";
import type { TranscriptMessage } from "../TranscriptMessage";
import type { TranscriptPersona } from "../TranscriptPersona";
import { ToolStepState } from "../ToolStep";
import type { ToolStep } from "../ToolStep";
import { astroGalaxyHtml } from "./AstroGalaxy";
import { CONVERSATION_SKILL_LABELS, answerSkill, skillForToolName } from "../ConversationSkill";
import { stateWordsHtml } from "../MessageChrome";
import type { EscapeHtml, MessageChrome } from "../MessageChrome";
import { AstroStyle } from "./AstroStyle";
import { skillMomentWords } from "./SkillMomentWords";

/** What Astro is doing in one answer, which picks its status words and, on
 * the newest answer only, how its galaxy moves. */
export enum AstroMood {
  THINKING = "thinking",
  WORKING = "working",
  TALKING = "talking",
  DONE = "done",
  STOPPED = "stopped",
}

/** What Astro says in place of "thinking" while the engine loads a model. */
const MODEL_LOADING_WORDS = { explorer: "is waking up its brain…", pilot: "loading the model…" };

const STATUS_BY_MOOD: Record<AstroMood, { explorer: string; pilot: string }> = {
  [AstroMood.THINKING]: { explorer: "is thinking…", pilot: "thinking…" },
  [AstroMood.WORKING]: { explorer: "is busy…", pilot: "working…" },
  [AstroMood.TALKING]: { explorer: "is talking…", pilot: "" },
  [AstroMood.DONE]: { explorer: "", pilot: "" },
  [AstroMood.STOPPED]: { explorer: "stopped", pilot: "stopped" },
};

export function astroMood(message: TranscriptMessage): AstroMood {
  if (message.state === ChatMessageState.STREAMING) {
    if (message.toolSteps.some((toolStep) => toolStep.state === ToolStepState.RUNNING)) {
      return AstroMood.WORKING;
    }
    return message.markdown.length === 0 ? AstroMood.THINKING : AstroMood.TALKING;
  }
  return message.state === ChatMessageState.COMPLETE ? AstroMood.DONE : AstroMood.STOPPED;
}

function byStyle(words: { explorer: string; pilot: string }, persona: TranscriptPersona): string {
  return persona.style === AstroStyle.PILOT ? words.pilot : words.explorer;
}

/**
 * The message chrome while Astro is shown: Astro's galaxy and name with a skill
 * tag and friendly status, the person's nickname, "How I worked it out", and
 * tool steps told as skill moments. Every value that comes from the
 * conversation (the nickname, tool labels) is escaped; the rest is fixed text.
 */
export class AstroMessageChrome implements MessageChrome {
  private readonly persona: TranscriptPersona;
  private readonly escapeHtml: EscapeHtml;

  public constructor(persona: TranscriptPersona, escapeHtml: EscapeHtml) {
    this.persona = persona;
    this.escapeHtml = escapeHtml;
  }

  /** The sender inside the person's bubble: their initial and nickname, or
   * "You" when they gave none. */
  public userRoleHtml(): string {
    if (this.persona.nickname.length === 0) {
      return `<span class="message__role">You</span>`;
    }
    const initial = Array.from(this.persona.nickname)[0] ?? "";
    return `<span class="message__role message__role--named"><b class="astro-you-badge">${this.escapeHtml(initial.toUpperCase())}</b>${this.escapeHtml(this.persona.nickname)}</span>`;
  }

  /** Astro's galaxy, moving for its mood, then "Astro", the answer's skill
   * tag and what Astro is doing now. A failed answer keeps the plain word,
   * because the failure banner explains it. */
  public assistantHeaderHtml(message: TranscriptMessage): string {
    const skill = answerSkill(message.markdown, message.toolSteps.map((toolStep) => toolStep.toolName));
    const mood = astroMood(message);
    const statusText = message.state === ChatMessageState.FAILED ? "failed" : byStyle(STATUS_BY_MOOD[mood], this.persona);
    const caret = message.state === ChatMessageState.STREAMING ? '<span class="message__streaming-caret"></span>' : "";
    return (
      `<div class="astro-avatar" data-mood="${mood}" aria-hidden="true">${astroGalaxyHtml()}</div>` +
      `<div class="message__meta message__meta--astro">` +
      `<span class="message__role">Astro</span>` +
      `<span class="astro-skill-tag astro-skill-tag--${skill}" data-testid="astro-skill-tag">${CONVERSATION_SKILL_LABELS[skill]}</span>` +
      `<span class="message__state">${stateWordsHtml(statusText, mood === AstroMood.THINKING ? byStyle(MODEL_LOADING_WORDS, this.persona) : null)}${caret}</span>` +
      `</div>`
    );
  }

  public reasoningSummary(): string {
    return this.persona.style === AstroStyle.PILOT ? "Reasoning" : "How I worked it out";
  }

  /** The skill a step belongs to colours it (astro-conversation.css). */
  public toolStepAttributes(toolStep: ToolStep): string {
    const toolSkill = skillForToolName(toolStep.toolName);
    return toolSkill !== null ? ` data-skill="${toolSkill}"` : "";
  }

  /** A tool call told as Astro using a skill; Pilots also see which image
   * model drew a picture. */
  public toolStepSummaryHtml(toolStep: ToolStep, imageModelSuffix: string): string {
    const words = skillMomentWords(toolStep.toolName, toolStep.label, toolStep.state, this.persona.style);
    const suffix = this.persona.style === AstroStyle.PILOT ? imageModelSuffix : "";
    return `<summary><span class="skill-moment__icon" aria-hidden="true"></span><span class="skill-moment__words">${this.escapeHtml(words)}${suffix}</span></summary>`;
  }
}
