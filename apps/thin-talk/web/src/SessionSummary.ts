import { ConversationSkill } from "./ConversationSkill";

/**
 * One entry in the sidebar's session list: everything the list needs without
 * loading the full transcript.
 */
export class SessionSummary {
  public readonly id: string;
  public readonly title: string;
  public readonly updatedAt: string;
  /** The chat's main skill, for its colour while Astro is shown; chat for
   * files saved before skills were recorded. */
  public readonly skill: ConversationSkill;

  public constructor(init: { id: string; title: string; updatedAt: string; skill?: ConversationSkill }) {
    this.id = init.id;
    this.title = init.title;
    this.updatedAt = init.updatedAt;
    this.skill = init.skill ?? ConversationSkill.CHAT;
  }
}
