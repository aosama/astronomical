/**
 * One entry in the sidebar's session list: everything the list needs without
 * loading the full transcript.
 */
export class SessionSummary {
  public readonly id: string;
  public readonly title: string;
  public readonly updatedAt: string;

  public constructor(init: { id: string; title: string; updatedAt: string }) {
    this.id = init.id;
    this.title = init.title;
    this.updatedAt = init.updatedAt;
  }
}
