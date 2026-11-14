import { ConversationSearch, type ConversationMatch } from "./ConversationSearch";
import type { ConversationTool, ToolCallContext, ToolParametersSchema } from "./ConversationTool";
import { runHostToolUse } from "./HostToolUseRunner";
import { SessionBridge } from "./SessionBridge";

/** The newest conversations searched; older ones are left out so a search
 * stays quick however long the history grows. */
const MAXIMUM_SEARCHED_CONVERSATIONS = 100;
/** The most matches returned to the model. */
const MAXIMUM_SEARCH_MATCHES = 10;
/** The tool waits for the user to answer the approval sheet, which takes as
 * long as they take; this bounds a use nobody ever answers. */
const SEARCH_TOOL_TIMEOUT_MILLISECONDS = 10 * 60_000;

/**
 * Searches the user's saved conversations for a phrase. The page does the
 * searching, since it owns the conversation format; the host only asks the
 * user and stores the files. It needs approval because it shows the model
 * conversations other than the one it is answering. Titles come from the list
 * the host already returns, so only conversations whose title misses are
 * loaded, one at a time.
 */
export class SearchConversationsTool implements ConversationTool {
  public readonly name = "search_conversations";
  public readonly label = "Search conversations";
  public readonly description =
    "Searches the user's saved Astronomical conversations for a word or phrase and returns where it appears. Use it when the user refers to something discussed in an earlier conversation.";
  public readonly parameters: ToolParametersSchema = {
    type: "object",
    properties: { query: { type: "string", description: "The word or phrase to look for." } },
    required: ["query"],
  };
  public readonly timeoutMilliseconds = SEARCH_TOOL_TIMEOUT_MILLISECONDS;
  public readonly needsUserAttention = true;
  private readonly bridge: SessionBridge;

  public constructor(bridge: SessionBridge) {
    this.bridge = bridge;
  }

  public async execute(toolArguments: Record<string, unknown>, signal: AbortSignal, context: ToolCallContext): Promise<string> {
    const query = typeof toolArguments["query"] === "string" ? toolArguments["query"].trim() : "";
    if (query.length === 0) {
      throw new Error("The search needs a word or phrase.");
    }
    await runHostToolUse(this.bridge, this.name, this.label, context, signal);
    const matches = await this.searchSavedConversations(query, context.conversationID, signal);
    if (matches.length === 0) {
      return `No saved conversation mentions "${query}".`;
    }
    const matchLines = matches.map((match) => `- ${match.title || "Untitled"} (${match.updatedAt.slice(0, 10)}): ${match.snippet}`);
    return `Conversations that mention "${query}":\n${matchLines.join("\n")}`;
  }

  /** The conversation being answered is left out: it holds the question
   * itself, which would always be the first match. A load that fails ends the
   * search with what was found, rather than waiting on a host that is not
   * answering once per remaining conversation. */
  private async searchSavedConversations(query: string, currentConversationID: string | null, signal: AbortSignal): Promise<ConversationMatch[]> {
    const conversationSummaries = (await this.bridge.list())
      .filter((conversationSummary) => conversationSummary.id !== currentConversationID)
      .slice(0, MAXIMUM_SEARCHED_CONVERSATIONS);
    const matches: ConversationMatch[] = [];
    for (const conversationSummary of conversationSummaries) {
      signal.throwIfAborted();
      if (matches.length >= MAXIMUM_SEARCH_MATCHES) {
        break;
      }
      const titleSnippet = ConversationSearch.snippetAround(conversationSummary.title, query);
      if (titleSnippet != null) {
        matches.push({ title: conversationSummary.title, updatedAt: conversationSummary.updatedAt, snippet: titleSnippet });
        continue;
      }
      const conversation = await this.bridge.load(conversationSummary.id).catch(() => undefined);
      if (conversation === undefined) {
        break;
      }
      const messageMatch = conversation != null ? ConversationSearch.firstMessageMatch(conversation, query) : null;
      if (messageMatch != null) {
        matches.push(messageMatch);
      }
    }
    return matches;
  }
}
