import { marked } from "marked";
import { MathPipeline } from "../MathPipeline";
import type { Token, Tokens } from "marked";

// Pictographs carry no words; the phonemizer would either drop them or spell
// out their names, and both sound like a glitch.
const PICTOGRAPH_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{200D}]/gu;
const SENTENCE_END_RE = /[.!?:;…]["')\]]*$/;

/**
 * The words a listener should hear for one reply, as plain sentences.
 *
 * Built from the reply's markdown rather than the rendered page, so it reads
 * the same in classic Thin Talk and Astro and never picks up button labels.
 * Prose, headings, lists and quotes are read; code, tables, pictures, raw
 * markup and maths are not, because hearing them recited is noise. Every
 * heading and list item ends with a full stop, so neighbours do not run
 * together into one breathless sentence.
 *
 * Returns "" when nothing speakable is left (a code-only or picture-only
 * reply), and the caller shows no Speak button for it.
 */
export function speakableTextOf(markdown: string): string {
  // A formula read symbol by symbol ("backslash frac x two") helps no
  // listener, so maths is left out, by the same rule the page uses to decide
  // what is maths: "$5 or $50" stays words, and code is never touched.
  const prose = MathPipeline.carveOutMath(markdown, []).replace(MathPipeline.MATH_TOKEN_PATTERN, " ");
  const paragraphs = blockSentences(marked.lexer(prose));
  return paragraphs
    .map((paragraph) => tidySpacing(paragraph.replace(PICTOGRAPH_RE, "")))
    .filter((paragraph) => /[\p{L}\p{N}]/u.test(paragraph))
    .join("\n\n");
}

/** Whether a reply has anything to read aloud; one without gets no Speak button. */
export function hasSpeakableText(markdown: string): boolean {
  return speakableTextOf(markdown).length > 0;
}

function blockSentences(blockTokens: readonly Token[]): string[] {
  const paragraphs: string[] = [];
  for (const blockToken of blockTokens) {
    switch (blockToken.type) {
      case "heading":
        paragraphs.push(withFullStop(inlineWords((blockToken as Tokens.Heading).tokens)));
        break;
      case "paragraph":
        paragraphs.push(inlineWords((blockToken as Tokens.Paragraph).tokens));
        break;
      case "text":
        paragraphs.push(textTokenWords(blockToken as Tokens.Text));
        break;
      case "blockquote":
        paragraphs.push(...blockSentences((blockToken as Tokens.Blockquote).tokens));
        break;
      case "list":
        paragraphs.push(listSentences(blockToken as Tokens.List));
        break;
      default:
        // code, table, html, hr, space and anything newer: not read aloud.
        break;
    }
  }
  return paragraphs;
}

function listSentences(listToken: Tokens.List): string {
  return listToken.items
    .map((listItem) => blockSentences(listItem.tokens).map(withFullStop).join(" "))
    .join(" ");
}

function textTokenWords(textToken: Tokens.Text): string {
  return textToken.tokens != null ? inlineWords(textToken.tokens) : textToken.text;
}

function inlineWords(inlineTokens: readonly Token[]): string {
  return inlineTokens.map(inlineTokenWords).join("");
}

function inlineTokenWords(inlineToken: Token): string {
  switch (inlineToken.type) {
    case "text":
      return textTokenWords(inlineToken as Tokens.Text);
    case "escape":
    case "codespan":
      return (inlineToken as Tokens.Escape | Tokens.Codespan).text;
    case "strong":
    case "em":
    case "del":
      return inlineWords((inlineToken as Tokens.Strong | Tokens.Em | Tokens.Del).tokens);
    case "link":
      return linkWords(inlineToken as Tokens.Link);
    case "br":
      return " ";
    default:
      // image and html: a picture is seen, not heard, and markup is not words.
      return "";
  }
}

const BARE_ADDRESS_RE = /^https?:\/\/\S+$/i;

/** A link is read by its words. When its words are nothing but the address
 * itself, the listener hears only the site ("example dot com"), not every
 * slash. Words that merely begin with an address are still words. */
function linkWords(linkToken: Tokens.Link): string {
  const words = inlineWords(linkToken.tokens);
  const trimmedWords = words.trim();
  if (!BARE_ADDRESS_RE.test(trimmedWords)) {
    return words;
  }
  try {
    return new URL(trimmedWords).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

function withFullStop(sentence: string): string {
  const trimmedSentence = sentence.trim();
  if (trimmedSentence.length === 0 || SENTENCE_END_RE.test(trimmedSentence)) {
    return trimmedSentence;
  }
  return `${trimmedSentence}.`;
}

function tidySpacing(paragraph: string): string {
  return paragraph.replace(/\s+/g, " ").replace(/\s+([.,!?;:])/g, "$1").trim();
}
