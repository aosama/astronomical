import { TrustBoundary } from "./TrustBoundary";

/** One carved-out maths span waiting to be rendered and grafted back. */
export interface MathSpan {
  text: string;
  displayMode: boolean;
}

const MATH_BLOCK_RE = /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]/g;
const MATH_INLINE_RE = /\$([^$\n]+?)\$|\\\(([\s\S]+?)\\\)/g;
const CODE_SEGMENT_RE = /(```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`[^`\n]*`)/g;

/**
 * Math spans (KaTeX).
 *
 * Marked would mangle LaTeX (`x_1` becomes emphasis) and the markdown allowlist
 * forbids the SVG and MathML that KaTeX legitimately renders, so maths takes a
 * different road: it is carved out of the source before markdown parsing, the
 * remaining prose is sanitised as normal, and KaTeX output is grafted into the
 * already-sanitised HTML. This class owns that round trip; the caller owns when
 * to sanitise in between.
 */
export class MathPipeline {
  private readonly trust: TrustBoundary;

  public constructor(trust: TrustBoundary) {
    this.trust = trust;
  }

  /** Carves maths spans out of the source, replacing each with a slot token and
   * collecting the spans in order for the later graft. */
  public tokenize(source: string, collectedMathSpans: MathSpan[]): string {
    const segments = source.split(CODE_SEGMENT_RE);
    const rebuiltMarkdown: string[] = [];
    for (let segmentIndex = 0; segmentIndex < segments.length; segmentIndex += 1) {
      const segment = segments[segmentIndex];
      if (segment === undefined) {
        continue;
      }
      if (segmentIndex % 2 === 1) {
        // Odd segments are the code the split captured: untouched.
        rebuiltMarkdown.push(segment);
        continue;
      }
      rebuiltMarkdown.push(this.sliceMathSpans(segment, collectedMathSpans));
    }
    return rebuiltMarkdown.join("");
  }

  /** Grafts KaTeX output into sanitised prose, failing closed to the raw token. */
  public graft(sanitisedHtml: string, collectedMathSpans: MathSpan[]): string {
    let html = sanitisedHtml;
    collectedMathSpans.forEach((mathSpan, slotIndex) => {
      const token = MathPipeline.mathToken(slotIndex);
      const rendered = MathPipeline.katexHtml(mathSpan.text, mathSpan.displayMode);
      html = html.split(token).join(rendered || this.trust.escapeHtml(token));
    });
    return html;
  }

  /**
   * Slot numbers come from a collection rebuilt for every call, so even a model
   * that emits literal "%%MATH0%%" text can only splice its own maths into another
   * slot of its own message, never across a message boundary or into the bridge.
   */
  private static mathToken(slotIndex: number): string {
    return `%%MATH${slotIndex}%%`;
  }

  private static isEmptyMathCandidate(text: string): boolean {
    return text.trim().length === 0;
  }

  /**
   * "$x^2$" is a formula, "$5-$10" is money. A span counts as TeX only when it
   * carries TeX signatures (commands, superscripts, braces) or is a single letter
   * or Greek character, which keeps prices and prose dollar signs as text.
   */
  private static looksLikeTeX(tex: string): boolean {
    if (tex.length === 1 && /[a-zA-Z\u0370-\u03ff]/.test(tex)) {
      return true;
    }
    return /[\\^_{}]/.test(tex);
  }

  private sliceMathSpans(prose: string, collectedMathSpans: MathSpan[]): string {
    return prose
      .replace(MATH_BLOCK_RE, (match: string, dollars: string, brackets: string) => {
        const tex = (dollars || brackets || "").trim();
        if (MathPipeline.isEmptyMathCandidate(tex)) {
          return match;
        }
        collectedMathSpans.push({ text: tex, displayMode: true });
        return MathPipeline.mathToken(collectedMathSpans.length - 1);
      })
      .replace(MATH_INLINE_RE, (match: string, dollars: string, brackets: string) => {
        const tex = (dollars || brackets || "").trim();
        if (MathPipeline.isEmptyMathCandidate(tex) || !MathPipeline.looksLikeTeX(tex)) {
          return match;
        }
        collectedMathSpans.push({ text: tex, displayMode: false });
        return MathPipeline.mathToken(collectedMathSpans.length - 1);
      });
  }

  private static katexHtml(tex: string, displayMode: boolean): string {
    const katex = window.katex;
    if (!katex || typeof katex.renderToString !== "function") {
      return "";
    }
    try {
      return katex.renderToString(tex, {
        displayMode,
        output: "htmlAndMathml",
        strict: "ignore",
        trust: false,
        throwOnError: false,
        maxSize: 2000,
      });
    } catch {
      return "";
    }
  }
}
