/*
 * Math spans (KaTeX).
 *
 * Marked would mangle LaTeX (`x_1` becomes emphasis) and the markdown allowlist
 * forbids the SVG and MathML that KaTeX legitimately renders, so maths takes a
 * different road: it is carved out of the source before markdown parsing, the
 * remaining prose is sanitised as normal, and KaTeX output is grafted into the
 * already-sanitised HTML. This file owns that round trip; the caller owns when to
 * sanitise in between.
 */
(function () {
  "use strict";

  var trust = window.__thintalkTrust;

  var MATH_BLOCK_RE = /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]/g;
  var MATH_INLINE_RE = /\$([^$\n]+?)\$|\\\(([\s\S]+?)\\\)/g;
  var CODE_SEGMENT_RE = /(```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`[^`\n]*`)/g;

  /**
   * Slot numbers come from a collection rebuilt for every call, so even a model
   * that emits literal "%%MATH0%%" text can only splice its own maths into another
   * slot of its own message, never across a message boundary or into the bridge.
   */
  function mathToken(slotIndex) {
    return "%%MATH" + slotIndex + "%%";
  }

  function isEmptyMathCandidate(text) {
    return !text || !text.trim();
  }

  /**
   * "$x^2$" is a formula, "$5-$10" is money. A span counts as TeX only when it
   * carries TeX signatures (commands, superscripts, braces) or is a single letter
   * or Greek character, which keeps prices and prose dollar signs as text.
   */
  function looksLikeTeX(tex) {
    if (tex.length === 1 && /[a-zA-Z\u0370-\u03ff]/.test(tex)) {
      return true;
    }
    return /[\\^_{}]/.test(tex);
  }

  function sliceMathSpans(prose, collectedMathSpans) {
    return prose
      .replace(MATH_BLOCK_RE, function (match, dollars, brackets) {
        var tex = (dollars || brackets || "").trim();
        if (isEmptyMathCandidate(tex)) {
          return match;
        }
        collectedMathSpans.push({ text: tex, displayMode: true });
        return mathToken(collectedMathSpans.length - 1);
      })
      .replace(MATH_INLINE_RE, function (match, dollars, brackets) {
        var tex = (dollars || brackets || "").trim();
        if (isEmptyMathCandidate(tex) || !looksLikeTeX(tex)) {
          return match;
        }
        collectedMathSpans.push({ text: tex, displayMode: false });
        return mathToken(collectedMathSpans.length - 1);
      });
  }

  /**
   * Fenced code, tilde fences, and inline code are split out first and left
   * verbatim, so a code example keeps its dollar signs instead of losing them to
   * the maths pipeline.
   */
  function tokenize(source, collectedMathSpans) {
    var segments = source.split(CODE_SEGMENT_RE);
    var rebuiltMarkdown = [];
    for (var segmentIndex = 0; segmentIndex < segments.length; segmentIndex += 1) {
      if (segmentIndex % 2 === 1) {
        // Odd segments are the code the split captured: untouched.
        rebuiltMarkdown.push(segments[segmentIndex]);
        continue;
      }
      rebuiltMarkdown.push(sliceMathSpans(segments[segmentIndex], collectedMathSpans));
    }
    return rebuiltMarkdown.join("");
  }

  function katexHtml(tex, displayMode) {
    if (!window.katex || typeof window.katex.renderToString !== "function") {
      return "";
    }
    try {
      return window.katex.renderToString(tex, {
        displayMode: displayMode,
        output: "htmlAndMathml",
        strict: "ignore",
        trust: false,
        throwOnError: false,
        maxSize: 2000
      });
    } catch (error) {
      return "";
    }
  }

  /** Grafts KaTeX output into sanitised prose, failing closed to the raw token. */
  function graft(sanitisedHtml, collectedMathSpans) {
    var html = sanitisedHtml;
    collectedMathSpans.forEach(function (mathSpan, slotIndex) {
      var token = mathToken(slotIndex);
      var rendered = katexHtml(mathSpan.text, mathSpan.displayMode);
      html = html.split(token).join(rendered || trust.escapeHtml(token));
    });
    return html;
  }

  window.__thintalkMath = { tokenize: tokenize, graft: graft };
})();