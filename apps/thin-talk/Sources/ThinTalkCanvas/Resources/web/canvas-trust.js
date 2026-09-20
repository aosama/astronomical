/*
 * The trust boundary for model text.
 *
 * Every string a model produced passes through this file before it reaches the
 * document, so the policy lives in exactly one place: what markup may exist, what
 * a link may point at, and what an image may load. Callers sanitise; they never
 * decide what is safe themselves, because two copies of a security rule drift.
 *
 * Math and diagram output deliberately do NOT pass through this allowlist: KaTeX
 * renders inline SVG and MathML, mermaid renders whole SVG documents, and admitting
 * either shape here would also admit model-authored SVG. Those two pipelines
 * sanitise separately (canvas-math.js, canvas-diagrams.js).
 */
(function () {
  "use strict";

  var SAFE_LINK_URI = /^(?:https?:|mailto:|thintalk-asset:|#)/i;
  var SAFE_IMAGE_URI = /^(?:thintalk-asset:|data:image\/|#)/i;

  var SANITIZE_OPTIONS = {
    ALLOWED_TAGS: [
      "p", "br", "hr", "strong", "em", "del", "code", "pre", "span",
      "h1", "h2", "h3", "h4", "h5", "h6",
      "ul", "ol", "li", "blockquote", "a", "img",
      "table", "thead", "tbody", "tr", "th", "td", "input", "details", "summary", "div"
    ],
    ALLOWED_ATTR: [
      "href", "title", "alt", "src", "class", "type", "checked", "disabled",
      "colspan", "rowspan", "start"
    ],
    ALLOWED_URI_REGEXP: /^(?:https?:|mailto:|thintalk-asset:|data:image\/|#)/i,
    FORBID_TAGS: [
      "script", "style", "iframe", "frame", "form", "button", "select", "textarea",
      "svg", "math", "object", "embed", "link", "meta", "base", "audio", "video",
      "source", "template"
    ],
    FORBID_ATTR: ["style", "srcset", "formaction", "ping", "onerror", "onload", "onclick"],
    KEEP_CONTENT: true
  };

  /**
   * Links may point at schemes a browser can open; images may only come from the
   * private asset scheme or inline image data. Separating the two keeps links
   * working while refusing remote images, which are a zero-click channel for
   * sending conversation content to a host the model chose.
   */
  DOMPurify.addHook("afterSanitizeAttributes", function (node) {
    if (node.nodeName === "A" && node.hasAttribute("href")) {
      var href = node.getAttribute("href") || "";
      if (!SAFE_LINK_URI.test(href)) {
        node.removeAttribute("href");
      } else {
        node.setAttribute("rel", "noopener noreferrer");
        node.setAttribute("target", "_blank");
      }
    }
    if (node.nodeName === "IMG") {
      var src = node.getAttribute("src") || "";
      if (!SAFE_IMAGE_URI.test(src)) {
        node.removeAttribute("src");
      }
    }
  });

  function sanitizeHtml(html) {
    return DOMPurify.sanitize(html, SANITIZE_OPTIONS);
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  window.__thintalkTrust = { sanitizeHtml: sanitizeHtml, escapeHtml: escapeHtml };
})();