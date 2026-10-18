/*
 * Stable content keys.
 *
 * Two layers ask the same question — "is this the same block as before?" — and
 * they must agree on the answer: transcript diffing decides whether a rendered
 * node may be reused, and the diagram cache decides whether a finished SVG can be
 * re-installed instead of parsed again. One hash function keeps those two answers
 * consistent, which is why it lives here rather than beside either caller.
 */
(function () {
  "use strict";

  /** Non-cryptographic by design: keys identify content within this page only. */
  function contentHash(text) {
    var hash = 5381;
    for (var index = 0; index < text.length; index += 1) {
      hash = ((hash << 5) + hash + text.charCodeAt(index)) >>> 0;
    }
    return hash.toString(36);
  }

  window.__thintalkContentKey = { hash: contentHash };
})();