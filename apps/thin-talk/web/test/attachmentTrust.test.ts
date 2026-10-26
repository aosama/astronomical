import assert from "node:assert/strict";
import { test } from "node:test";
import { TrustBoundary } from "../src/TrustBoundary";
import { MathPipeline } from "../src/MathPipeline";
import { DiagramPipeline } from "../src/DiagramPipeline";
import { Renderer } from "../src/Renderer";
import { TranscriptMessage } from "../src/TranscriptMessage";
import { TranscriptAttachment } from "../src/TranscriptAttachment";
import { ChatRole } from "../src/ChatRole";
import { ChatMessageState } from "../src/ChatMessageState";

// The attachment strip builds its own markup instead of passing through
// sanitizeHtml, so its image sources must be decided by the trust boundary's
// single image-scheme rule — the same rule markdown images answer to.

const trust = new TrustBoundary();
const renderer = new Renderer({ trust, math: new MathPipeline(trust), diagrams: new DiagramPipeline() });

function messageHtmlFor(attachments: TranscriptAttachment[]): string {
  return renderer.messageHtml(
    new TranscriptMessage({
      id: "attachment-message-id",
      role: ChatRole.ASSISTANT,
      markdown: "",
      reasoning: "",
      state: ChatMessageState.COMPLETE,
      attachments,
    }),
  );
}

test("should_render_an_attachment_whose_source_is_the_private_asset_scheme", () => {
  const html = messageHtmlFor([
    new TranscriptAttachment({ id: "asset-private", assetURL: "thintalk-asset://attachment/2f1c9a", label: "figure" }),
  ]);
  assert.match(html, /src="thintalk-asset:\/\/attachment\/2f1c9a"/);
});

test("should_refuse_an_attachment_image_from_a_remote_or_disallowed_source", () => {
  const html = messageHtmlFor([
    new TranscriptAttachment({ id: "asset-remote", assetURL: "https://tracker.example/pixel.png", label: "tracker" }),
  ]);
  assert.ok(!html.includes("tracker.example"), "a disallowed image source must not reach an img element");
  assert.ok(!html.includes("<img"), "no image element may render for a disallowed source");
  assert.match(html, /tracker/, "the attachment keeps its readable label instead");
});

test("should_allow_inline_image_data_as_an_attachment_source", () => {
  const html = messageHtmlFor([
    new TranscriptAttachment({ id: "asset-inline", assetURL: "data:image/png;base64,AAAA", label: "inline" }),
  ]);
  assert.match(html, /src="data:image\/png;base64,AAAA"/);
});
