import Foundation;

/// Escapes untrusted text that shares a Qwen prompt with reserved control
/// markers: ordinary less-than signs pass through while every reserved Qwen
/// marker prefix is entity-escaped so seeded content cannot close or open a
/// control block.
///
/// Mirrors crates/model-serving/src/qwen3_5/text/template_safe_content.rs.
enum Qwen35TemplateSafeContent {

    static func escapedContent(_ untrustedContent: String) -> String {
        var escapedContent = "";
        appendTemplateSafeContent(&escapedContent, untrustedContent);
        return escapedContent;
    }

    static func appendTemplateSafeContent(
        _ renderedPrompt: inout String, _ untrustedContent: String
    ) {
        var remainingContent = Substring(untrustedContent);
        while let markerOffset = remainingContent.firstIndex(of: "<") {
            renderedPrompt += remainingContent[remainingContent.startIndex..<markerOffset];
            let markerSuffix = remainingContent[remainingContent.index(after: markerOffset)...];
            if Qwen35TemplateSafeContent.startsReservedTemplateMarker(markerSuffix) {
                renderedPrompt += "&lt;";
            } else {
                renderedPrompt += "<";
            }
            remainingContent = markerSuffix;
        }
        renderedPrompt += remainingContent;
    }

    private static func startsReservedTemplateMarker(_ markerSuffix: Substring) -> Bool {
        let reservedMarkerPrefixes: Array<String> = [
            "|", "think>", "/think>", "tool_call>", "/tool_call>",
            "tool_response>", "/tool_response>", "tools>", "/tools>",
            "function=", "/function>", "parameter=", "/parameter>",
        ];
        return reservedMarkerPrefixes.contains { (reservedPrefix: String) -> Bool in
            markerSuffix.hasPrefix(reservedPrefix);
        };
    }
}
