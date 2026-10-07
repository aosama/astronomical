import Foundation;

/// Control-marker vocabulary and scanning primitives for the Qwen3.5 output
/// parser. All scanning runs over UTF-8 bytes because the byte budgets
/// (16 KB fragment cap, 128 KB pending cap) and the Rust reference parser are
/// byte-addressed; marker literals are ASCII, so byte offsets and string
/// content stay consistent.
enum Qwen35OutputParserMarkers {

    static let thinkStart = qwenMarker("<", "think", ">");
    static let thinkEnd = qwenMarker("<", "/think", ">");
    static let toolCallStart = qwenMarker("<", "tool_call", ">");
    static let toolCallEnd = qwenMarker("<", "/tool_call", ">");
    static let bareFunctionStart = qwenMarker("<", "function=");
    static let bareParameterStart = qwenMarker("<", "parameter=");
    static let functionEnd = qwenMarker("<", "/function", ">");
    static let invokeStart = qwenMarker("<", "invoke");
    static let invokeEnd = qwenMarker("<", "/invoke", ">");

    static let textScanMarkers: Array<Array<UInt8>> = [
        Array(thinkStart.utf8), Array(toolCallStart.utf8),
        Array(bareFunctionStart.utf8), Array(invokeStart.utf8),
    ];

    /// mlx-lm transitions from reasoning to tool on a tool-call start. Ornith
    /// often writes the call inside the still-open think channel; scanning only
    /// for the think end marker would dump that call as reasoning text and end
    /// the turn.
    static let reasoningScanMarkers: Array<Array<UInt8>> = [
        Array(thinkEnd.utf8), Array(toolCallStart.utf8),
        Array(bareFunctionStart.utf8), Array(invokeStart.utf8),
    ];

    /// Assembles marker literals from fragments so the repository source never
    /// spells a live control marker that downstream scanners could trip over.
    private static func qwenMarker(_ fragments: String...) -> String {
        return fragments.joined();
    }
}

/// The parser's channel states. The tool-call state carries the entered
/// attempt so quoted-prose resync and end-of-stream salvage can reconstruct
/// the buffered body.
enum Qwen35OutputParserState: Equatable {
    case text;
    case reasoning;
    case toolCall(Qwen35ToolCallEntry);
    case suppressedLateReasoning;

    /// Only the channel states that retain arbitrary model text drain at the
    /// bounded pending-output cap; a buffered tool call must never flush
    /// because its bytes are argument payload.
    var requiresMarkerScanPendingOutputCap: Bool {
        switch self {
        case .text, .reasoning, .suppressedLateReasoning: return true;
        case .toolCall: return false;
        }
    }
}

/// One entered tool-call attempt: its dialect kind, the exact opener marker to
/// restore when the attempt turns out to be quoted prose, and the channel the
/// opener appeared in.
struct Qwen35ToolCallEntry: Equatable {
    let kind: Qwen35ToolCallEntryKind;
    let opener: Array<UInt8>;
    let openedInReasoning: Bool;
}

enum Qwen35ToolCallEntryKind: Equatable {
    case envelope;
    case bareFunction;
    case invokeTag;
}

/// Shared UTF-8 byte scanning utilities for the parser and its tool-schema
/// reader. The ASCII whitespace set stands in for Rust `char::is_whitespace`;
/// function and parameter names are ASCII in practice.
enum Qwen35ByteText {

    static let asciiWhitespace: Set<UInt8> = [0x20, 0x09, 0x0A, 0x0B, 0x0C, 0x0D];

    static func bytes(_ source: String) -> Array<UInt8> {
        return Array(source.utf8);
    }

    static func text(_ source: ArraySlice<UInt8>) -> String {
        return String(decoding: source, as: UTF8.self);
    }

    /// First byte offset of any ASCII-whitespace byte, or `nil`.
    static func firstWhitespaceOffset(in source: Array<UInt8>) -> Int? {
        for (byteOffset, byte) in source.enumerated() where asciiWhitespace.contains(byte) {
            return byteOffset;
        }
        return nil;
    }

    static func trimStartMatches(_ source: Array<UInt8>, byte: UInt8) -> Array<UInt8> {
        var startOffset = 0;
        while startOffset < source.count && source[startOffset] == byte {
            startOffset += 1;
        }
        return Array(source[startOffset...]);
    }

    static func trimmed(_ source: Array<UInt8>) -> Array<UInt8> {
        var startOffset = 0;
        var endOffset = source.count;
        while startOffset < endOffset && asciiWhitespace.contains(source[startOffset]) {
            startOffset += 1;
        }
        while endOffset > startOffset && asciiWhitespace.contains(source[endOffset - 1]) {
            endOffset -= 1;
        }
        return Array(source[startOffset..<endOffset]);
    }

    static func stripPrefix(_ source: Array<UInt8>, _ prefix: Array<UInt8>) -> Array<UInt8>? {
        guard source.count >= prefix.count, Array(source[0..<prefix.count]) == prefix else {
            return nil;
        }
        return Array(source[prefix.count...]);
    }

    static func stripSuffix(_ source: Array<UInt8>, _ suffix: Array<UInt8>) -> Array<UInt8>? {
        guard source.count >= suffix.count,
            Array(source[(source.count - suffix.count)...]) == suffix
        else {
            return nil;
        }
        return Array(source[0..<(source.count - suffix.count)]);
    }

    /// Offsets of every occurrence of a non-empty needle.
    static func offsets(of needle: Array<UInt8>, in source: Array<UInt8>) -> Array<Int> {
        guard !needle.isEmpty && source.count >= needle.count else {
            return [];
        }
        var foundOffsets: Array<Int> = [];
        var searchOffset = 0;
        while searchOffset + needle.count <= source.count {
            if Array(source[searchOffset..<(searchOffset + needle.count)]) == needle {
                foundOffsets.append(searchOffset);
                searchOffset += 1;
            } else {
                searchOffset += 1;
            }
        }
        return foundOffsets;
    }

    static func firstOffset(of needle: Array<UInt8>, in source: Array<UInt8>) -> Int? {
        return offsets(of: needle, in: source).first;
    }

    /// Earliest marker occurrence across the pending text; ties keep the first
    /// marker in the caller's list, matching Rust `min_by_key` stability.
    static func earliestMarker(
        in text: Array<UInt8>, markers: Array<Array<UInt8>>
    ) -> (byteOffset: Int, marker: Array<UInt8>)? {
        var earliest: (byteOffset: Int, marker: Array<UInt8>)? = nil;
        for marker in markers {
            guard let markerOffset = firstOffset(of: marker, in: text) else {
                continue;
            }
            if earliest == nil || markerOffset < earliest!.byteOffset {
                earliest = (markerOffset, marker);
            }
        }
        return earliest;
    }

    /// Length in bytes of the longest pending suffix that is a strict prefix
    /// of any marker, so partial markers stay buffered across chunk bounds.
    /// Splits that would cut a multibyte character are skipped, mirroring the
    /// Rust char-boundary guard.
    static func longestSuffixPrefixForMarkers(
        text: Array<UInt8>, markers: Array<Array<UInt8>>
    ) -> Int {
        let maximumMarkerPrefixBytes = markers.map { (marker: Array<UInt8>) -> Int in
            max(marker.count - 1, 0)
        }.max() ?? 0;
        let maximumPrefixBytes = min(maximumMarkerPrefixBytes, text.count);
        if maximumPrefixBytes == 0 {
            return 0;
        }
        var suffixBytes = maximumPrefixBytes;
        while suffixBytes >= 1 {
            let suffixStart = text.count - suffixBytes;
            if isCharBoundary(text, byteOffset: suffixStart) {
                let suffix = Array(text[suffixStart...]);
                if markers.contains(where: { (marker: Array<UInt8>) -> Bool in
                    marker.count >= suffix.count && Array(marker[0..<suffix.count]) == suffix
                }) {
                    return suffixBytes;
                }
            }
            suffixBytes -= 1;
        }
        return 0;
    }

    static func isCharBoundary(_ source: Array<UInt8>, byteOffset: Int) -> Bool {
        guard byteOffset > 0 && byteOffset < source.count else {
            return true;
        }
        return source[byteOffset] & 0xC0 != 0x80;
    }
}

/// Splits a normalized tool-call body into the function name and the raw
/// parameter content. Closed envelopes still reach the harness when the model
/// drops `<` or the function close tag, so both defect shapes parse here.
func splitQwenFunctionEnvelope(
    toolCallBody: Array<UInt8>
) throws -> (functionName: Array<UInt8>, parameterContent: Array<UInt8>) {
    let bareFunctionStartBytes = Qwen35ByteText.bytes(Qwen35OutputParserMarkers.bareFunctionStart);
    guard
        let afterFunctionOpen = Qwen35ByteText.stripPrefix(toolCallBody, bareFunctionStartBytes)
            ?? Qwen35ByteText.stripPrefix(toolCallBody, Qwen35ByteText.bytes("function="))
    else {
        throw Qwen35OutputParserError.toolCallMissingFunction;
    }
    let functionNameEnd = afterFunctionOpen.firstIndex(where: { (bodyByte: UInt8) -> Bool in
        bodyByte == UInt8(ascii: ">") || bodyByte == UInt8(ascii: "<")
            || Qwen35ByteText.asciiWhitespace.contains(bodyByte)
    }) ?? afterFunctionOpen.count;
    let functionName = Qwen35ByteText.trimmed(Array(afterFunctionOpen[0..<functionNameEnd]));
    if functionName.isEmpty {
        throw Qwen35OutputParserError.toolCallMissingFunction;
    }
    var afterFunctionName = Qwen35ByteText.trimmed(
        Qwen35ByteText.trimStartMatches(
            Array(afterFunctionOpen[functionNameEnd...]), byte: UInt8(ascii: ">")));
    if let withoutFunctionEnd = Qwen35ByteText.stripSuffix(
        afterFunctionName, Qwen35ByteText.bytes(Qwen35OutputParserMarkers.functionEnd))
    {
        afterFunctionName = Qwen35ByteText.trimmed(withoutFunctionEnd);
    }
    return (functionName, afterFunctionName);
}
