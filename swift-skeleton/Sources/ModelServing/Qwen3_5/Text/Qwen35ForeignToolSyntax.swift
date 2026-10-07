import Foundation;

/// Normalizes non-canonical tool-call dialects into the Qwen3.5 function
/// grammar. Quantized models mix Claude-style attribute tags into Qwen output;
/// rewriting those tags here lets the function/parameter parser stay fail-open
/// instead of inventing a second grammar.
///
/// Mirrors crates/model-serving/src/qwen3_5/text/output_parser/foreign_syntax.rs.
enum Qwen35ForeignToolSyntax {

    static func normalizeForeignToolCallSyntax(_ toolCallBody: Array<UInt8>) -> Array<UInt8> {
        let withCanonicalFunctionOpen = normalizeInvokeFunctionOpen(toolCallBody);
        return rewriteAttributeParameterOpens(withCanonicalFunctionOpen);
    }

    /// Rewrites `<invoke name="f">` (or the envelope-less `invoke name="f"`)
    /// into `<function=f>`; bodies without a named attribute pass through.
    private static func normalizeInvokeFunctionOpen(
        _ toolCallBody: Array<UInt8>
    ) -> Array<UInt8> {
        let leadingWhitespaceBytes = toolCallBody.count - trimmedStart(toolCallBody).count;
        let trimmedBody = Array(toolCallBody[leadingWhitespaceBytes...]);
        let invokeStartBytes = Qwen35ByteText.bytes(Qwen35OutputParserMarkers.invokeStart);
        guard
            let afterInvokeOpen = Qwen35ByteText.stripPrefix(trimmedBody, invokeStartBytes)
                ?? Qwen35ByteText.stripPrefix(trimmedBody, Qwen35ByteText.bytes("invoke")),
            let extractedAttribute = extractNamedAttribute(afterInvokeOpen)
        else {
            return toolCallBody;
        }
        if extractedAttribute.attributeName.isEmpty {
            return toolCallBody;
        }
        var normalized: Array<UInt8> = Array(toolCallBody[0..<leadingWhitespaceBytes]);
        normalized += Qwen35ByteText.bytes(Qwen35OutputParserMarkers.bareFunctionStart);
        normalized += extractedAttribute.attributeName;
        normalized.append(UInt8(ascii: ">"));
        normalized += extractedAttribute.afterOpeningTag;
        return normalized;
    }

    private static func trimmedStart(_ source: Array<UInt8>) -> Array<UInt8> {
        var startOffset = 0;
        while startOffset < source.count
            && Qwen35ByteText.asciiWhitespace.contains(source[startOffset])
        {
            startOffset += 1;
        }
        return Array(source[startOffset...]);
    }

    /// Extracts `name="value"` (single or double quoted) and the text after the
    /// optional closing `>` of the opening tag. Searches for the `name=`
    /// keyword first, mirroring the invoke-open rewrite path.
    private static func extractNamedAttribute(
        _ afterTagName: Array<UInt8>
    ) -> (attributeName: Array<UInt8>, afterOpeningTag: Array<UInt8>)? {
        let nameKeyword = Qwen35ByteText.bytes("name=");
        guard let nameKeywordOffset = Qwen35ByteText.firstOffset(of: nameKeyword, in: afterTagName)
        else {
            return nil;
        }
        let afterNameKeyword = Array(afterTagName[(nameKeywordOffset + nameKeyword.count)...]);
        return extractNamedAttributeValue(afterNameKeyword);
    }

    /// Parses a quoted attribute value directly after `name=`.
    private static func extractNamedAttributeValue(
        _ afterNameKeyword: Array<UInt8>
    ) -> (attributeName: Array<UInt8>, afterOpeningTag: Array<UInt8>)? {
        guard let firstQuoteByte = afterNameKeyword.first,
            firstQuoteByte == UInt8(ascii: "\"") || firstQuoteByte == UInt8(ascii: "'")
        else {
            return nil;
        }
        let searchBody = Array(afterNameKeyword[1...]);
        guard let closingQuoteOffset = Qwen35ByteText.firstOffset(
            of: [firstQuoteByte], in: searchBody)
        else {
            return nil;
        }
        let attributeName = Array(searchBody[0..<closingQuoteOffset]);
        let afterQuotedName = Array(searchBody[(closingQuoteOffset + 1)...]);
        let afterOpeningTag = Qwen35ByteText.stripPrefix(
            afterQuotedName, [UInt8(ascii: ">")]) ?? afterQuotedName;
        return (attributeName, afterOpeningTag);
    }

    /// Rewrites every `<parameter name="k">` / `parameter name="k">` open into
    /// the canonical `<parameter=k>`.
    private static func rewriteAttributeParameterOpens(
        _ toolCallBody: Array<UInt8>
    ) -> Array<UInt8> {
        var normalized: Array<UInt8> = [];
        var remaining = toolCallBody;
        while let parameterOpenOffset = nextAttributeParameterOpenOffset(remaining) {
            normalized += Array(remaining[0..<parameterOpenOffset]);
            let afterOpen = Array(remaining[parameterOpenOffset...]);
            if let rewrittenOpen = consumeAttributeParameterOpen(afterOpen) {
                normalized += Qwen35ByteText.bytes(Qwen35OutputParserMarkers.bareParameterStart);
                normalized += rewrittenOpen.parameterName;
                normalized.append(UInt8(ascii: ">"));
                remaining = rewrittenOpen.afterOpeningTag;
            } else {
                normalized.append(remaining[parameterOpenOffset]);
                remaining = Array(remaining[(parameterOpenOffset + 1)...]);
            }
        }
        normalized += remaining;
        return normalized;
    }

    /// Offset of the next attribute-style parameter open, tolerating a missing
    /// `<` exactly like the Rust scanner.
    private static func nextAttributeParameterOpenOffset(_ remaining: Array<UInt8>) -> Int? {
        let parameterKeyword = Qwen35ByteText.bytes("parameter");
        let nameKeyword = Qwen35ByteText.bytes("name=");
        var searchOffset = 0;
        while searchOffset < remaining.count {
            let haystack = Array(remaining[searchOffset...]);
            guard let relativeOffset = Qwen35ByteText.firstOffset(of: parameterKeyword, in: haystack)
            else {
                return nil;
            }
            let absoluteOffset = searchOffset + relativeOffset;
            let hasOpeningBracket = absoluteOffset > 0
                && remaining[absoluteOffset - 1] == UInt8(ascii: "<");
            let markerStart = hasOpeningBracket ? absoluteOffset - 1 : absoluteOffset;
            let afterMarker = Array(remaining[(absoluteOffset + parameterKeyword.count)...]);
            let afterWhitespace = trimmedStart(afterMarker);
            if Qwen35ByteText.stripPrefix(afterWhitespace, nameKeyword) != nil {
                return markerStart;
            }
            searchOffset = absoluteOffset + 1;
        }
        return nil;
    }

    private static func consumeAttributeParameterOpen(
        _ afterOpen: Array<UInt8>
    ) -> (parameterName: Array<UInt8>, afterOpeningTag: Array<UInt8>)? {
        let afterOptionalBracket = Qwen35ByteText.stripPrefix(
            afterOpen, [UInt8(ascii: "<")]) ?? afterOpen;
        let afterParameterWord = Qwen35ByteText.stripPrefix(
            afterOptionalBracket, Qwen35ByteText.bytes("parameter"));
        guard let afterParameter = afterParameterWord else {
            return nil;
        }
        let afterWhitespace = trimmedStart(afterParameter);
        let nameKeyword = Qwen35ByteText.bytes("name=");
        guard let afterNameKeyword = Qwen35ByteText.stripPrefix(afterWhitespace, nameKeyword) else {
            return nil;
        }
        guard let extractedValue = extractNamedAttributeValue(afterNameKeyword) else {
            return nil;
        }
        return (parameterName: extractedValue.attributeName,
                afterOpeningTag: extractedValue.afterOpeningTag);
    }
}
