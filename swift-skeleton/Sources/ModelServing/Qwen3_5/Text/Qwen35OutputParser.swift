import Foundation;

import IpcProtocol;

/// Bounded incremental parser for Qwen3.5 reasoning and XML-style function
/// output. Tool-call defects fail open so a coding client can reject or retry;
/// incomplete control-marker prefixes flush as visible text at generation end;
/// tool-call markers quoted inside model prose restore to their origin channel
/// instead of hijacking the state machine.
///
/// Mirrors crates/model-serving/src/qwen3_5/text/output_parser.rs with the
/// salvage path from output_parser/salvage.rs. Dense and MoE share it.
public final class Qwen35OutputParser {

    private static let maximumOutputFragmentBytes = 16 * 1024;
    private static let maximumMarkerScanPendingOutputBytes = 128 * 1024;

    private var completedToolCallCount: UInt16 = 0;
    private var declaredTools: Dictionary<String, Qwen35DeclaredTool> = [:];
    private var hasStreamedVisibleText = false;
    private var pendingOutput: Array<UInt8> = [];
    private var state: Qwen35OutputParserState;

    /// Creates an output parser from the exact tools declared for one request.
    /// Declared schemas are validated up front so a broken declaration surfaces
    /// as a typed rejection naming the offending tool, never mid-generation.
    public convenience init(declaredTools: Array<ChatToolDefinition>) throws {
        try self.init(declaredTools: declaredTools, startsInsideThinking: false);
    }

    /// Creates a parser for generation that continues after the prompt's think
    /// prefix: the first decoded bytes belong to the reasoning channel.
    public convenience init(
        declaredTools: Array<ChatToolDefinition>, startsInsideThinking: Bool
    ) throws {
        try self.init(
            declaredTools: declaredTools,
            initialState: startsInsideThinking ? .reasoning : .text);
    }

    init(
        declaredTools: Array<ChatToolDefinition>, initialState: Qwen35OutputParserState
    ) throws {
        var toolsByName: Dictionary<String, Qwen35DeclaredTool> = [:];
        for toolDefinition in declaredTools {
            if toolsByName[toolDefinition.name] != nil {
                throw Qwen35OutputParserError.duplicateDeclaredTool(
                    functionName: toolDefinition.name);
            }
            toolsByName[toolDefinition.name] = try Qwen35DeclaredTool(
                toolDefinition: toolDefinition);
        }
        self.declaredTools = toolsByName;
        self.state = initialState;
    }

    /// Processes one decoded text fragment and emits only stable structured
    /// content.
    public func pushFragment(_ decodedFragment: String) throws -> Array<Qwen35OutputEvent> {
        let fragmentBytes = Qwen35ByteText.bytes(decodedFragment);
        if fragmentBytes.count > Qwen35OutputParser.maximumOutputFragmentBytes {
            if case .toolCall = self.state {
                // The fragment cannot be buffered. Salvage the pending call so a
                // usable name still reaches the harness instead of aborting.
                return self.salvageUnclosedToolCall();
            }
            var outputEvents = self.flushPendingAsVisibleDelta();
            if let fragmentEvent = self.visibleDeltaForCurrentState(fragmentBytes) {
                outputEvents.append(fragmentEvent);
            }
            return outputEvents;
        }
        let pendingOutputBytes = self.pendingOutput.count + fragmentBytes.count;
        if self.state.requiresMarkerScanPendingOutputCap
            && pendingOutputBytes > Qwen35OutputParser.maximumMarkerScanPendingOutputBytes
        {
            // Drain retained text so memory stays bounded. Generation continues.
            var outputEvents = self.flushPendingAsVisibleDelta();
            self.pendingOutput += fragmentBytes;
            while try self.advance(into: &outputEvents) {}
            return outputEvents;
        }
        self.pendingOutput += fragmentBytes;

        var outputEvents: Array<Qwen35OutputEvent> = [];
        while try self.advance(into: &outputEvents) {}
        return outputEvents;
    }

    /// Completes the stream. Unclosed tool calls salvage; leftover marker
    /// prefixes flush as text.
    public func finish() -> Array<Qwen35OutputEvent> {
        switch self.state {
        case .text, .reasoning:
            return self.flushPendingAsVisibleDelta();
        case .toolCall:
            return self.salvageUnclosedToolCall();
        case .suppressedLateReasoning:
            self.pendingOutput = [];
            return [];
        }
    }

    private func advance(
        into outputEvents: inout Array<Qwen35OutputEvent>
    ) throws -> Bool {
        switch self.state {
        case .text:
            return self.advanceText(into: &outputEvents);
        case .reasoning:
            return self.advanceReasoning(into: &outputEvents);
        case let .toolCall(entry):
            return self.advanceToolCall(entry, into: &outputEvents);
        case .suppressedLateReasoning:
            return self.advanceSuppressedLateReasoning();
        }
    }

    private func advanceText(into outputEvents: inout Array<Qwen35OutputEvent>) -> Bool {
        let textScanMarkers = Qwen35OutputParserMarkers.textScanMarkers;
        if let markerMatch = Qwen35ByteText.earliestMarker(
            in: self.pendingOutput, markers: textScanMarkers)
        {
            if markerMatch.byteOffset > 0 {
                self.hasStreamedVisibleText = true;
                outputEvents.append(.textDelta(self.takePendingPrefix(markerMatch.byteOffset)));
                return true;
            }
            _ = self.takePendingPrefix(markerMatch.marker.count);
            self.enterStateAfterStartMarker(markerMatch.marker, openedInReasoning: false);
            return true;
        }
        let stableTextBytes = self.pendingOutput.count
            - Qwen35ByteText.longestSuffixPrefixForMarkers(
                text: self.pendingOutput, markers: textScanMarkers);
        if stableTextBytes == 0 {
            return false;
        }
        self.hasStreamedVisibleText = true;
        outputEvents.append(.textDelta(self.takePendingPrefix(stableTextBytes)));
        return true;
    }

    private func advanceReasoning(into outputEvents: inout Array<Qwen35OutputEvent>) -> Bool {
        let reasoningScanMarkers = Qwen35OutputParserMarkers.reasoningScanMarkers;
        if let markerMatch = Qwen35ByteText.earliestMarker(
            in: self.pendingOutput, markers: reasoningScanMarkers)
        {
            if markerMatch.byteOffset > 0 {
                outputEvents.append(
                    .reasoningDelta(self.takePendingPrefix(markerMatch.byteOffset)));
                return true;
            }
            _ = self.takePendingPrefix(markerMatch.marker.count);
            if markerMatch.marker == Qwen35ByteText.bytes(Qwen35OutputParserMarkers.thinkEnd) {
                self.state = .text;
            } else {
                self.enterStateAfterStartMarker(markerMatch.marker, openedInReasoning: true);
            }
            return true;
        }
        let stableReasoningBytes = self.pendingOutput.count
            - Qwen35ByteText.longestSuffixPrefixForMarkers(
                text: self.pendingOutput, markers: reasoningScanMarkers);
        if stableReasoningBytes == 0 {
            return false;
        }
        outputEvents.append(.reasoningDelta(self.takePendingPrefix(stableReasoningBytes)));
        return true;
    }

    private func advanceSuppressedLateReasoning() -> Bool {
        let thinkEndBytes = Qwen35ByteText.bytes(Qwen35OutputParserMarkers.thinkEnd);
        if let markerIndex = Qwen35ByteText.firstOffset(of: thinkEndBytes, in: self.pendingOutput) {
            let hiddenReasoningAndMarkerBytes = markerIndex + thinkEndBytes.count;
            _ = self.takePendingPrefix(hiddenReasoningAndMarkerBytes);
            self.state = .text;
            return true;
        }
        let stableHiddenReasoningBytes = self.pendingOutput.count
            - Qwen35ByteText.longestSuffixPrefixForMarkers(
                text: self.pendingOutput, markers: [thinkEndBytes]);
        if stableHiddenReasoningBytes == 0 {
            return false;
        }
        _ = self.takePendingPrefix(stableHiddenReasoningBytes);
        return true;
    }

    private func advanceToolCall(
        _ entry: Qwen35ToolCallEntry, into outputEvents: inout Array<Qwen35OutputEvent>
    ) -> Bool {
        let endMarkerMatch = Qwen35ByteText.earliestMarker(
            in: self.pendingOutput, markers: Qwen35OutputParserSalvage.toolCallEndMarkers(entry.kind));
        let thinkEndBytes = Qwen35ByteText.bytes(Qwen35OutputParserMarkers.thinkEnd);
        let spuriousThinkCloseIndex = Qwen35ByteText.firstOffset(
            of: thinkEndBytes, in: self.pendingOutput);
        let openerWasQuotedProse: Bool;
        switch (endMarkerMatch, spuriousThinkCloseIndex) {
        case let (endMatch?, thinkCloseIndex?):
            openerWasQuotedProse = thinkCloseIndex < endMatch.byteOffset;
        case (.none, .some):
            openerWasQuotedProse = true;
        default:
            openerWasQuotedProse = false;
        }
        if openerWasQuotedProse {
            self.resyncSpuriousToolCall(entry, into: &outputEvents);
            return true;
        }
        guard let endMatch = endMarkerMatch else {
            return false;
        }
        let remainingBody = self.takePendingPrefixBytes(endMatch.byteOffset);
        _ = self.takePendingPrefix(endMatch.marker.count);
        if endMatch.marker != Qwen35ByteText.bytes(Qwen35OutputParserMarkers.toolCallEnd) {
            self.consumeTrailingEnvelopeCloseIfPresent();
        }
        let reconstructedBody = Qwen35OutputParserSalvage.reconstructToolCallBody(
            entry.kind, remainingBody);
        do {
            let toolCall = try self.parseToolCall(reconstructedBody);
            outputEvents.append(
                self.emitToolCallOrVisibleText(.toolCall(toolCall), reconstructedBody));
        } catch {
            // Closed envelopes fail open: the harness owns retries.
            if let failOpenEvent = self.failOpenClosedToolCall(reconstructedBody) {
                outputEvents.append(
                    self.emitToolCallOrVisibleText(failOpenEvent, reconstructedBody));
            }
        }
        self.state = .text;
        return true;
    }

    /// A think close inside a tool-call body proves the opener was quoted
    /// prose: a real tool-call body cannot contain a think close. Restore the
    /// opener and the buffered text to the origin channel, consume the marker,
    /// and resume channel scanning so a real tool call later in the generation
    /// still arrives.
    private func resyncSpuriousToolCall(
        _ entry: Qwen35ToolCallEntry, into outputEvents: inout Array<Qwen35OutputEvent>
    ) {
        let thinkEndBytes = Qwen35ByteText.bytes(Qwen35OutputParserMarkers.thinkEnd);
        let thinkCloseIndex = Qwen35ByteText.firstOffset(
            of: thinkEndBytes, in: self.pendingOutput)!;
        let bufferedBody = self.takePendingPrefixBytes(thinkCloseIndex);
        _ = self.takePendingPrefix(thinkEndBytes.count);
        var restoredProse = entry.opener;
        restoredProse += bufferedBody;
        if entry.openedInReasoning {
            outputEvents.append(.reasoningDelta(Qwen35ByteText.text(restoredProse[...])));
        } else {
            self.hasStreamedVisibleText = true;
            outputEvents.append(.textDelta(Qwen35ByteText.text(restoredProse[...])));
        }
        self.state = .text;
    }

    private func parseToolCall(_ toolCallBody: Array<UInt8>) throws -> Qwen35ToolCall {
        let normalizedBody = Qwen35ForeignToolSyntax.normalizeForeignToolCallSyntax(toolCallBody);
        let (functionName, parameterContent) = try splitQwenFunctionEnvelope(
            toolCallBody: Qwen35ByteText.trimmed(normalizedBody));
        // Unknown names and sloppy-but-closed XML are forwarded so the harness
        // can return "no such tool" and the model can retry.
        let functionNameKey = Qwen35ByteText.text(functionName[...]);
        let parsedArguments = Qwen35ToolSchemaParser.parseToolParameters(
            parameterContent: parameterContent, declaredTool: self.declaredTools[functionNameKey]);
        let argumentsObject = parsedArguments.map { (name, value) in (key: name, value: value) };
        let argumentsJson = Qwen35ParsedToolValue.object(Array(argumentsObject)).serializedJson();
        return Qwen35ToolCall(
            index: self.completedToolCallCount, functionName: functionNameKey,
            argumentsJson: argumentsJson);
    }

    private func failOpenClosedToolCall(
        _ toolCallBody: Array<UInt8>
    ) -> Qwen35OutputEvent? {
        let normalizedBody = Qwen35ForeignToolSyntax.normalizeForeignToolCallSyntax(toolCallBody);
        guard
            let (functionName, parameterContent) = try? splitQwenFunctionEnvelope(
                toolCallBody: Qwen35ByteText.trimmed(normalizedBody))
        else {
            if toolCallBody.isEmpty {
                return nil;
            }
            return .textDelta(Qwen35ByteText.text(toolCallBody[...]));
        }
        // Declared-schema rejection would still abort. Passthrough lets the
        // harness return invalid-argument or unknown-tool instead of killing
        // the stream.
        let parsedArguments = Qwen35ToolSchemaParser.parseToolParameters(
            parameterContent: parameterContent, declaredTool: nil);
        let argumentsObject = parsedArguments.map { (name, value) in (key: name, value: value) };
        let argumentsJson = Qwen35ParsedToolValue.object(Array(argumentsObject)).serializedJson();
        return .toolCall(Qwen35ToolCall(
            index: self.completedToolCallCount,
            functionName: Qwen35ByteText.text(functionName[...]),
            argumentsJson: argumentsJson));
    }

    private func enterStateAfterStartMarker(
        _ marker: Array<UInt8>, openedInReasoning: Bool
    ) {
        if marker == Qwen35ByteText.bytes(Qwen35OutputParserMarkers.thinkStart) {
            self.state = self.hasStreamedVisibleText
                ? .suppressedLateReasoning : .reasoning;
            return;
        }
        if marker == Qwen35ByteText.bytes(Qwen35OutputParserMarkers.toolCallStart) {
            self.state = .toolCall(Qwen35ToolCallEntry(
                kind: .envelope,
                opener: Qwen35ByteText.bytes(Qwen35OutputParserMarkers.toolCallStart),
                openedInReasoning: openedInReasoning));
            return;
        }
        if marker == Qwen35ByteText.bytes(Qwen35OutputParserMarkers.bareFunctionStart) {
            self.state = .toolCall(Qwen35ToolCallEntry(
                kind: .bareFunction,
                opener: Qwen35ByteText.bytes(Qwen35OutputParserMarkers.bareFunctionStart),
                openedInReasoning: openedInReasoning));
            return;
        }
        if marker == Qwen35ByteText.bytes(Qwen35OutputParserMarkers.invokeStart) {
            self.state = .toolCall(Qwen35ToolCallEntry(
                kind: .invokeTag,
                opener: Qwen35ByteText.bytes(Qwen35OutputParserMarkers.invokeStart),
                openedInReasoning: openedInReasoning));
            return;
        }
        self.state = .text;
    }

    private func takePendingPrefix(_ byteCount: Int) -> String {
        let prefixText = Qwen35ByteText.text(self.pendingOutput[0..<byteCount]);
        self.pendingOutput.removeFirst(byteCount);
        return prefixText;
    }

    /// Byte-preserving variant for paths that reassemble buffered text (tool
    /// call salvage, quoted-prose resync) instead of emitting it directly.
    private func takePendingPrefixBytes(_ byteCount: Int) -> Array<UInt8> {
        let prefixBytes = Array(self.pendingOutput[0..<byteCount]);
        self.pendingOutput.removeFirst(byteCount);
        return prefixBytes;
    }

    private func consumeTrailingEnvelopeCloseIfPresent() {
        let trimmedPending = Qwen35OutputParser.trimmedStart(self.pendingOutput);
        let toolCallEndBytes = Qwen35ByteText.bytes(Qwen35OutputParserMarkers.toolCallEnd);
        guard let afterMarker = Qwen35ByteText.stripPrefix(trimmedPending, toolCallEndBytes) else {
            return;
        }
        let consumedBytes = self.pendingOutput.count - afterMarker.count;
        _ = self.takePendingPrefix(consumedBytes);
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

    private func flushPendingAsVisibleDelta() -> Array<Qwen35OutputEvent> {
        if self.pendingOutput.isEmpty {
            return [];
        }
        let pendingBytes = self.pendingOutput;
        self.pendingOutput = [];
        if let pendingEvent = self.visibleDeltaForCurrentState(pendingBytes) {
            return [pendingEvent];
        }
        return [];
    }

    private func visibleDeltaForCurrentState(_ textBytes: Array<UInt8>) -> Qwen35OutputEvent? {
        switch self.state {
        case .text, .toolCall:
            self.hasStreamedVisibleText = true;
            return .textDelta(Qwen35ByteText.text(textBytes[...]));
        case .reasoning:
            return .reasoningDelta(Qwen35ByteText.text(textBytes[...]));
        case .suppressedLateReasoning:
            return nil;
        }
    }
}

/// Salvage of Qwen3.5 tool calls when generation ends or a memory bound is
/// crossed. Resource bounds stop unbounded buffering; aborting the stream as
/// malformed output would drop a usable function name the coding client could
/// reject or retry, so salvage forwards instead.
///
/// Mirrors crates/model-serving/src/qwen3_5/text/output_parser/salvage.rs.
extension Qwen35OutputParser {

    func salvageUnclosedToolCall() -> Array<Qwen35OutputEvent> {
        let entryKind: Qwen35ToolCallEntryKind;
        if case let .toolCall(entry) = self.state {
            entryKind = entry.kind;
        } else {
            entryKind = .envelope;
        }
        let remainingBody = self.pendingOutput;
        self.pendingOutput = [];
        self.state = .text;
        let toolCallBody = Qwen35OutputParserSalvage.reconstructToolCallBody(
            entryKind, remainingBody);
        guard let salvagedEvent = self.failOpenClosedToolCall(toolCallBody) else {
            return [];
        }
        return [self.emitToolCallOrVisibleText(salvagedEvent, toolCallBody)];
    }

    func emitToolCallOrVisibleText(
        _ salvagedEvent: Qwen35OutputEvent, _ toolCallBody: Array<UInt8>
    ) -> Qwen35OutputEvent {
        switch salvagedEvent {
        case .toolCall:
            if self.tryRecordCompletedToolCall() {
                return salvagedEvent;
            }
            return .textDelta(Qwen35ByteText.text(toolCallBody[...]));
        default:
            return salvagedEvent;
        }
    }

    private func tryRecordCompletedToolCall() -> Bool {
        // OpenAI tool_call.index is u16. Overflowing that width must not
        // abort generation.
        guard self.completedToolCallCount < UInt16.max else {
            return false;
        }
        self.completedToolCallCount += 1;
        return true;
    }
}

/// Pure salvage helpers shared by the closed-path and end-of-stream flows.
enum Qwen35OutputParserSalvage {

    static func reconstructToolCallBody(
        _ entryKind: Qwen35ToolCallEntryKind, _ remainingBody: Array<UInt8>
    ) -> Array<UInt8> {
        switch entryKind {
        case .envelope:
            return remainingBody;
        case .bareFunction:
            return Qwen35ByteText.bytes(Qwen35OutputParserMarkers.bareFunctionStart) + remainingBody;
        case .invokeTag:
            return Qwen35ByteText.bytes(Qwen35OutputParserMarkers.invokeStart) + remainingBody;
        }
    }

    static func toolCallEndMarkers(
        _ entryKind: Qwen35ToolCallEntryKind
    ) -> Array<Array<UInt8>> {
        switch entryKind {
        case .envelope:
            return [Qwen35ByteText.bytes(Qwen35OutputParserMarkers.toolCallEnd)];
        case .bareFunction, .invokeTag:
            return [
                Qwen35ByteText.bytes(Qwen35OutputParserMarkers.toolCallEnd),
                Qwen35ByteText.bytes(Qwen35OutputParserMarkers.functionEnd),
                Qwen35ByteText.bytes(Qwen35OutputParserMarkers.invokeEnd),
            ];
        }
    }
}
