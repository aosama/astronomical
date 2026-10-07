import Foundation;

import IpcProtocol;
import MLXLMCommon;
import MLXHuggingFace;

/// The real Qwen3.5 dense chat processor over the upstream bridged
/// tokenizer: chat-template preparation, seeded sampling settings, the
/// tagged `<think>` reasoning channel routed through the upstream reasoning
/// collector, and end-of-sequence ids from the artifact config.
///
/// Mirrors the serving behavior of the Rust `ModelGenerationProcessor`
/// implementation on the adapter seam. The thinking channel seed and the
/// thinking budget enforcement land with their epic slices and fail closed
/// here with bounded rejection reasons; tool-call output parses fail-open
/// through `Qwen35OutputParser`.
public final class Qwen35ChatProcessor: ModelGenerationProcessor {

    private let tokenizer: any Tokenizer;
    private let reasoningConfig: ReasoningConfig;
    private let modelId: String;
    private let endOfSequenceTokenIds: Set<UInt32>;
    private let capabilities: ChatModelCapabilities;

    public init(
        tokenizer: any Tokenizer,
        modelId: String,
        endOfSequenceTokenIds: Set<UInt32>,
        capabilities: ChatModelCapabilities,
        reasoningConfig: ReasoningConfig = QwenReasoningProtocol.tagged
    ) {
        self.tokenizer = tokenizer;
        self.reasoningConfig = reasoningConfig;
        self.modelId = modelId;
        self.endOfSequenceTokenIds = endOfSequenceTokenIds;
        self.capabilities = capabilities;
    }

    /// Derives the advertised chat capabilities from the artifact config and
    /// the selected autoregressive policy, never from hardcoded model names.
    public static func capabilities(
        autoregressiveConfiguration: WorkerAutoregressiveModelConfiguration,
        maximumPositionCount: UInt32
    ) -> ChatModelCapabilities {
        let maximumOutputTokens: UInt32 = min(
            autoregressiveConfiguration.maximumOutputTokens, maximumPositionCount);
        return ChatModelCapabilities(
            supportsReasoning: true,
            supportsToolCalls: true,
            hasVision: false,
            maxInputTokens: maximumPositionCount > maximumOutputTokens
                ? maximumPositionCount - maximumOutputTokens : 0,
            maxOutputTokens: maximumOutputTokens,
            contextWindow: maximumPositionCount);
    }

    public func readyEvent() -> WorkerEvent {
        return .ready(
            modelId: self.modelId,
            capabilities: WorkerModelCapabilities(
                chat: self.capabilities, imageGeneration: nil, embeddings: nil));
    }

    public func prepareChatGeneration(
        _ chatGenerationCommand: ChatGenerationCommand
    ) throws -> any ActiveChatGeneration {
        if let thinkingChannelSeed = chatGenerationCommand.qwenThinkingChannelSeed,
            thinkingChannelSeed.isEmpty == false {
            throw ChatPreparationRejection(reason: .invalidRequest(
                reason: "the thinking channel seed lands with the thinking-budget slice"));
        }
        if chatGenerationCommand.settings.thinkingBudget != nil {
            throw ChatPreparationRejection(reason: .invalidRequest(
                reason: "the thinking budget lands with the thinking-budget slice"));
        }
        // The parser validates declared tool schemas up front so a broken
        // declaration surfaces as a typed malformed-output rejection naming
        // the offending tool, mirroring the Rust request-output parser path.
        let outputParser: Qwen35OutputParser;
        do {
            outputParser = try Qwen35OutputParser(
                declaredTools: chatGenerationCommand.tools, startsInsideThinking: true);
        } catch {
            throw ChatPreparationRejection(reason: .malformedModelOutput);
        }
        var guidedConstraint: Qwen35GuidedConstraint? = nil;
        if let enforcedConstraint = chatGenerationCommand.structuredGeneration {
            do {
                guidedConstraint = try Qwen35GuidedConstraint.compile(
                    constraint: enforcedConstraint,
                    tokenizer: self.tokenizer,
                    endOfSequenceTokenId: Int32(bitPattern: UInt32(
                        self.endOfSequenceTokenIds.min() ?? 0)));
            } catch let guidedRejection as Qwen35GuidedConstraintError {
                throw ChatPreparationRejection(reason: .invalidRequest(
                    reason: guidedRejection.description));
            } catch {
                throw ChatPreparationRejection(reason: .invalidRequest(
                    reason: "the structured-generation constraint failed to compile"));
            }
        }
        let promptTokenIds: Array<Int>;
        do {
            promptTokenIds = try self.tokenizer.applyChatTemplate(
                messages: Qwen35ChatProcessor.templateMessages(
                    from: chatGenerationCommand.messages),
                tools: Qwen35ChatProcessor.templateTools(
                    from: chatGenerationCommand.tools),
                additionalContext: nil);
        } catch {
            throw ChatPreparationRejection(reason: .invalidRequest(
                reason: "the conversation could not be rendered into the model chat template"));
        }
        return Qwen35ActiveChatGeneration(
            tokenizer: self.tokenizer,
            outputParser: outputParser,
            inferenceRequest: Qwen35PreparedInferenceRequest(
                promptTokenIds: promptTokenIds.map { (promptTokenId: Int) -> UInt32 in
                    return UInt32(clamping: promptTokenId);
                },
                samplingSettings: Qwen35SamplingSettings(
                    chatGenerationSettings: chatGenerationCommand.settings),
                guidedConstraint: guidedConstraint,
                startsInsideThinking: true,
                naturalReasoningEndTokenIds: self.naturalReasoningEndTokenIds()),
            endOfSequenceTokenIds: self.endOfSequenceTokenIds);
    }

    /// Resolves the `</think>` boundary token ids from the tokenizer so the
    /// engine can flip the constraint from dormant (inside thinking) to
    /// masking (visible answer), mirroring the Rust generation-start rule
    /// where the template's opened thinking channel gates the mask.
    private func naturalReasoningEndTokenIds() -> Set<UInt32> {
        let encodedBoundaryIds: Array<Int> = self.tokenizer.encode(
            text: self.reasoningConfig.endDelimiter, addSpecialTokens: false);
        return Set<UInt32>(encodedBoundaryIds.map { (boundaryTokenId: Int) -> UInt32 in
            return UInt32(clamping: boundaryTokenId);
        });
    }

    /// Maps the wire conversation onto the chat-template message dictionaries.
    static func templateMessages(
        from chatMessages: Array<ChatMessage>
    ) -> Array<Dictionary<String, any Sendable>> {
        return chatMessages.map { (chatMessage: ChatMessage) -> Dictionary<String, any Sendable> in
            switch chatMessage {
            case let .system(content):
                return ["role": "system", "content": content];
            case let .user(content, _):
                // Image inputs belong to the vision slice; the dense text
                // template consumes the textual content.
                return ["role": "user", "content": content];
            case let .assistant(content, _, _):
                return ["role": "assistant", "content": content ?? ""];
            case let .tool(toolCallId, content):
                return ["role": "tool", "content": content, "tool_call_id": toolCallId];
            }
        };
    }

    /// Maps tool definitions onto the chat-template tool schema dictionaries;
    /// `nil` keeps the template tool-free.
    static func templateTools(
        from toolDefinitions: Array<ChatToolDefinition>
    ) -> Array<Dictionary<String, any Sendable>>? {
        if toolDefinitions.isEmpty {
            return nil;
        }
        return toolDefinitions.map { (toolDefinition: ChatToolDefinition) -> Dictionary<String, any Sendable> in
            var functionSchema: Dictionary<String, any Sendable> = [
                "name": toolDefinition.name,
            ];
            if let description = toolDefinition.description {
                functionSchema["description"] = description;
            }
            functionSchema["parameters"] = toolDefinition.parametersJson;
            return [
                "type": "function",
                "function": functionSchema,
            ];
        };
    }
}

/// Request-local translation over the Swift Qwen3.5 output parser: decoded
/// deltas route through the parser's reasoning, text, and tool-call channels,
/// mirroring the Rust request-output decode loop. The streaming detokenizer
/// keeps multibyte characters intact across token boundaries.
final class Qwen35ActiveChatGeneration: ActiveChatGeneration {

    private let outputParser: Qwen35OutputParser;
    private var detokenizer: NaiveStreamingDetokenizer;
    let inferenceRequest: any PreparedInferenceRequest;
    let promptTokenCount: Int;
    private let endOfSequenceTokenIds: Set<UInt32>;

    init(
        tokenizer: any Tokenizer,
        outputParser: Qwen35OutputParser,
        inferenceRequest: Qwen35PreparedInferenceRequest,
        endOfSequenceTokenIds: Set<UInt32>
    ) {
        self.outputParser = outputParser;
        self.detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer);
        self.inferenceRequest = inferenceRequest;
        self.promptTokenCount = inferenceRequest.promptTokenCount;
        self.endOfSequenceTokenIds = endOfSequenceTokenIds;
    }

    func isEndOfSequenceToken(_ generatedTokenId: UInt32) -> Bool {
        return self.endOfSequenceTokenIds.contains(generatedTokenId);
    }

    func translateGeneratedToken(
        _ generatedTokenId: UInt32
    ) throws -> ModelGeneratedTokenTranslation {
        self.detokenizer.append(token: Int(generatedTokenId));
        guard let decodedFragment = self.detokenizer.next() else {
            return ModelGeneratedTokenTranslation(publicOutputs: []);
        }
        let outputEvents = try self.outputParser.pushFragment(decodedFragment);
        return ModelGeneratedTokenTranslation(
            publicOutputs: Qwen35ActiveChatGeneration.outputs(from: outputEvents));
    }

    func finishOutputs() throws -> Array<ChatGenerationOutput> {
        return Qwen35ActiveChatGeneration.outputs(from: self.outputParser.finish());
    }

    private static func outputs(
        from outputEvents: Array<Qwen35OutputEvent>
    ) -> Array<ChatGenerationOutput> {
        return outputEvents.map { (outputEvent: Qwen35OutputEvent) -> ChatGenerationOutput in
            switch outputEvent {
            case let .reasoningDelta(text): return .reasoning(text: text);
            case let .textDelta(text): return .text(text: text);
            case let .toolCall(toolCall): return .toolCall(
                toolCallIndex: toolCall.index, functionName: toolCall.functionName,
                argumentsJson: toolCall.argumentsJson);
            }
        };
    }
}
