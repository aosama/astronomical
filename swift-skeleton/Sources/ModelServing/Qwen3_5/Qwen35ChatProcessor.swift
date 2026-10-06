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
/// implementation on the adapter seam. The structured tool-call output
/// parser, the thinking channel seed, the thinking budget enforcement, and
/// token-masked structured generation land with their epic slices and fail
/// closed here with bounded rejection reasons.
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
            supportsToolCalls: false,
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
        if chatGenerationCommand.tools.isEmpty == false {
            // The tool-call output parser lands with the structured
            // generation slice; until then a tool-bearing conversation would
            // render without its marker parser, so it fails closed.
            throw ChatPreparationRejection(reason: .invalidRequest(
                reason: "tool-call parsing lands with the structured generation slice"));
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
            reasoningConfig: self.reasoningConfig,
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

/// Request-local translation over the upstream reasoning collector: decoded
/// deltas route to the reasoning and assistant-visible text channels, and
/// the collector's token retention keeps multibyte characters intact.
final class Qwen35ActiveChatGeneration: ActiveChatGeneration {

    /// The protocol's translation seam is non-mutating, so the value-type
    /// collector lives behind one box owned by this active generation.
    private let collectorBox: MutableBox<ReasoningTokenCollector>;
    let inferenceRequest: any PreparedInferenceRequest;
    let promptTokenCount: Int;
    private let endOfSequenceTokenIds: Set<UInt32>;

    init(
        tokenizer: any Tokenizer,
        reasoningConfig: ReasoningConfig,
        inferenceRequest: Qwen35PreparedInferenceRequest,
        endOfSequenceTokenIds: Set<UInt32>
    ) {
        self.collectorBox = MutableBox(ReasoningTokenCollector(
            config: reasoningConfig, primedInside: false, tokenizer: tokenizer));
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
        let segments: Array<ReasoningEventEmitter.Segment> = self.collectorBox.value
            .ingest(Int(generatedTokenId));
        return ModelGeneratedTokenTranslation(
            publicOutputs: Qwen35ActiveChatGeneration.outputs(from: segments));
    }

    func finishOutputs() throws -> Array<ChatGenerationOutput> {
        let segments: Array<ReasoningEventEmitter.Segment> = self.collectorBox.value.finalize();
        return Qwen35ActiveChatGeneration.outputs(from: segments);
    }

    private static func outputs(
        from segments: Array<ReasoningEventEmitter.Segment>
    ) -> Array<ChatGenerationOutput> {
        return segments.map { (segment: ReasoningEventEmitter.Segment) -> ChatGenerationOutput in
            switch segment {
            case let .reasoning(text): return .reasoning(text: text);
            case let .response(text): return .text(text: text);
            }
        };
    }
}
