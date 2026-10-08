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
        // A zero budget disables the thinking channel entirely: the template's
        // opened block closes immediately and generation starts in the visible
        // answer channel, mirroring the Rust enable-thinking rule.
        let enableThinking = chatGenerationCommand.settings.thinkingBudget != 0;
        let forcedTransitionTokenIds = self.forcedThinkingTransitionTokenIds();
        let effectiveThinkingAllowance = Qwen35ThinkingAllowance
            .resolveEffectiveThinkingAllowance(
                requestedThinkingBudget: chatGenerationCommand.settings.thinkingBudget,
                enableThinking: enableThinking,
                maximumOutputTokens: chatGenerationCommand.settings.maxOutputTokens,
                transitionTokenCount: forcedTransitionTokenIds.count);
        if let effectiveThinkingAllowance,
            let requestedBudget = chatGenerationCommand.settings.thinkingBudget,
            enableThinking && requestedBudget > 0
        {
            let minimumBoundedOutputTokens = Qwen35ThinkingAllowance
                .minimumBoundedOutputTokenCount(
                    thinkingBudget: effectiveThinkingAllowance,
                    forcedTransitionTokenCount: forcedTransitionTokenIds.count) ?? Int.max;
            if Int(chatGenerationCommand.settings.maxOutputTokens) < minimumBoundedOutputTokens {
                throw ChatPreparationRejection(reason: .invalidRequest(
                    reason:
                        "the thinking allowance \(effectiveThinkingAllowance) cannot reserve its forced reasoning transition (\(forcedTransitionTokenIds.count) tokens) and one visible answer token within the \(chatGenerationCommand.settings.maxOutputTokens) output-token limit"));
            }
        }
        let thinkingBudgetState: Qwen35ThinkingBudgetState;
        do {
            thinkingBudgetState = try Qwen35ThinkingBudgetState(
                startsInsideThinking: enableThinking,
                thinkingBudget: effectiveThinkingAllowance,
                forcedTransitionTokenIds: forcedTransitionTokenIds,
                naturalReasoningEndTokenIds: Array(
                    self.naturalReasoningEndTokenIdsIncludingToolCalls()));
        } catch {
            throw ChatPreparationRejection(reason: .invalidRequest(
                reason: "invalid Qwen3.5 thinking-budget configuration: \(error)"));
        }
        // The parser validates declared tool schemas up front so a broken
        // declaration surfaces as a typed malformed-output rejection naming
        // the offending tool, mirroring the Rust request-output parser path.
        let outputParser: Qwen35OutputParser;
        do {
            outputParser = try Qwen35OutputParser(
                declaredTools: chatGenerationCommand.tools,
                startsInsideThinking: enableThinking);
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
        var promptTokenIds: Array<Int>;
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
        // The stock template opened the thinking channel; a disabled channel
        // closes it immediately. Thinking windows open and close under budget
        // control only — callers cannot seed content into them.
        if enableThinking == false {
            promptTokenIds += self.tokenizer.encode(
                text: self.reasoningConfig.endDelimiter + "\n\n", addSpecialTokens: false);
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
                startsInsideThinking: enableThinking,
                naturalReasoningEndTokenIds: self.naturalReasoningEndTokenIdsIncludingToolCalls(),
                thinkingBudgetState: thinkingBudgetState),
            endOfSequenceTokenIds: self.endOfSequenceTokenIds);
    }

    /// Resolves the forced reasoning-transition ids: the template-authored
    /// pivot text tokenized without special tokens, closed by the single
    /// think-end token id so the transition ends at a recognized reasoning
    /// boundary. A tokenizer without a single-id think close cannot serve a
    /// positive budget — the state constructor rejects the empty transition.
    private func forcedThinkingTransitionTokenIds() -> Array<UInt32> {
        var transitionTokenIds = self.tokenizer.encode(
            text: Qwen35ThinkingBudgetState.thinkingBudgetTransitionText,
            addSpecialTokens: false).map { (transitionTokenId: Int) -> UInt32 in
            return UInt32(clamping: transitionTokenId);
        };
        if let thinkEndTokenId = singleAddedTokenId(forText: self.reasoningConfig.endDelimiter) {
            transitionTokenIds.append(thinkEndTokenId);
        }
        return transitionTokenIds;
    }

    /// The reasoning-end ids plus the `<tool_call>` start: mlx-lm transitions
    /// from reasoning to tool on a tool-call start, so an emitted tool call
    /// also closes the thinking channel naturally. Only single-id special
    /// tokens count as boundaries — a tokenizer that spells a marker as
    /// character pieces has no such boundary, and inventing one from a
    /// character or unknown-fallback id would corrupt the budget validation.
    private func naturalReasoningEndTokenIdsIncludingToolCalls() -> Set<UInt32> {
        var boundaryTokenIds: Set<UInt32> = [];
        if let thinkEndTokenId = singleAddedTokenId(forText: self.reasoningConfig.endDelimiter) {
            boundaryTokenIds.insert(thinkEndTokenId);
        }
        if let toolCallStartTokenId = singleAddedTokenId(forText: "<tool_call>") {
            boundaryTokenIds.insert(toolCallStartTokenId);
        }
        return boundaryTokenIds;
    }

    /// The token id for `text` only when the tokenizer encodes it as exactly
    /// one added token.
    private func singleAddedTokenId(forText text: String) -> UInt32? {
        let encodedTokenIds = self.tokenizer.encode(text: text, addSpecialTokens: false);
        guard encodedTokenIds.count == 1, let singleTokenId = encodedTokenIds.first else {
            return nil;
        }
        return UInt32(clamping: singleTokenId);
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
        let publicOutputs = Qwen35ActiveChatGeneration.outputs(from: outputEvents);
        return ModelGeneratedTokenTranslation(publicOutputs: publicOutputs);
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
