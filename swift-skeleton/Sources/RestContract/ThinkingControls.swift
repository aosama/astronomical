// ThinkingControls.swift — RestContract
//
// Port of crates/rest-contract/src/thinking_controls.rs.
//
// Unified resolution of the client spellings that control the thinking budget.
// Clients trained on different gateways express the same control in several
// shapes: the OpenAI `reasoning_effort` level and numeric budget aliases, the
// OpenRouter-style `reasoning` object, a flat `enable_thinking` flag, and the
// vLLM-style `chat_template_kwargs` block. This unit is the single place that
// resolves every spelling into the worker's one hard thinking budget plus the
// stream exclusion preference, and fails loudly on any contradiction so a
// caller bug never silently changes how much a model thinks.

import Foundation;
import IpcProtocol;

/// OpenRouter-style `reasoning` object accepted on both generation endpoints.
///
/// Unknown fields are absorbed rather than rejected: this object is
/// provider-shaped and new spellings appear weekly (issue #777), so only the
/// contradictory *control values* below are caller errors.
public struct ReasoningRequestObject: Equatable {

    /// Effort level name; see `reasoning_effort_level` for the accepted set.
    public let effort: String?;

    /// Direct thinking-token limit (Anthropic-style), equivalent to the
    /// top-level numeric budget spellings.
    public let maxTokens: UInt32?;

    /// Explicit on/off switch for the thinking channel.
    public let enabled: Bool?;

    /// Keep the model thinking but withhold reasoning deltas from the stream.
    public let exclude: Bool?;

    /// OpenAI summary-inclusion preference (`auto`/`concise`/`detailed`).
    /// Absorbed without behavior change: Astronomical always streams
    /// reasoning deltas unless `exclude` withholds them.
    public let summary: String?;

    public init(
        effort: String? = nil,
        maxTokens: UInt32? = nil,
        enabled: Bool? = nil,
        exclude: Bool? = nil,
        summary: String? = nil
    ) {
        self.effort = effort;
        self.maxTokens = maxTokens;
        self.enabled = enabled;
        self.exclude = exclude;
        self.summary = summary;
    }

    /// Mirrors the serde derive: every field is `Option` with `default`, so a
    /// missing key or an explicit null decodes to `None` and unknown fields
    /// are absorbed silently.
    public static func decoded(wireValue: JsonWireValue) throws -> ReasoningRequestObject {
        let wireObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        var decodedEffort: String? = nil;
        var decodedMaxTokens: UInt32? = nil;
        var decodedEnabled: Bool? = nil;
        var decodedExclude: Bool? = nil;
        var decodedSummary: String? = nil;
        for entry in wireObject.entries {
            switch entry.key {
            case "effort":
                if decodedEffort != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "effort");
                }
                if entry.value.isNull == false {
                    decodedEffort = try JsonWireValue.extractString(entry.value);
                }
            case "max_tokens":
                if decodedMaxTokens != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "max_tokens");
                }
                if entry.value.isNull == false {
                    decodedMaxTokens = try JsonWireValue.clampToUInt32(try JsonWireValue.extractUInt64(entry.value));
                }
            case "enabled":
                if decodedEnabled != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "enabled");
                }
                if entry.value.isNull == false {
                    decodedEnabled = try JsonWireValue.extractBool(entry.value);
                }
            case "exclude":
                if decodedExclude != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "exclude");
                }
                if entry.value.isNull == false {
                    decodedExclude = try JsonWireValue.extractBool(entry.value);
                }
            case "summary":
                if decodedSummary != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "summary");
                }
                if entry.value.isNull == false {
                    decodedSummary = try JsonWireValue.extractString(entry.value);
                }
            default:
                continue;
            }
        }
        return ReasoningRequestObject(
            effort: decodedEffort,
            maxTokens: decodedMaxTokens,
            enabled: decodedEnabled,
            exclude: decodedExclude,
            summary: decodedSummary
        );
    }
}

/// vLLM-style template-kwarg block; only the thinking toggle is meaningful here.
/// Unknown template variables are absorbed: gateways add template-specific
/// kwargs that are not thinking controls.
public struct ChatTemplateKwargsRequestObject: Equatable {

    public let enableThinking: Bool?;

    public init(enableThinking: Bool? = nil) {
        self.enableThinking = enableThinking;
    }

    /// Mirrors the serde derive: a missing key or explicit null means `None`,
    /// and unknown template variables are absorbed.
    public static func decoded(wireValue: JsonWireValue) throws -> ChatTemplateKwargsRequestObject {
        let wireObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        var decodedEnableThinking: Bool? = nil;
        for entry in wireObject.entries {
            switch entry.key {
            case "enable_thinking":
                if decodedEnableThinking != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "enable_thinking");
                }
                if entry.value.isNull == false {
                    decodedEnableThinking = try JsonWireValue.extractBool(entry.value);
                }
            default:
                continue;
            }
        }
        return ChatTemplateKwargsRequestObject(enableThinking: decodedEnableThinking);
    }
}

/// The one resolved thinking control set consumed by translations and streams.
public struct ThinkingControls: Equatable {

    /// Worker hard budget. A zero value is an explicit disable that renders the
    /// thinking channel closed; `nil` leaves the model's default in force.
    public let budget: UInt32?;

    /// Whether reasoning deltas must be withheld from the response stream.
    public let reasoningExcluded: Bool;

    public init(budget: UInt32?, reasoningExcluded: Bool) {
        self.budget = budget;
        self.reasoningExcluded = reasoningExcluded;
    }
}

/// Every submitted thinking-control spelling, in endpoint-field order. The
/// Rust original holds borrowed references (`&'a`); Swift value copies are
/// equivalent here because the payload structs are tiny immutable carriers.
public struct ThinkingControlsInputs {

    public let thinkingBudget: UInt32?;
    public let thinkingTokenBudget: UInt32?;
    public let thinkingBudgetTokens: UInt32?;
    public let reasoning: ReasoningRequestObject?;
    public let reasoningEffort: String?;
    public let enableThinking: Bool?;
    public let chatTemplateKwargs: ChatTemplateKwargsRequestObject?;

    public init(
        thinkingBudget: UInt32? = nil,
        thinkingTokenBudget: UInt32? = nil,
        thinkingBudgetTokens: UInt32? = nil,
        reasoning: ReasoningRequestObject? = nil,
        reasoningEffort: String? = nil,
        enableThinking: Bool? = nil,
        chatTemplateKwargs: ChatTemplateKwargsRequestObject? = nil
    ) {
        self.thinkingBudget = thinkingBudget;
        self.thinkingTokenBudget = thinkingTokenBudget;
        self.thinkingBudgetTokens = thinkingBudgetTokens;
        self.reasoning = reasoning;
        self.reasoningEffort = reasoningEffort;
        self.enableThinking = enableThinking;
        self.chatTemplateKwargs = chatTemplateKwargs;
    }

    /// Resolves all spellings into one budget plus the exclusion preference.
    ///
    /// Precedence: explicit numbers beat level names, and an explicit disable
    /// (`off`/`none` level or any false flag) beats level names. Numbers
    /// disagreeing with each other, levels disagreeing with each other, flags
    /// disagreeing with each other, or a disable combined with a positive
    /// budget are caller bugs that fail loudly.
    public func resolve() throws -> ThinkingControls {
        let reasoning: ReasoningRequestObject = self.reasoning ?? ReasoningRequestObject();
        let resolvedNumericBudget: UInt32? = try ThinkingControlsResolution.resolveNumericGroup(
            [
                self.thinkingBudget,
                self.thinkingTokenBudget,
                self.thinkingBudgetTokens,
                reasoning.maxTokens,
            ],
            inputs: self
        );
        let levelBudget: ResolvedEffort? = try ThinkingControlsResolution.resolveLevelGroup(
            self.reasoningEffort,
            reasoning.effort
        );
        let thinkingEnabledByFlags: Bool = try ThinkingControlsResolution.resolveEnableFlagGroup(
            self,
            reasoning.enabled
        );

        var positiveNumericBudget: UInt32? = nil;
        if let numericBudget = resolvedNumericBudget {
            if numericBudget > 0 {
                positiveNumericBudget = numericBudget;
            }
        }
        var positiveLevelBudget: UInt32? = nil;
        if case let .budget(levelBudgetValue) = levelBudget {
            positiveLevelBudget = levelBudgetValue;
        }
        // Explicit numbers beat level names.
        let positiveBudget: UInt32? = positiveNumericBudget ?? positiveLevelBudget;
        // A zero budget, any false flag, or an off/none level disables thinking.
        var numericBudgetIsZero: Bool = false;
        if resolvedNumericBudget == 0 {
            numericBudgetIsZero = true;
        }
        var levelIsDisabled: Bool = false;
        if case .disabled = levelBudget {
            levelIsDisabled = true;
        }
        let disabled: Bool = numericBudgetIsZero || (thinkingEnabledByFlags == false) || levelIsDisabled;

        let budget: UInt32?;
        if disabled {
            if let requestedBudget = positiveBudget {
                throw ThinkingControlsError.thinkingDisabledWhileBudgetRequested(thinkingBudget: requestedBudget);
            }
            budget = 0;
        } else {
            budget = positiveBudget;
        }
        return ThinkingControls(budget: budget, reasoningExcluded: reasoning.exclude ?? false);
    }
}

/// Outcome of mapping one effort level name: a token budget or a disable.
fileprivate enum ResolvedEffort {
    case budget(UInt32);
    case disabled;
}

/// Module-private resolution helpers mirroring the Rust private functions.
fileprivate enum ThinkingControlsResolution {

    fileprivate static func resolveNumericGroup(
        _ numericSpellings: Array<UInt32?>,
        inputs: ThinkingControlsInputs
    ) throws -> UInt32? {
        var resolvedBudget: UInt32? = nil;
        for numericSpelling in numericSpellings {
            guard let candidateBudget = numericSpelling else {
                continue;
            }
            if let agreedBudget = resolvedBudget {
                if agreedBudget != candidateBudget {
                    throw ThinkingControlsError.conflictingNumericThinkingBudgets(
                        thinkingBudget: inputs.thinkingBudget,
                        thinkingTokenBudget: inputs.thinkingTokenBudget,
                        thinkingBudgetTokens: inputs.thinkingBudgetTokens,
                        reasoningMaxTokens: inputs.reasoning?.maxTokens
                    );
                }
            } else {
                resolvedBudget = candidateBudget;
            }
        }
        return resolvedBudget;
    }

    fileprivate static func resolveLevelGroup(
        _ reasoningEffort: String?,
        _ reasoningObjectEffort: String?
    ) throws -> ResolvedEffort? {
        var topLevelEffort: ResolvedEffort? = nil;
        if let effortText = reasoningEffort {
            topLevelEffort = try ThinkingControlsResolution.reasoningEffortLevel(effortText);
        }
        var nestedEffort: ResolvedEffort? = nil;
        if let nestedEffortText = reasoningObjectEffort {
            nestedEffort = try ThinkingControlsResolution.reasoningEffortLevel(nestedEffortText);
        }
        switch (topLevelEffort, nestedEffort) {
        case (.none, .none):
            return nil;
        case (.some(let resolvedTopLevelEffort), .some(let resolvedNestedEffort)):
            if case let .budget(actualBudget) = resolvedTopLevelEffort {
                if case let .budget(expectedBudget) = resolvedNestedEffort {
                    if actualBudget != expectedBudget {
                        throw ThinkingControlsError.conflictingReasoningEfforts(
                            reasoningEffort: reasoningEffort ?? "",
                            reasoningObjectEffort: reasoningObjectEffort ?? ""
                        );
                    }
                }
            }
            // A disable in either spelling outranks a level name in the other.
            if case .disabled = resolvedTopLevelEffort {
                return .disabled;
            }
            if case .disabled = resolvedNestedEffort {
                return .disabled;
            }
            return resolvedTopLevelEffort;
        case (.some(let resolvedTopLevelEffort), .none):
            return resolvedTopLevelEffort;
        case (.none, .some(let resolvedNestedEffort)):
            return resolvedNestedEffort;
        }
    }

    fileprivate static func resolveEnableFlagGroup(
        _ inputs: ThinkingControlsInputs,
        _ reasoningEnabled: Bool?
    ) throws -> Bool {
        var chatTemplateEnableThinking: Bool? = nil;
        if let chatTemplateKwargs = inputs.chatTemplateKwargs {
            chatTemplateEnableThinking = chatTemplateKwargs.enableThinking;
        }
        let flagValues: Array<Bool?> = [
            reasoningEnabled,
            inputs.enableThinking,
            chatTemplateEnableThinking,
        ];
        var thinkingEnabled: Bool = true;
        var flagsObserved: Int = 0;
        for flagValue in flagValues {
            guard let observedFlagValue = flagValue else {
                continue;
            }
            flagsObserved = flagsObserved + 1;
            if (flagsObserved > 1) && (observedFlagValue != thinkingEnabled) {
                throw ThinkingControlsError.conflictingThinkingEnableFlags(
                    reasoningEnabled: reasoningEnabled,
                    enableThinking: inputs.enableThinking,
                    chatTemplateEnableThinking: chatTemplateEnableThinking
                );
            }
            thinkingEnabled = observedFlagValue;
        }
        return thinkingEnabled;
    }

    /// Maps an effort level name to the thinking-token budget coding agents
    /// display for that level. The values mirror Pi's default thinking budgets so
    /// the number an agent shows its user is the number this server enforces.
    /// `xhigh` and `max` clamp to `high`, matching the agents' own clamping, and
    /// `off`/`none` disable the thinking channel entirely.
    fileprivate static func reasoningEffortLevel(_ reasoningEffort: String) throws -> ResolvedEffort {
        switch reasoningEffort {
        case "minimal": return .budget(1024);
        case "low": return .budget(2048);
        case "medium": return .budget(8192);
        case "high", "xhigh", "max": return .budget(16384);
        case "off", "none": return .disabled;
        default: throw ThinkingControlsError.unknownReasoningEffort(reasoningEffort: reasoningEffort);
        }
    }
}

/// Rust `{:?}` Debug formatting for `Option` payloads so error text stays
/// byte-identical to the thiserror output (`Some(1024)` / `None`).
fileprivate enum ThinkingControlsFormatting {

    fileprivate static func describedOptionalUInt32(_ optionalValue: UInt32?) -> String {
        guard let unwrappedValue = optionalValue else {
            return "None";
        }
        return "Some(\(unwrappedValue))";
    }

    fileprivate static func describedOptionalBool(_ optionalValue: Bool?) -> String {
        guard let unwrappedValue = optionalValue else {
            return "None";
        }
        return "Some(\(unwrappedValue))";
    }
}

/// A contradictory or unrecognized thinking-control spelling, rejected before
/// worker admission.
public enum ThinkingControlsError: Error, Equatable {

    /// Two or more numeric budget spellings were supplied and disagreed.
    case conflictingNumericThinkingBudgets(
        thinkingBudget: UInt32?,
        thinkingTokenBudget: UInt32?,
        thinkingBudgetTokens: UInt32?,
        reasoningMaxTokens: UInt32?
    );

    /// Two level names mapped to different thinking budgets.
    case conflictingReasoningEfforts(reasoningEffort: String, reasoningObjectEffort: String);

    /// An effort level name this server cannot enforce.
    case unknownReasoningEffort(reasoningEffort: String);

    /// Thinking enable flags were supplied and disagreed.
    case conflictingThinkingEnableFlags(
        reasoningEnabled: Bool?,
        enableThinking: Bool?,
        chatTemplateEnableThinking: Bool?
    );

    /// The client disabled thinking while also requesting a positive budget.
    case thinkingDisabledWhileBudgetRequested(thinkingBudget: UInt32);

    public var errorDescription: String? {
        switch self {
        case let .conflictingNumericThinkingBudgets(thinkingBudget, thinkingTokenBudget, thinkingBudgetTokens, reasoningMaxTokens):
            return "thinking_budget (\(ThinkingControlsFormatting.describedOptionalUInt32(thinkingBudget))) conflicts with thinking_token_budget (\(ThinkingControlsFormatting.describedOptionalUInt32(thinkingTokenBudget))), thinking_budget_tokens (\(ThinkingControlsFormatting.describedOptionalUInt32(thinkingBudgetTokens))), and reasoning.max_tokens (\(ThinkingControlsFormatting.describedOptionalUInt32(reasoningMaxTokens)))";
        case let .conflictingReasoningEfforts(reasoningEffort, reasoningObjectEffort):
            return "reasoning_effort '\(reasoningEffort)' conflicts with reasoning.effort '\(reasoningObjectEffort)'";
        case let .unknownReasoningEffort(reasoningEffort):
            return "reasoning_effort '\(reasoningEffort)' is not a recognized thinking level; expected minimal, low, medium, high, xhigh, max, off, or none";
        case let .conflictingThinkingEnableFlags(reasoningEnabled, enableThinking, chatTemplateEnableThinking):
            return "thinking enable flags disagree: reasoning.enabled (\(ThinkingControlsFormatting.describedOptionalBool(reasoningEnabled))), enable_thinking (\(ThinkingControlsFormatting.describedOptionalBool(enableThinking))), chat_template_kwargs.enable_thinking (\(ThinkingControlsFormatting.describedOptionalBool(chatTemplateEnableThinking)))";
        case let .thinkingDisabledWhileBudgetRequested(thinkingBudget):
            return "thinking is disabled while a positive thinking budget of \(thinkingBudget) tokens is requested";
        }
    }
}
