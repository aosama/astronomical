import Foundation;

/**
 * Shared Laguna root-template selection keeps public discovery and worker
 * startup aligned, porting crates/config/src/laguna_template_source.rs. The
 * caseless enum is the Swift equivalent of the Rust module of free functions
 * and constants.
 */
internal enum LagunaTemplateSource {

    private static let CHAT_TEMPLATE_FIELD_NAME: String = "chat_template";

    /**
     * Bounded filesystem evidence supplied by discovery or authoritative
     * validation.
     */
    internal enum StandaloneChatTemplateState: Equatable, Sendable {
        case missing;
        case empty;
        case nonEmpty;
    }

    /**
     * The one root source selected before include resolution or template
     * compilation.
     */
    internal enum RootChatTemplateSource: Equatable, Sendable {
        case embedded(templateSource: String, standaloneTemplateRequiresInclude: Bool);
        case standalone;
    }

    /**
     * A deterministic root-source selection failure shared by discovery and
     * model startup.
     */
    internal enum RootChatTemplateSelectionError: Error, CustomStringConvertible {
        case malformedTokenizerConfig(underlyingDescription: String);
        case tokenizerConfigMustBeObject;
        case duplicateChatTemplateField;
        case emptyEmbeddedChatTemplate;
        case unsupportedEmbeddedChatTemplateType;
        case missingRootChatTemplate;
        case emptyStandaloneChatTemplate;
        case conflictingRootChatTemplates;

        internal var description: String {
            switch (self) {
            case .malformedTokenizerConfig(let underlyingDescription):
                return "Laguna tokenizer configuration is not valid JSON: \(underlyingDescription)";
            case .tokenizerConfigMustBeObject:
                return "Laguna tokenizer configuration root must be a JSON object";
            case .duplicateChatTemplateField:
                return "Laguna tokenizer configuration contains duplicate chat_template fields";
            case .emptyEmbeddedChatTemplate:
                return "Laguna embedded chat_template must not be empty";
            case .unsupportedEmbeddedChatTemplateType:
                return "Laguna embedded chat_template must be a string or null";
            case .missingRootChatTemplate:
                return "Laguna artifact does not provide a root chat template";
            case .emptyStandaloneChatTemplate:
                return "Laguna standalone chat_template.jinja must not be empty";
            case .conflictingRootChatTemplates:
                return "Laguna artifact provides conflicting embedded and standalone chat templates";
            }
        }
    }

    /**
     * Selects one authority without opening files so every caller can
     * preserve its own I/O policy.
     */
    internal static func selectRootChatTemplate(
        tokenizerConfigBytes: Data,
        standaloneTemplateState: LagunaTemplateSource.StandaloneChatTemplateState
    ) throws -> LagunaTemplateSource.RootChatTemplateSource {
        let tokenizerConfigValue: Any;
        do {
            tokenizerConfigValue = try DuplicateKeyJson.parseJsonAllowingDuplicateKeys(configBytes: tokenizerConfigBytes);
        } catch let parseError {
            throw LagunaTemplateSource.RootChatTemplateSelectionError.malformedTokenizerConfig(
                underlyingDescription: String(describing: parseError)
            );
        }
        if (tokenizerConfigValue is Dictionary<String, Any>) == false {
            throw LagunaTemplateSource.RootChatTemplateSelectionError.tokenizerConfigMustBeObject;
        }

        let templateProjection: TokenizerTemplateProjection = try LagunaTemplateSource.templateProjection(
            tokenizerConfigBytes: tokenizerConfigBytes
        );
        if (templateProjection.chatTemplateOccurrenceCount > 1) {
            throw LagunaTemplateSource.RootChatTemplateSelectionError.duplicateChatTemplateField;
        }

        if let templateSource: String = templateProjection.chatTemplateValue as? String {
            if (templateSource.isEmpty) {
                throw LagunaTemplateSource.RootChatTemplateSelectionError.emptyEmbeddedChatTemplate;
            }
            var standaloneTemplateRequiresInclude: Bool = false;
            switch (standaloneTemplateState) {
            case .nonEmpty: standaloneTemplateRequiresInclude = true;
            case .empty: standaloneTemplateRequiresInclude = false;
            case .missing: standaloneTemplateRequiresInclude = false;
            }
            // A physical chat_template.jinja may be an explicitly selected
            // include. Callers resolve the graph before deciding that it is a
            // second root authority.
            return LagunaTemplateSource.RootChatTemplateSource.embedded(
                templateSource: templateSource,
                standaloneTemplateRequiresInclude: standaloneTemplateRequiresInclude
            );
        }
        if (templateProjection.chatTemplateValue is NSNull) || (templateProjection.chatTemplateValue == nil) {
            switch (standaloneTemplateState) {
            case .nonEmpty:
                return LagunaTemplateSource.RootChatTemplateSource.standalone;
            case .empty:
                throw LagunaTemplateSource.RootChatTemplateSelectionError.emptyStandaloneChatTemplate;
            case .missing:
                throw LagunaTemplateSource.RootChatTemplateSelectionError.missingRootChatTemplate;
            }
        }
        throw LagunaTemplateSource.RootChatTemplateSelectionError.unsupportedEmbeddedChatTemplateType;
    }

    /**
     * Rejects a nonempty standalone file unless the embedded root selected it
     * as an include.
     */
    internal static func validateStandaloneChatTemplateRole(
        rootTemplateSource: LagunaTemplateSource.RootChatTemplateSource,
        standaloneTemplateIsSelectedInclude: Bool
    ) throws -> Void {
        var embeddedTemplateRequiresInclude: Bool = false;
        switch (rootTemplateSource) {
        case .embedded(_, let standaloneTemplateRequiresInclude):
            embeddedTemplateRequiresInclude = standaloneTemplateRequiresInclude;
        case .standalone:
            embeddedTemplateRequiresInclude = false;
        }
        if (embeddedTemplateRequiresInclude && (standaloneTemplateIsSelectedInclude == false)) {
            throw LagunaTemplateSource.RootChatTemplateSelectionError.conflictingRootChatTemplates;
        }
    }

    /**
     * Projects only the chat_template evidence out of the tokenizer
     * configuration, counting occurrences so a duplicated field fails
     * deterministically instead of silently keeping the last value.
     */
    private static func templateProjection(
        tokenizerConfigBytes: Data
    ) throws -> TokenizerTemplateProjection {
        let topLevelEntries: Array<(key: String, jsonValue: Any)>;
        do {
            topLevelEntries = try DuplicateKeyJson.parseJsonTopLevelObjectEntriesAllowingDuplicateKeys(
                configBytes: tokenizerConfigBytes
            );
        } catch let parseError {
            throw LagunaTemplateSource.RootChatTemplateSelectionError.malformedTokenizerConfig(
                underlyingDescription: String(describing: parseError)
            );
        }
        var chatTemplateValue: Any?;
        var chatTemplateOccurrenceCount: Int = 0;
        for (key: fieldName, jsonValue: fieldValue) in topLevelEntries {
            if (fieldName == LagunaTemplateSource.CHAT_TEMPLATE_FIELD_NAME) {
                chatTemplateOccurrenceCount += 1;
                if (chatTemplateValue == nil) {
                    chatTemplateValue = fieldValue;
                }
            }
        }
        return TokenizerTemplateProjection(
            chatTemplateValue: chatTemplateValue,
            chatTemplateOccurrenceCount: chatTemplateOccurrenceCount
        );
    }

    private struct TokenizerTemplateProjection {
        let chatTemplateValue: Any?;
        let chatTemplateOccurrenceCount: Int;
    }
}
