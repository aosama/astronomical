// StructuredOutputs.swift — RestContract
//
// Port of crates/rest-contract/src/openai_structured_outputs.rs.
//
// Extra-body structured generation that must be token-enforced or rejected.
// OpenAI `response_format` may degrade to a prompt plus Warning. These fields
// must not: if the worker cannot mask illegal tokens, the request fails.

import Foundation;
import IpcProtocol;

/// vLLM-style extra body for constrained decoding.
public struct OpenAiStructuredOutputs: Equatable {

    /// Bounds one regex pattern so DFA compilation stays bounded and predictable.
    public static let MAXIMUM_STRUCTURED_REGEX_PATTERN_BYTES: Int = 2_048;

    /// Wire key `json`, with `json_schema` accepted as a serde-style alias.
    public let jsonSchema: JsonWireValue?;
    public let regex: String?;
    public let choice: Array<String>?;
    public let grammar: String?;

    public init(
        jsonSchema: JsonWireValue? = nil,
        regex: String? = nil,
        choice: Array<String>? = nil,
        grammar: String? = nil
    ) {
        self.jsonSchema = jsonSchema;
        self.regex = regex;
        self.choice = choice;
        self.grammar = grammar;
    }

    /// Mirrors the serde derive for
    /// `#[serde(rename = "json", alias = "json_schema", default)]`: both wire
    /// keys feed one field, setting it twice is a duplicate-field error, a
    /// missing key or explicit null means `None`, and unknown fields are
    /// absorbed because the Rust struct carries no deny_unknown_fields.
    public static func decoded(wireValue: JsonWireValue) throws -> OpenAiStructuredOutputs {
        let wireObject: JsonWireObject = try JsonWireValue.extractObject(wireValue);
        var decodedJsonSchema: JsonWireValue? = nil;
        var decodedRegex: String? = nil;
        var decodedChoice: Array<String>? = nil;
        var decodedGrammar: String? = nil;
        for entry in wireObject.entries {
            switch entry.key {
            case "json", "json_schema":
                if decodedJsonSchema != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "json_schema");
                }
                if entry.value.isNull == false {
                    decodedJsonSchema = entry.value;
                }
            case "regex":
                if decodedRegex != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "regex");
                }
                if entry.value.isNull == false {
                    decodedRegex = try JsonWireValue.extractString(entry.value);
                }
            case "choice":
                if decodedChoice != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "choice");
                }
                if entry.value.isNull == false {
                    decodedChoice = try JsonWireValue.extractArray(
                        entry.value,
                        mappedElement: { (elementWireValue: JsonWireValue) throws -> String in
                            return try JsonWireValue.extractString(elementWireValue);
                        });
                }
            case "grammar":
                if decodedGrammar != nil {
                    throw JsonWireProblem.duplicateField(fieldName: "grammar");
                }
                if entry.value.isNull == false {
                    decodedGrammar = try JsonWireValue.extractString(entry.value);
                }
            default:
                continue;
            }
        }
        return OpenAiStructuredOutputs(
            jsonSchema: decodedJsonSchema,
            regex: decodedRegex,
            choice: decodedChoice,
            grammar: decodedGrammar
        );
    }

    /// Compiles extra-body fields into a worker-enforced constraint or fails closed.
    public func intoEnforcedGeneration() throws -> EnforcedStructuredGeneration {
        var setFieldCount: Int = 0;
        if self.jsonSchema != nil {
            setFieldCount = setFieldCount + 1;
        }
        if self.regex != nil {
            setFieldCount = setFieldCount + 1;
        }
        if self.choice != nil {
            setFieldCount = setFieldCount + 1;
        }
        if self.grammar != nil {
            setFieldCount = setFieldCount + 1;
        }
        if setFieldCount != 1 {
            throw OpenAiStructuredOutputsValidationError.multipleOrEmptyFields;
        }
        if let regexPattern = self.regex {
            if regexPattern.isEmpty {
                throw OpenAiStructuredOutputsValidationError.regexNotCompilable(reason: "pattern is empty");
            }
            // Rust measures `str::len` in UTF-8 bytes.
            if regexPattern.utf8.count > OpenAiStructuredOutputs.MAXIMUM_STRUCTURED_REGEX_PATTERN_BYTES {
                throw OpenAiStructuredOutputsValidationError.regexPatternTooLarge(
                    maximumPatternBytes: OpenAiStructuredOutputs.MAXIMUM_STRUCTURED_REGEX_PATTERN_BYTES);
            }
            // The Rust side compiles a regex-automata dense DFA; ICU via
            // NSRegularExpression is the equivalent bounded-compile check here.
            // Error wording differs between the engines, and ICU accepts a few
            // constructs the DFA engine rejects (lookaround, backreferences);
            // the worker-side compile remains the final gate.
            do {
                _ = try NSRegularExpression(pattern: regexPattern);
            } catch {
                throw OpenAiStructuredOutputsValidationError.regexNotCompilable(reason: error.localizedDescription);
            }
            // Failing the automaton build here means the request cannot be
            // enforced, so it must fail closed at the public boundary instead
            // of at the worker.
            return .regex(pattern: regexPattern);
        }
        if self.grammar != nil {
            throw OpenAiStructuredOutputsValidationError.grammarNotEnforced;
        }
        if let requestedChoices = self.choice {
            var trimmedChoices: Array<String> = Array<String>();
            for requestedChoice in requestedChoices {
                let trimmedChoice: String = requestedChoice.trimmingCharacters(in: .whitespacesAndNewlines);
                if trimmedChoice.isEmpty == false {
                    trimmedChoices.append(trimmedChoice);
                }
            }
            if trimmedChoices.isEmpty {
                throw OpenAiStructuredOutputsValidationError.emptyChoice;
            }
            return .choice(choices: trimmedChoices);
        }
        guard let schemaWireValue = self.jsonSchema else {
            // Unreachable: exactly one field was counted as set and every
            // other branch already returned; the Rust side uses expect().
            throw OpenAiStructuredOutputsValidationError.multipleOrEmptyFields;
        }
        return try EnforcedStructuredGeneration.enforced_generation_from_json_schema(schemaWireValue);
    }
}

/// Validated extra-body constraint that the worker can compile into a token mask.
public enum EnforcedStructuredGeneration: Equatable {

    /// One JSON object.
    case jsonObject;
    /// JSON matching a bounded object schema.
    case jsonSchema(schema: JsonWireValue);
    /// Exact one of these UTF-8 strings.
    case choice(choices: Array<String>);
    /// The complete visible answer must match this regular expression.
    case regex(pattern: String);

    /// Compiles one parsed JSON-schema value into the enforced generation. This
    /// is the single home of the object-schema rule (an empty object means any
    /// JSON object; a non-object schema is not enforceable) so the REST surface
    /// and any other schema-carrying surface cannot drift apart.
    public static func enforced_generation_from_json_schema(_ schema: JsonWireValue) throws -> EnforcedStructuredGeneration {
        if case let .object(schemaObject) = schema {
            if schemaObject.entries.isEmpty {
                return .jsonObject;
            }
            return .jsonSchema(schema: schema);
        }
        throw OpenAiStructuredOutputsValidationError.jsonSchemaMustBeObject;
    }

    /// Compiles extra-body `structured_outputs` or `guided_grammar`, never both.
    public static func enforced_generation_from_extra_body(
        structuredOutputs: OpenAiStructuredOutputs?,
        guidedGrammar: String?
    ) throws -> EnforcedStructuredGeneration? {
        if (structuredOutputs != nil) && (guidedGrammar != nil) {
            throw OpenAiStructuredOutputsValidationError.conflictingExtraBodyFields;
        }
        if let parsedStructuredOutputs = structuredOutputs {
            return try parsedStructuredOutputs.intoEnforcedGeneration();
        }
        if let guidedGrammarText = guidedGrammar {
            return try EnforcedStructuredGeneration.guided_grammar_to_enforced_generation(guidedGrammarText);
        }
        return nil;
    }

    /// Compiles a guided EBNF string. Current workers cannot enforce EBNF.
    public static func guided_grammar_to_enforced_generation(_ guidedGrammar: String) throws -> EnforcedStructuredGeneration {
        if guidedGrammar.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw OpenAiStructuredOutputsValidationError.guidedGrammarNotEnforced;
        }
        throw OpenAiStructuredOutputsValidationError.guidedGrammarNotEnforced;
    }
}

/// Why extra-body structured generation was rejected.
public enum OpenAiStructuredOutputsValidationError: Error, Equatable {

    case multipleOrEmptyFields;
    case emptyChoice;
    case regexPatternTooLarge(maximumPatternBytes: Int);
    case regexNotCompilable(reason: String);
    case grammarNotEnforced;
    case guidedGrammarNotEnforced;
    case jsonSchemaMustBeObject;
    case conflictingExtraBodyFields;

    public var errorDescription: String? {
        switch self {
        case .multipleOrEmptyFields:
            return "structured_outputs must set exactly one of json, regex, choice, or grammar";
        case .emptyChoice:
            return "structured_outputs.choice must contain at least one non-empty string";
        case let .regexPatternTooLarge(maximumPatternBytes):
            return "structured_outputs.regex pattern exceeds the bounded \(maximumPatternBytes)-byte limit";
        case let .regexNotCompilable(reason):
            return "structured_outputs.regex pattern is not a supported regular expression: \(reason)";
        case .grammarNotEnforced:
            return "structured_outputs.grammar cannot be enforced yet";
        case .guidedGrammarNotEnforced:
            return "guided_grammar cannot be enforced yet";
        case .jsonSchemaMustBeObject:
            return "structured_outputs.json must be an object schema";
        case .conflictingExtraBodyFields:
            return "set only one of structured_outputs or guided_grammar";
        }
    }
}
