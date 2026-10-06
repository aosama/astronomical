import Foundation;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

/// Validates the raw JSON schema text one IPC chat request carries before it
/// may become a worker generation constraint.
///
/// Mirrors apps/supervisor/src/daemon_ipc_chat_schema.rs + structured_output.rs.
/// The CLI stays thin: it reads the schema file and forwards the text. The
/// daemon owns validation because the IPC request is a trust boundary: an
/// oversized, unparseable, or non-object schema is rejected here with a clear
/// reason instead of reaching the worker or failing mid-generation.
public enum ChatSchemaConstraint {

    /// A user-facing rejection reason for one chat schema.
    public struct ChatSchemaRejectionError: Error {
        public let reason: String;
    }

    /// Parses and bounds the raw schema text into the worker-enforced JSON
    /// constraint. The thrown reason is a user-facing error message.
    public static func validated(
        _ schemaJson: String
    ) throws -> StructuredGenerationConstraint {
        if schemaJson.utf8.count > StructuredGenerationConstraint.maximumChatSchemaJsonBytes {
            throw ChatSchemaRejectionError(reason:
                "the --schema file is \(schemaJson.utf8.count) bytes, over the "
                + "\(StructuredGenerationConstraint.maximumChatSchemaJsonBytes)-byte limit; "
                + "send a smaller schema file");
        }
        let parsedSchema: JsonWireValue;
        do {
            parsedSchema = try JsonWireParser.parseDocument(documentBytes: Data(schemaJson.utf8));
        } catch {
            throw ChatSchemaRejectionError(reason: "the --schema file is not valid JSON: \(error)");
        }
        let enforcedGeneration: EnforcedStructuredGeneration;
        do {
            enforcedGeneration = try EnforcedStructuredGeneration.enforced_generation_from_json_schema(
                parsedSchema);
        } catch {
            throw ChatSchemaRejectionError(reason: "the --schema file was rejected: \(error)");
        }
        // Re-serializing through the constraint keeps the compact canonical
        // form the worker's DFA compiler sees and drops trailing whitespace.
        return ChatSchemaConstraint.constraintFromEnforcedGeneration(enforcedGeneration);
    }

    static func constraintFromEnforcedGeneration(
        _ enforcedStructuredGeneration: EnforcedStructuredGeneration
    ) -> StructuredGenerationConstraint {
        switch (enforcedStructuredGeneration) {
        case .jsonObject:
            return .jsonObject;
        case let .jsonSchema(schema):
            return .jsonSchema(schemaJson: (try? schema.serializedText) ?? "{}");
        case let .choice(choices):
            return .choice(choices: choices);
        case let .regex(pattern):
            return .regex(pattern: pattern);
        }
    }

    /// Prompt instruction paired with an enforced IPC schema constraint. The
    /// CLI sends a bare schema file, so there is no schema name or description
    /// to include; the wording mirrors the enforced REST instruction otherwise.
    static func enforcedSchemaOutputInstruction(schemaJson: String) -> String {
        return "Output a single JSON object matching this schema and nothing else "
            + "after any reasoning: no markdown fences and no prose. Schema: \(schemaJson)";
    }

    /// Pairs the enforced constraint with its prompt instruction: the first
    /// system message is the root instruction templates treat as such, so the
    /// schema rule is appended there; otherwise a system message leads.
    public static func insertJsonOutputInstruction(
        _ chatMessages: inout Array<ChatMessage>,
        jsonOutputInstruction: String
    ) -> Void {
        if case let .system(content) = chatMessages.first {
            chatMessages[0] = .system(content: content + "\n\n" + jsonOutputInstruction);
            return;
        }
        chatMessages.insert(.system(content: jsonOutputInstruction), at: 0);
    }
}
