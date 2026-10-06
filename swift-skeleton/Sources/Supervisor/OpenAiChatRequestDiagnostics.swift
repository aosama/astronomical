import Foundation;

import CryptoKit;
import IpcProtocol;
import RestContract;
import os;

/// Request data captured for trace-level OpenAI chat diagnostics: only the
/// byte count and a fingerprint of the raw body, never the payload itself.
struct OpenAiChatRequestDiagnosticSnapshot: Equatable, CustomStringConvertible {

    /// Number of raw HTTP body bytes received from the client.
    let requestBodyBytes: Int;
    /// SHA-256 fingerprint of raw request bytes for request correlation.
    let requestBodySha256: String;

    var description: String {
        return "OpenAiChatRequestDiagnosticSnapshot("
            + "request_body_bytes: \(self.requestBodyBytes), "
            + "request_body_sha256: \(self.requestBodySha256))";
    }
}

/// Bounded request data that is safe enough for default info-level request
/// correlation: message counts and role shapes, with user text reduced to a
/// character count and a fingerprint so secrets never reach a log.
struct OpenAiChatRequestInfoDiagnosticSnapshot: Equatable, CustomStringConvertible {

    let messageCount: Int?;
    let messageRoleSequencePreview: String?;
    let lastUserMessageCharacterCount: Int?;
    let lastUserMessageSha256: String?;

    var description: String {
        return "OpenAiChatRequestInfoDiagnosticSnapshot("
            + "message_count: \(Self.optionalText(self.messageCount)), "
            + "message_role_sequence_preview: \(Self.optionalText(self.messageRoleSequencePreview)), "
            + "last_user_message_character_count: \(Self.optionalText(self.lastUserMessageCharacterCount)), "
            + "last_user_message_sha256: \(Self.optionalText(self.lastUserMessageSha256)))";
    }

    private static func optionalText(_ optionalValue: Any?) -> String {
        guard let unwrappedValue: Any = optionalValue else {
            return "None";
        }
        return "Some(\(unwrappedValue))";
    }
}

/// Builds the bounded request snapshots the REST chat endpoint writes to
/// trace, debug, and warn logs before validation mutates the data.
///
/// Port of apps/supervisor/src/chat_diagnostics.rs. The role preview and
/// fingerprints exist so rejections are correlatable without ever retaining
/// prompt text, API keys, or other user content.
enum OpenAiChatRequestDiagnostics {

    private static let diagnosticsLog: Logger = Logger(subsystem: "dev.astronomical.supervisor", category: "chat-diagnostics");
    private static let messageRoleSequencePreviewLimit: Int = 16;

    /** Builds the request snapshot written to trace logs before validation. */
    static func buildRequestDiagnosticSnapshot(
        requestBodyBytes: Data
    ) -> OpenAiChatRequestDiagnosticSnapshot {
        return OpenAiChatRequestDiagnosticSnapshot(
            requestBodyBytes: requestBodyBytes.count,
            requestBodySha256: OpenAiChatRequestDiagnostics.sha256Hex(requestBodyBytes));
    }

    /** Builds a compact request summary for info logs without dumping the
    full prompt. */
    static func buildRequestInfoDiagnosticSnapshot(
        requestBodyBytes: Data
    ) -> OpenAiChatRequestInfoDiagnosticSnapshot {
        let requestWireValue: JsonWireValue;
        do {
            requestWireValue = try JsonWireParser.parseDocument(documentBytes: requestBodyBytes);
        } catch {
            return OpenAiChatRequestDiagnostics.emptyInfoDiagnosticSnapshot();
        }
        guard case let .object(requestObject) = requestWireValue,
              let messagesWireValue: JsonWireValue = requestObject.value(forKey: "messages"),
              case let .array(messagesJsonValues) = messagesWireValue else {
            return OpenAiChatRequestDiagnostics.emptyInfoDiagnosticSnapshot();
        }
        var lastUserMessageContent: String? = nil;
        for messageJsonValue: JsonWireValue in messagesJsonValues.reversed() {
            guard case let .object(messageObject) = messageJsonValue,
                  let roleWireValue: JsonWireValue = messageObject.value(forKey: "role"),
                  case let .string(roleText) = roleWireValue,
                  roleText == "user" else {
                continue;
            }
            if let contentWireValue: JsonWireValue = messageObject.value(forKey: "content") {
                lastUserMessageContent = OpenAiChatRequestDiagnostics.extractTextFromOpenAiContentValue(
                    contentWireValue);
            }
            break;
        }
        return OpenAiChatRequestInfoDiagnosticSnapshot(
            messageCount: messagesJsonValues.count,
            messageRoleSequencePreview: OpenAiChatRequestDiagnostics.summarizeMessageRoles(
                messagesJsonValues),
            lastUserMessageCharacterCount: lastUserMessageContent.map(
                { (messageContent: String) -> Int in
                    // Rust counts chars() (Unicode scalars), not graphemes.
                    return messageContent.unicodeScalars.count;
                }),
            lastUserMessageSha256: lastUserMessageContent.map(
                { (messageContent: String) -> String in
                    return OpenAiChatRequestDiagnostics.sha256Hex(Data(messageContent.utf8));
                }));
    }

    private static func emptyInfoDiagnosticSnapshot() -> OpenAiChatRequestInfoDiagnosticSnapshot {
        return OpenAiChatRequestInfoDiagnosticSnapshot(
            messageCount: nil,
            messageRoleSequencePreview: nil,
            lastUserMessageCharacterCount: nil,
            lastUserMessageSha256: nil);
    }

    private static func summarizeMessageRoles(
        _ messagesJsonValues: Array<JsonWireValue>
    ) -> String {
        var messageRoleSequencePreview: String = String();
        for (messageIndex, messageJsonValue) in messagesJsonValues.prefix(
            messageRoleSequencePreviewLimit).enumerated() {
            if messageIndex > 0 {
                messageRoleSequencePreview += ",";
            }
            messageRoleSequencePreview += OpenAiChatRequestDiagnostics.messageRoleForDiagnostics(
                messageJsonValue);
        }
        if messagesJsonValues.count > messageRoleSequencePreviewLimit {
            messageRoleSequencePreview += ",...";
        }
        return messageRoleSequencePreview;
    }

    private static func messageRoleForDiagnostics(
        _ messageJsonValue: JsonWireValue
    ) -> String {
        guard case let .object(messageObject) = messageJsonValue,
              let roleWireValue: JsonWireValue = messageObject.value(forKey: "role"),
              case let .string(roleText) = roleWireValue else {
            return "unknown";
        }
        switch (roleText) {
        case "system": return "system";
        case "user": return "user";
        case "assistant": return "assistant";
        case "tool": return "tool";
        default: return "unknown";
        }
    }

    private static func extractTextFromOpenAiContentValue(
        _ contentJsonValue: JsonWireValue
    ) -> String {
        if case let .string(contentText) = contentJsonValue {
            return contentText;
        }
        guard case let .array(contentParts) = contentJsonValue else {
            return String();
        }
        var combinedTextContent: String = String();
        for contentPart: JsonWireValue in contentParts {
            guard case let .object(contentPartObject) = contentPart,
                  let partTypeWireValue: JsonWireValue = contentPartObject.value(forKey: "type"),
                  case let .string(partTypeText) = partTypeWireValue,
                  partTypeText == "text",
                  let textWireValue: JsonWireValue = contentPartObject.value(forKey: "text"),
                  case let .string(textContentPart) = textWireValue else {
                continue;
            }
            combinedTextContent += textContentPart;
        }
        return combinedTextContent;
    }

    private static func sha256Hex(_ bytes: Data) -> String {
        let sha256Digest: SHA256Digest = SHA256.hash(data: bytes);
        return sha256Digest.compactMap({ (digestByte: UInt8) -> String in
            return String(format: "%02x", digestByte);
        }).joined();
    }

    /** Emits the trace-level capture line for one admitted chat request. */
    static func logRequestCapture(_ diagnosticSnapshot: OpenAiChatRequestDiagnosticSnapshot) -> Void {
        diagnosticsLog.trace("""
            captured REST chat completion request metadata for diagnostics \
            request_body_bytes=\(diagnosticSnapshot.requestBodyBytes, privacy: .public) \
            request_body_sha256=\(diagnosticSnapshot.requestBodySha256, privacy: .public)
            """);
    }

    /** Emits the info-level rejection line carrying only bounded shapes. */
    static func logRequestRejection(
        reason: String,
        diagnosticSnapshot: OpenAiChatRequestDiagnosticSnapshot,
        infoDiagnosticSnapshot: OpenAiChatRequestInfoDiagnosticSnapshot
    ) -> Void {
        diagnosticsLog.warning("""
            rejected invalid REST chat completion request \
            reason=\(reason, privacy: .public) \
            request_body_bytes=\(diagnosticSnapshot.requestBodyBytes, privacy: .public) \
            request_body_sha256=\(diagnosticSnapshot.requestBodySha256, privacy: .public) \
            message_count=\(infoDiagnosticSnapshot.messageCount.map(String.init) ?? "None", privacy: .public) \
            message_role_sequence_preview=\(infoDiagnosticSnapshot.messageRoleSequencePreview ?? "None", privacy: .public)
            """);
    }
}
