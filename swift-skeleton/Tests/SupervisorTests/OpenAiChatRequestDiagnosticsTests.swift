import Foundation;

import Testing;

import IpcProtocol;
import RestContract;

@testable import Supervisor;

/**
 * Chat-diagnostics journeys: request snapshots carry only non-payload
 * metadata — byte counts, fingerprints, and role shapes — so rejections are
 * correlatable while prompt text, API keys, and secrets never reach a log.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class OpenAiChatRequestDiagnosticsTests {

    @Test
    func should_keep_only_non_payload_request_metadata_in_diagnostics() throws {
        let requestBodyText: String = "{\n"
            + "    \"model\":\"astronomical/fake-mixture-of-experts\",\n"
            + "    \"api_key\":\"body-api-secret\",\n"
            + "    \"messages\":[{\"role\":\"user\",\"content\":\"inspect the failure\\n"
            + "api_key: user-api-secret\\ncontinue safely\"}],\n"
            + "    \"stream\":true\n"
            + "}";
        let requestBodyBytes: Data = Data(requestBodyText.utf8);

        let diagnosticSnapshot: OpenAiChatRequestDiagnosticSnapshot =
            OpenAiChatRequestDiagnostics.buildRequestDiagnosticSnapshot(
                requestBodyBytes: requestBodyBytes);
        let diagnosticDebugText: String = String(describing: diagnosticSnapshot);

        #expect(diagnosticSnapshot.requestBodyBytes == requestBodyBytes.count);
        #expect(diagnosticSnapshot.requestBodySha256.count == 64);
        #expect(diagnosticDebugText.contains("inspect the failure") == false);
        #expect(diagnosticDebugText.contains("body-api-secret") == false);
        #expect(diagnosticDebugText.contains("user-api-secret") == false);
    }

    @Test
    func should_summarize_the_latest_user_message_for_info_diagnostics() throws {
        let requestBodyText: String = "{\n"
            + "    \"model\":\"astronomical/fake-mixture-of-experts\",\n"
            + "    \"messages\":[\n"
            + "        {\"role\":\"system\",\"content\":\"system\"},\n"
            + "        {\"role\":\"user\",\"content\":\"earlier\"},\n"
            + "        {\"role\":\"assistant\",\"content\":\"reply\"},\n"
            + "        {\"role\":\"user\",\"content\":\"yeah i am still thinking\"}\n"
            + "    ],\n"
            + "    \"stream\":true\n"
            + "}";

        let diagnosticSnapshot: OpenAiChatRequestInfoDiagnosticSnapshot =
            OpenAiChatRequestDiagnostics.buildRequestInfoDiagnosticSnapshot(
                requestBodyBytes: Data(requestBodyText.utf8));

        #expect(diagnosticSnapshot.messageCount == 4);
        #expect(diagnosticSnapshot.lastUserMessageCharacterCount == 24);
        #expect(diagnosticSnapshot.lastUserMessageSha256 != nil);
        #expect(String(describing: diagnosticSnapshot).contains("yeah i am still thinking") == false);
    }

    @Test
    func should_summarize_message_roles_for_translation_rejection_diagnostics() throws {
        let requestBodyText: String = "{\n"
            + "    \"model\":\"astronomical/fake-mixture-of-experts\",\n"
            + "    \"messages\":[\n"
            + "        {\"role\":\"user\",\"content\":\"earlier context\"},\n"
            + "        {\"role\":\"system\",\"content\":\"a chronological update\"},\n"
            + "        {\"role\":\"assistant\",\"content\":\"reply\"},\n"
            + "        {\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"tool output\"},\n"
            + "        {\"role\":\"untrusted role text that must not appear in logs\","
            + "\"content\":\"ignored\"}\n"
            + "    ],\n"
            + "    \"stream\":true\n"
            + "}";

        let diagnosticSnapshot: OpenAiChatRequestInfoDiagnosticSnapshot =
            OpenAiChatRequestDiagnostics.buildRequestInfoDiagnosticSnapshot(
                requestBodyBytes: Data(requestBodyText.utf8));

        #expect(diagnosticSnapshot.messageRoleSequencePreview
            == "user,system,assistant,tool,unknown");
    }

    @Test
    func should_not_retain_secret_bearing_user_message_text_in_info_diagnostics() throws {
        let requestBodyText: String = "{\n"
            + "    \"model\":\"astronomical/fake-mixture-of-experts\",\n"
            + "    \"messages\":[\n"
            + "        {\"role\":\"user\",\"content\":\"please use token: definitely-secret\\n"
            + "normal followup\"}\n"
            + "    ],\n"
            + "    \"stream\":true\n"
            + "}";

        let diagnosticSnapshot: OpenAiChatRequestInfoDiagnosticSnapshot =
            OpenAiChatRequestDiagnostics.buildRequestInfoDiagnosticSnapshot(
                requestBodyBytes: Data(requestBodyText.utf8));

        #expect(diagnosticSnapshot.lastUserMessageCharacterCount == 51);
        #expect(diagnosticSnapshot.lastUserMessageSha256 != nil);
        #expect(String(describing: diagnosticSnapshot).contains("definitely-secret") == false);
    }
}
