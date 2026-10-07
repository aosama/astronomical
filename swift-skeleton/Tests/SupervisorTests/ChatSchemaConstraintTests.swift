import Foundation;

import Testing;

import IpcProtocol;
import JourneyCategories;

@testable import Supervisor;

/**
 * Acceptance journey for direct chat-schema validation, migrating
 * should_validate_chat_schema_constraints_directly from
 * apps/supervisor/tests/hermetic/daemon_ipc_schema.rs: the daemon bounds the
 * schema text, rejects non-object documents, and re-serializes a valid
 * object schema into the compact canonical form the worker compiles.
 */
@Suite(.tags(.hermeticJourney))
final class ChatSchemaConstraintTests {

    @Test
    func should_validate_chat_schema_constraints_directly() throws -> Void {
        let oversizeSchemaText: String = "{\"values\":[\"" + String(repeating: "x", count: 70_000) + "\"]}";
        let oversizeRejection: ChatSchemaConstraint.ChatSchemaRejectionError = try #require(
            ChatSchemaConstraintTests.validatedRejection(for: oversizeSchemaText),
            "an oversize schema must reject before parsing");
        #expect(oversizeRejection.reason.contains("65536-byte limit") == true,
            "the oversize reason should state the bound: \(oversizeRejection.reason)");

        let arraySchemaRejection: ChatSchemaConstraint.ChatSchemaRejectionError = try #require(
            ChatSchemaConstraintTests.validatedRejection(for: "[1,2,3]"),
            "a non-object schema must reject");
        #expect(arraySchemaRejection.reason.contains("object schema") == true,
            "the non-object reason should state the object rule: \(arraySchemaRejection.reason)");

        let validatedConstraint: StructuredGenerationConstraint = try ChatSchemaConstraint.validated(
            "{ \"type\": \"object\" }");
        #expect(validatedConstraint == .jsonSchema(schemaJson: "{\"type\":\"object\"}"),
            "the daemon should re-serialize the schema canonically");
    }

    private static func validatedRejection(
        for schemaJson: String
    ) -> ChatSchemaConstraint.ChatSchemaRejectionError? {
        do {
            _ = try ChatSchemaConstraint.validated(schemaJson);
            return nil;
        } catch let rejectionError as ChatSchemaConstraint.ChatSchemaRejectionError {
            return rejectionError;
        } catch {
            return nil;
        }
    }
}
