import Foundation

import Testing;

import AstronomicalCli;
import IpcProtocol;
import JourneyCategories;

@testable import AstronomicalCli;

/// `astronomical schema object` journeys, porting schema_command.rs.
@Suite(.tags(.hermeticJourney))
final class SchemaCommandTests {

    private static func parseSchema(
        _ arguments: Array<String>
    ) -> Result<SchemaArguments, UsageError> {
        switch (CliJourneySupport.parse(["schema", "object"] + arguments)) {
        case let .success(.schema(schemaArguments)):
            return .success(schemaArguments);
        case let .failure(usageError):
            return .failure(usageError);
        default:
            return .failure(.unknownSchemaTarget("internal"));
        }
    }

    @Test
    func should_parse_schema_object_with_basic_properties() {
        guard case let .success(schemaArguments) = SchemaCommandTests.parseSchema([
            "--name", "Thing", "--string", "label", "--int", "count",
        ]) else {
            Issue.record("a basic schema should parse");
            return;
        }
        #expect(schemaArguments.objectName == "Thing");
        #expect(schemaArguments.properties.count == 2);
        #expect(schemaArguments.properties[0].dottedPath == "label");
        #expect(schemaArguments.properties[0].kind == .string);
    }

    @Test
    func should_parse_every_property_kind() {
        guard case let .success(schemaArguments) = SchemaCommandTests.parseSchema([
            "--name", "Thing", "--string", "s", "--int", "i", "--double", "d", "--boolean", "b",
        ]) else {
            Issue.record("every kind should parse");
            return;
        }
        #expect(schemaArguments.properties.map { (property: SchemaPropertyInput) -> String in
            return property.kind.jsonTypeName;
        } == ["string", "integer", "number", "boolean"]);
    }

    @Test
    func should_apply_modifiers_to_the_preceding_property() {
        guard case let .success(schemaArguments) = SchemaCommandTests.parseSchema([
            "--name", "Thing", "--string", "s", "--array", "--optional", "--description", "words",
        ]) else {
            Issue.record("modifiers should attach to the preceding property");
            return;
        }
        #expect(schemaArguments.properties[0].isArray);
        #expect(schemaArguments.properties[0].isOptional);
        #expect(schemaArguments.properties[0].description == "words");
    }

    @Test
    func should_reject_schema_object_without_name() {
        guard case let .failure(usageError) = SchemaCommandTests.parseSchema(["--string", "s"]) else {
            Issue.record("schema without a name must be a usage error");
            return;
        }
        #expect(usageError == .schemaNameRequired);
    }

    @Test
    func should_reject_schema_object_without_properties() {
        guard case let .failure(usageError) = SchemaCommandTests.parseSchema(["--name", "Thing"]) else {
            Issue.record("schema without properties must be a usage error");
            return;
        }
        #expect(usageError == .schemaPropertyRequired);
    }

    @Test
    func should_reject_modifier_before_any_property() {
        guard case let .failure(usageError) = SchemaCommandTests.parseSchema([
            "--name", "Thing", "--array",
        ]) else {
            Issue.record("a modifier before any property must be a usage error");
            return;
        }
        #expect(usageError == .schemaModifierWithoutProperty("--array"));
    }

    @Test
    func should_reject_duplicate_property_path() {
        guard case let .failure(usageError) = SchemaCommandTests.parseSchema([
            "--name", "Thing", "--string", "s", "--int", "s",
        ]) else {
            Issue.record("a duplicate property path must be a usage error");
            return;
        }
        #expect(usageError == .schemaDuplicateProperty("s"));
    }

    @Test
    func should_reject_unknown_schema_flag() {
        guard case let .failure(usageError) = SchemaCommandTests.parseSchema([
            "--name", "Thing", "--string", "s", "--fancy",
        ]) else {
            Issue.record("an unknown schema flag must be a usage error");
            return;
        }
        #expect(usageError == .unknownArgument("--fancy"));
    }

    @Test
    func should_build_strict_object_schema_document() {
        guard case let .success(schemaArguments) = SchemaCommandTests.parseSchema([
            "--name", "Thing", "--string", "label",
        ]) else {
            Issue.record("a basic schema should parse");
            return;
        }
        let schemaDocument: Dictionary<String, Any> = SchemaCommand.buildSchemaDocument(schemaArguments);
        #expect(schemaDocument["title"] as? String == "Thing");
        #expect(schemaDocument["type"] as? String == "object");
        #expect(schemaDocument["additionalProperties"] as? Bool == false);
        #expect((schemaDocument["required"] as? Array<String>) == ["label"]);
        let properties: Dictionary<String, Any> = schemaDocument["properties"] as? Dictionary<String, Any> ?? [:];
        let labelProperty: Dictionary<String, Any> = properties["label"] as? Dictionary<String, Any> ?? [:];
        #expect(labelProperty["type"] as? String == "string");
    }

    @Test
    func should_nest_dotted_property_paths() {
        guard case let .success(schemaArguments) = SchemaCommandTests.parseSchema([
            "--name", "Thing", "--string", "meta.author.name",
        ]) else {
            Issue.record("a dotted path should parse");
            return;
        }
        let schemaDocument: Dictionary<String, Any> = SchemaCommand.buildSchemaDocument(schemaArguments);
        let properties: Dictionary<String, Any> = schemaDocument["properties"] as? Dictionary<String, Any> ?? [:];
        let metaProperty: Dictionary<String, Any> = properties["meta"] as? Dictionary<String, Any> ?? [:];
        #expect(metaProperty["type"] as? String == "object");
        #expect((metaProperty["required"] as? Array<String>) == ["author"]);
        let nestedProperties: Dictionary<String, Any> = metaProperty["properties"] as? Dictionary<String, Any> ?? [:];
        let authorProperty: Dictionary<String, Any> = nestedProperties["author"] as? Dictionary<String, Any> ?? [:];
        #expect((authorProperty["required"] as? Array<String>) == ["name"]);
    }

    @Test
    func should_exclude_optional_properties_from_required() {
        guard case let .success(schemaArguments) = SchemaCommandTests.parseSchema([
            "--name", "Thing", "--string", "a", "--string", "b", "--optional",
        ]) else {
            Issue.record("an optional property should parse");
            return;
        }
        let schemaDocument: Dictionary<String, Any> = SchemaCommand.buildSchemaDocument(schemaArguments);
        #expect((schemaDocument["required"] as? Array<String>) == ["a"]);
    }

    @Test
    func should_wrap_array_properties_in_items() {
        guard case let .success(schemaArguments) = SchemaCommandTests.parseSchema([
            "--name", "Thing", "--int", "values", "--array",
        ]) else {
            Issue.record("an array property should parse");
            return;
        }
        let schemaDocument: Dictionary<String, Any> = SchemaCommand.buildSchemaDocument(schemaArguments);
        let properties: Dictionary<String, Any> = schemaDocument["properties"] as? Dictionary<String, Any> ?? [:];
        let valuesProperty: Dictionary<String, Any> = properties["values"] as? Dictionary<String, Any> ?? [:];
        #expect(valuesProperty["type"] as? String == "array");
        let itemsProperty: Dictionary<String, Any> = valuesProperty["items"] as? Dictionary<String, Any> ?? [:];
        #expect(itemsProperty["type"] as? String == "integer");
    }

    @Test
    func should_attach_property_descriptions() {
        guard case let .success(schemaArguments) = SchemaCommandTests.parseSchema([
            "--name", "Thing", "--string", "label", "--description", "the label",
        ]) else {
            Issue.record("a described property should parse");
            return;
        }
        let schemaDocument: Dictionary<String, Any> = SchemaCommand.buildSchemaDocument(schemaArguments);
        let properties: Dictionary<String, Any> = schemaDocument["properties"] as? Dictionary<String, Any> ?? [:];
        let labelProperty: Dictionary<String, Any> = properties["label"] as? Dictionary<String, Any> ?? [:];
        #expect(labelProperty["description"] as? String == "the label");
    }

    @Test
    func should_reject_schema_without_object_noun() {
        guard case let .failure(usageError) = CliJourneySupport.parse(["schema", "array"]) else {
            Issue.record("a non-object schema target must be a usage error");
            return;
        }
        #expect(usageError == .unknownSchemaTarget("array"));
    }

    @Test
    func should_reject_leaf_property_conflicting_with_nested_path() {
        guard case let .failure(usageError) = SchemaCommandTests.parseSchema([
            "--name", "Thing", "--string", "meta", "--string", "meta.author",
        ]) else {
            Issue.record("a leaf conflicting with a nested path must be a usage error");
            return;
        }
        #expect(usageError == .schemaDuplicateProperty("meta.author"));
    }

    @Test
    func should_write_pretty_json_schema_to_stdout() throws {
        guard case let .success(schemaArguments) = SchemaCommandTests.parseSchema([
            "--name", "Thing", "--string", "label",
        ]) else {
            Issue.record("a basic schema should parse");
            return;
        }
        let renderedOutput: BufferedTextOutputWriter = BufferedTextOutputWriter();
        #expect(SchemaCommand.run(schemaArguments, renderedOutput: renderedOutput));
        let renderedText: String = renderedOutput.text;
        #expect(renderedText.hasSuffix("\n"));
        let parsedDocument: Any = try #require(try? JSONSerialization.jsonObject(with: Data(renderedText.utf8)));
        let documentObject: Dictionary<String, Any> = try #require(parsedDocument as? Dictionary<String, Any>);
        #expect(documentObject["title"] as? String == "Thing");
    }

    @Test
    func should_map_double_kind_to_json_number() {
        #expect(SchemaPropertyKind.double.jsonTypeName == "number");
    }
}

/// `--schema` input journeys for respond, porting respond_schema.rs.
@Suite(.tags(.hermeticJourney))
final class RespondSchemaInputTests {

    @Test
    func should_reject_an_oversize_schema_file() throws {
        let schemaDirectory: String = CliJourneySupport.freshTestDirectory("schema-oversize");
        defer { try? FileManager.default.removeItem(atPath: schemaDirectory) }
        let oversizeBytes: Data = Data(repeating: 0x61, count: StructuredGenerationConstraint.maximumChatSchemaJsonBytes + 1);
        let schemaPath: String = schemaDirectory + "/huge.json";
        try oversizeBytes.write(to: URL(fileURLWithPath: schemaPath));
        do {
            _ = try RespondInputs.readSchemaInput(schemaPath: schemaPath);
            Issue.record("an oversize schema file must be rejected");
        } catch let respondError as RespondError {
            guard case let .schemaTooLarge(actualBytes, maximumBytes) = respondError else {
                Issue.record("the oversize rejection must carry its bounds: \(respondError)");
                return;
            }
            #expect(actualBytes == StructuredGenerationConstraint.maximumChatSchemaJsonBytes + 1);
            #expect(maximumBytes == StructuredGenerationConstraint.maximumChatSchemaJsonBytes);
        }
    }

    @Test
    func should_reject_a_non_utf8_schema_file() throws {
        let schemaDirectory: String = CliJourneySupport.freshTestDirectory("schema-utf8");
        defer { try? FileManager.default.removeItem(atPath: schemaDirectory) }
        let schemaPath: String = schemaDirectory + "/binary.json";
        try Data([0xFF, 0xFE, 0x00]).write(to: URL(fileURLWithPath: schemaPath));
        do {
            _ = try RespondInputs.readSchemaInput(schemaPath: schemaPath);
            Issue.record("a non-UTF-8 schema file must be rejected");
        } catch let respondError as RespondError {
            guard case .schemaNotUtf8 = respondError else {
                Issue.record("the rejection must name the encoding failure: \(respondError)");
                return;
            }
        }
    }
}

/// `--image` input journeys for respond, porting the inline tests of
/// respond_image.rs.
@Suite(.tags(.hermeticJourney))
final class RespondImageInputTests {

    private func writeImageFile(
        _ directoryName: String,
        fileName: String,
        contents: [UInt8]
    ) throws -> (directory: String, path: String) {
        let imageDirectory: String = CliJourneySupport.freshTestDirectory(directoryName);
        let imagePath: String = imageDirectory + "/" + fileName;
        try Data(contents).write(to: URL(fileURLWithPath: imagePath));
        return (imageDirectory, imagePath);
    }

    @Test
    func should_map_a_png_extension_to_the_png_mime_type() {
        #expect(RespondInputs.mimeForExtension(path: "snapshot.png") == "image/png");
    }

    @Test
    func should_map_a_jpeg_extension_in_both_spellings_to_the_jpeg_mime_type() {
        #expect(RespondInputs.mimeForExtension(path: "scan.JPG") == "image/jpeg");
        #expect(RespondInputs.mimeForExtension(path: "document.jpeg") == "image/jpeg");
    }

    @Test
    func should_reject_an_unsupported_extension() {
        #expect(RespondInputs.mimeForExtension(path: "memo.txt") == nil);
        #expect(RespondInputs.mimeForExtension(path: "notes.md") == nil);
    }

    @Test
    func should_reject_a_file_with_no_extension() {
        #expect(RespondInputs.mimeForExtension(path: "image") == nil);
    }

    @Test
    func should_read_a_supported_image_into_a_decoded_input() throws {
        let (imageDirectory, imagePath): (String, String) = try writeImageFile(
            "image-read",
            fileName: "picture.png",
            contents: [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        );
        defer { try? FileManager.default.removeItem(atPath: imageDirectory) }
        let imageInput: ChatImageInput = try RespondInputs.readImageInput(path: imagePath);
        #expect(imageInput.mimeType == "image/png");
        #expect(imageInput.decodedBytes == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    }

    @Test
    func should_explain_a_missing_image_file() throws {
        let missingDirectory: String = CliJourneySupport.freshTestDirectory("image-missing");
        defer { try? FileManager.default.removeItem(atPath: missingDirectory) }
        do {
            _ = try RespondInputs.readImageInput(path: missingDirectory + "/nope.png");
            Issue.record("a missing image should surface as a read failure");
        } catch let respondError as RespondError {
            guard case .imageReadFailed = respondError else {
                Issue.record("a missing image should surface as a read failure: \(respondError)");
                return;
            }
        }
    }

    @Test
    func should_reject_an_unsupported_image_file() throws {
        let (imageDirectory, textPath): (String, String) = try writeImageFile(
            "image-unsupported",
            fileName: "memo.txt",
            contents: Array("hello".utf8)
        );
        defer { try? FileManager.default.removeItem(atPath: imageDirectory) }
        do {
            _ = try RespondInputs.readImageInput(path: textPath);
            Issue.record("a non-image file should surface as unsupported");
        } catch let respondError as RespondError {
            guard case .unsupportedImage = respondError else {
                Issue.record("a non-image file should surface as unsupported: \(respondError)");
                return;
            }
        }
    }

    @Test
    func should_enforce_the_shared_decoded_byte_budget_across_images() throws {
        let (imageDirectory, firstPath): (String, String) = try writeImageFile(
            "image-budget",
            fileName: "a.png",
            contents: [UInt8](repeating: 0, count: 10 * 1024 * 1024)
        );
        defer { try? FileManager.default.removeItem(atPath: imageDirectory) }
        let secondPath: String = imageDirectory + "/b.png";
        try Data([UInt8](repeating: 0, count: 10 * 1024 * 1020)).write(to: URL(fileURLWithPath: secondPath));
        do {
            _ = try RespondInputs.readImageInputs([firstPath, secondPath]);
            Issue.record("two images past the shared budget should surface as too large");
        } catch let respondError as RespondError {
            guard case .imageTooLarge = respondError else {
                Issue.record("two images past the shared budget should surface as too large: \(respondError)");
                return;
            }
        }
        let smallSecondPath: String = imageDirectory + "/c.png";
        try Data([UInt8](repeating: 0, count: 2 * 1024 * 1024)).write(to: URL(fileURLWithPath: smallSecondPath));
        let imageInputs: Array<ChatImageInput> = try RespondInputs.readImageInputs([firstPath, smallSecondPath]);
        #expect(imageInputs.count == 2, "both images should be read when under budget");
    }

    @Test
    func should_read_no_images_when_the_paths_are_empty() throws {
        let imageInputs: Array<ChatImageInput> = try RespondInputs.readImageInputs([]);
        #expect(imageInputs.isEmpty, "an empty request should carry no images");
    }

    @Test
    func should_accept_two_images_that_sum_exactly_to_the_budget() throws {
        let halfBudget: Int = RespondInputs.maximumTotalImageDecodedBytes / 2;
        let (imageDirectory, firstPath): (String, String) = try writeImageFile(
            "image-boundary",
            fileName: "a.png",
            contents: [UInt8](repeating: 0, count: halfBudget)
        );
        defer { try? FileManager.default.removeItem(atPath: imageDirectory) }
        let secondPath: String = imageDirectory + "/b.png";
        try Data([UInt8](repeating: 0, count: halfBudget)).write(to: URL(fileURLWithPath: secondPath));
        let imageInputs: Array<ChatImageInput> = try RespondInputs.readImageInputs([firstPath, secondPath]);
        #expect(imageInputs.count == 2, "two images summing exactly to the budget should both read");
    }
}

/// Inline lifecycle helper journeys, porting the unit tests of
/// model_lifecycle.rs.
@Suite(.tags(.hermeticJourney))
final class ModelLifecycleHelperTests {

    private static func catalogEntry(
        huggingfaceId: String,
        requestableModelId: String?
    ) -> DaemonCatalogEntry {
        return DaemonCatalogEntry(
            huggingfaceId: huggingfaceId,
            displayName: huggingfaceId,
            family: "qwen",
            approximateSizeBytes: 1_000_000_000,
            readyOnThisMac: false,
            requestableModelId: requestableModelId,
            downloadState: nil,
            contextWindow: 32_768,
            supportsReasoning: false,
            supportsVision: false,
            supportsToolCalls: false,
            supportsImageGeneration: false,
            supportsEmbeddings: false
        );
    }

    @Test
    func should_match_catalog_entries_by_huggingface_id_or_leaf_even_when_not_ready() {
        let readyEntry: DaemonCatalogEntry = ModelLifecycleHelperTests.catalogEntry(
            huggingfaceId: "mlx-community/Qwen3.5-2B-4bit",
            requestableModelId: "Qwen3.5-2B-4bit"
        );
        #expect(ModelLifecycle.catalogEntryMatches(readyEntry, requestedModelId: "mlx-community/Qwen3.5-2B-4bit"));
        #expect(ModelLifecycle.catalogEntryMatches(readyEntry, requestedModelId: "Qwen3.5-2B-4bit"));
        #expect(ModelLifecycle.catalogEntryMatches(readyEntry, requestedModelId: "other-ns/Qwen3.5-2B-4bit"));
        #expect(!ModelLifecycle.catalogEntryMatches(readyEntry, requestedModelId: "Qwen3.5-4B-4bit"));
        // Not-ready entries carry no requestable id on the wire; the leaf
        // must still match because the daemon derives it from the hf id.
        let notReadyEntry: DaemonCatalogEntry = ModelLifecycleHelperTests.catalogEntry(
            huggingfaceId: "mlx-community/Qwen3.5-2B-4bit",
            requestableModelId: nil
        );
        #expect(ModelLifecycle.catalogEntryMatches(notReadyEntry, requestedModelId: "Qwen3.5-2B-4bit"));
    }

    @Test
    func should_render_download_progress_with_decimal_gigabytes() {
        let downloadJob: DaemonDownloadJob = DaemonDownloadJob(
            huggingfaceId: "example/2gb-model",
            state: "downloading",
            bytesCompleted: 1_500_000_000,
            bytesTotal: 2_000_000_000,
            error: nil
        );
        #expect(ModelLifecycle.renderDownloadProgress(downloadJob) == "example/2gb-model: downloading 1.5 GB / 2 GB (75%)");
    }

    @Test
    func should_render_download_progress_without_a_total_only_shows_state() {
        let downloadJob: DaemonDownloadJob = DaemonDownloadJob(
            huggingfaceId: "example/unknown-size",
            state: "fetching_manifest",
            bytesCompleted: 0,
            bytesTotal: 0,
            error: nil
        );
        #expect(ModelLifecycle.renderDownloadProgress(downloadJob) == "example/unknown-size: fetching_manifest");
    }

    @Test
    func should_list_near_matches_and_the_supported_hint_in_unknown_model_reason() {
        let installedModels: Array<DaemonListedModel> = [
            DaemonListedModel(
                modelId: "Qwen3.5-2B-4bit",
                family: "qwen3_5",
                contextWindow: 32_768,
                supportsEmbeddings: false,
                isResident: false,
                sizeBytes: 1_500_000_000
            ),
        ];
        let reason: String = ModelLifecycle.unknownModelReason(
            requestedModelId: "Qwen3.5-14B-4bit",
            installedModels: installedModels,
            requiredCapability: .chat
        );
        #expect(reason.contains("did you mean: qwen3.5-2b-4bit"));
        #expect(reason.contains("astronomical models supported"));
    }
}
