import Foundation;

import Testing;

import ModelServing;

/// Qwen3.5 artifact validation failures must attribute their cause: the
/// public load-failure reason names the file, tensor, and violated rule
/// instead of a generic wrapper, without leaking local paths, and stays
/// bounded because file names, tensor names, and dtype strings come from
/// untrusted model directories, port of
/// crates/model-serving/tests/hermetic/artifact_public_reason.rs.
@Suite
final class ArtifactPublicReasonTests {

    @Test
    func shouldNameTheDtypeRuleAWeightTensorViolates() {
        let validationError = ArtifactValidationError.tensorDtypeMismatch(
            tensorName: "language_model.model.layers.3.linear_attn.out_proj.weight",
            expectedDtype: .uint32,
            actualDtype: "BF16");

        let publicFailureReason = publicArtifactFailureReason(validationError);

        #expect(
            publicFailureReason.contains("language_model.model.layers.3.linear_attn.out_proj.weight"));
        #expect(publicFailureReason.contains("dtype"));
    }

    @Test
    func shouldNameTheShapeRuleAWeightTensorViolates() {
        let validationError = ArtifactValidationError.tensorShapeMismatch(
            tensorName: "language_model.model.layers.0.self_attn.q_proj.weight",
            expectedShape: [2048, 384],
            actualShape: [2048, 512]);

        let publicFailureReason = publicArtifactFailureReason(validationError);

        #expect(
            publicFailureReason.contains("language_model.model.layers.0.self_attn.q_proj.weight"));
        #expect(publicFailureReason.contains("shape"));
    }

    @Test
    func shouldNameTheMissingTensorAndItsWeightFile() {
        let validationError = ArtifactValidationError.tensorMissing(
            tensorName: "language_model.lm_head.scales",
            fileName: "model-00006-of-00006.safetensors");

        let publicFailureReason = publicArtifactFailureReason(validationError);

        #expect(publicFailureReason.contains("language_model.lm_head.scales"));
        #expect(publicFailureReason.contains("model-00006-of-00006.safetensors"));
    }

    @Test
    func shouldNameAnUnexpectedTensorWithoutALocalPath() {
        let validationError = ArtifactValidationError.unexpectedTensor(
            tensorName: "language_model.model.layers.0.mystery.weight");

        let publicFailureReason = publicArtifactFailureReason(validationError);

        #expect(
            publicFailureReason.contains("language_model.model.layers.0.mystery.weight"));
        #expect(!publicFailureReason.contains("/"));
    }

    @Test
    func shouldNameAnUnrecognizedDtypeAndItsTensor() {
        let validationError = ArtifactValidationError.unknownSafetensorsDtype(
            fileName: "model-00001-of-00006.safetensors",
            tensorName: "language_model.model.embed_tokens.weight",
            dtypeString: "F2X");

        let publicFailureReason = publicArtifactFailureReason(validationError);

        #expect(
            publicFailureReason.contains("language_model.model.embed_tokens.weight"));
        #expect(publicFailureReason.contains("F2X"));
    }

    @Test
    func shouldNameATruncatedWeightFile() {
        let validationError = ArtifactValidationError.truncatedSafetensorsFile(
            fileName: "model-00003-of-00006.safetensors",
            expectedMinimumBytes: 5_000,
            actualFileSizeBytes: 1_000);

        let publicFailureReason = publicArtifactFailureReason(validationError);

        #expect(publicFailureReason.contains("model-00003-of-00006.safetensors"));
        #expect(publicFailureReason.contains("truncated"));
    }

    @Test
    func shouldNameARequiredFileThatDoesNotMatchItsDeclaredSize() {
        let validationError = ArtifactValidationError.requiredFileSizeMismatch(
            fileName: "tokenizer.json",
            expectedSizeBytes: 100,
            actualSizeBytes: 40);

        let publicFailureReason = publicArtifactFailureReason(validationError);

        #expect(publicFailureReason.contains("tokenizer.json"));
        #expect(!publicFailureReason.contains("/"));
    }

    @Test
    func shouldKeepTheMissingDirectoryReasonFreeOfLocalPaths() {
        let localModelDirectory = "/private/models/example-model";
        let validationError = ArtifactValidationError.modelDirectoryNotFound(
            modelDirectory: localModelDirectory);

        let publicFailureReason = publicArtifactFailureReason(validationError);

        #expect(!publicFailureReason.contains(localModelDirectory));
        #expect(
            publicFailureReason.contains("directory"),
            "the reason should still explain that the directory was missing: \(publicFailureReason)");
    }

    @Test
    func shouldBoundThePublicReasonForUntrustedTensorNames() {
        let untrustedTensorName = String(repeating: "untrusted-tensor-", count: 200);
        let validationError = ArtifactValidationError.unexpectedTensor(
            tensorName: untrustedTensorName);

        let publicFailureReason = publicArtifactFailureReason(validationError);

        #expect(publicFailureReason.count <= 512);
        #expect(publicFailureReason.hasSuffix("…"));
    }

    @Test
    func shouldNameTheSafetensorsCauseThroughTheQwenWrapper() {
        let validationError = Qwen35ArtifactValidationError.artifact(
            .tensorMissing(
                tensorName: "language_model.lm_head.weight",
                fileName: "model-00006-of-00006.safetensors"));

        let publicFailureReason = validationError.publicFailureReason();

        #expect(publicFailureReason.hasPrefix("Qwen3.5 artifact validation failed"));
        #expect(publicFailureReason.contains("language_model.lm_head.weight"));
        #expect(!publicFailureReason.contains("/"));
    }

    @Test
    func shouldPreserveTheExistingConfigFamilyPublicReasonThroughTheQwenWrapper() {
        let validationError = Qwen35ArtifactValidationError.artifact(
            .requiredFileSizeMismatch(
                fileName: "config.json",
                expectedSizeBytes: 100,
                actualSizeBytes: 40));

        #expect(validationError.publicFailureReason().contains("config.json"));
    }

    /// Wraps the validation error the way the worker's public load-failure
    /// reason composition will, so the tests exercise the composed string
    /// rather than a production-only accessor.
    private func publicArtifactFailureReason(
        _ validationError: ArtifactValidationError
    ) -> String {
        return Qwen35ArtifactValidationError.artifact(validationError).publicFailureReason();
    }
}
