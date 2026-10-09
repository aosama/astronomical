import Foundation;

import MLX;
import Testing;

import JourneyCategories;
import ModelServing;

extension MlxGpuJourneyContainer {

    @Suite(.serialized, .tags(.realModelJourney),
        .enabled(if: RealModelJourneyGate.installedArtifactDirectory(
            environmentVariableName: "ASTRONOMICAL_ORNITH35_MOE_ARTIFACT_DIRECTORY") != nil))
    final class Ornith35ArtifactValidationJourneyTests {

        @Test(.timeLimit(.minutes(2)))
        func should_validate_only_the_ornith_artifact_headers_with_attribution() throws {
            let modelDirectory: String = try #require(
                RealModelJourneyGate.installedArtifactDirectory(
                    environmentVariableName: "ASTRONOMICAL_ORNITH35_MOE_ARTIFACT_DIRECTORY"),
                "the Ornith artifact directory must resolve");
            Self.emitProgress("validating config, index, and bounded shard headers only");
            let validationStart: ContinuousClock.Instant = ContinuousClock.now;
            let validatedArtifact: ValidatedQwen35Artifact = try Qwen35ArtifactValidator()
                .validate(
                    modelDirectory: modelDirectory,
                    maxOutputTokens: 128,
                    performanceAttributionEnabled: true);
            let elapsedSeconds: Double = Self.elapsedSeconds(since: validationStart);
            let expertFootprint: (totalPayloadBytes: UInt64,
                largestGateUpFusionTransientBytes: UInt64) =
                validatedArtifact.sparseExpertPayloadFootprint(
                    canonicalTensorNames: Self.routedExpertTensorNames(
                        validatedArtifact: validatedArtifact));
            let residentTensorNames: Set<String> = Set(Qwen3_5TensorSpec
                .qwen3_5ResidentLanguageTensorProfiles(
                    qwen3_5Config: validatedArtifact.config())
                .map({ (tensorProfile: TensorProfile) -> String in
                    return tensorProfile.name;
                }));
            let residentPayloadBytes: UInt64 = try #require(
                validatedArtifact.payloadByteCount(canonicalTensorNames: residentTensorNames));
            let maximumContextTokenCount: UInt32 = 24_576;
            let contextWindowReserveBytes: UInt64 = try #require(
                Qwen35MoeArtifactExpertResidencyPolicy.contextWindowReserveBytes(
                    fullAttentionLayerCount: UInt64(
                        validatedArtifact.config().fullAttentionDecoderLayerIndexes().count),
                    keyValueHeadCount: UInt64(validatedArtifact.config().keyValueHeadCount()),
                    headDimension: UInt64(validatedArtifact.config().headDimension()),
                    bytesPerElement: validatedArtifact.config().activationDtype() == "float32"
                        ? 4 : 2,
                    artifactMaximumPositionCount: UInt64(
                        validatedArtifact.config().maximumPositionCount()),
                    maximumContextTokenCount: maximumContextTokenCount));
            let mlxMemoryCeilingBytes: UInt64 = UInt64(
                MLX.GPU.maxRecommendedWorkingSetBytes() ?? 0);
            let residency: Qwen35MoeArtifactExpertResidency =
                Qwen35MoeArtifactExpertResidencyPolicy.decide(
                    residentPayloadBytes: residentPayloadBytes,
                    expertPayloadBytes: expertFootprint.totalPayloadBytes,
                    contextWindowReserveBytes: contextWindowReserveBytes,
                    activationHeadroomBytes: 0,
                    largestGateUpFusionTransientBytes:
                        expertFootprint.largestGateUpFusionTransientBytes,
                    mlxMemoryCeilingBytes: mlxMemoryCeilingBytes);
            print("[ornith-validation] elapsed_seconds=\(String(format: "%.3f", elapsedSeconds)) "
                + "shard_count=\(validatedArtifact.shardCount()) "
                + "indexed_payload_bytes=\(validatedArtifact.totalPayloadBytes()) "
                + "routed_expert_payload_bytes=\(expertFootprint.totalPayloadBytes) "
                + "resident_core_payload_bytes=\(residentPayloadBytes) "
                + "context_reserve_bytes=\(contextWindowReserveBytes) "
                + "mlx_ceiling_bytes=\(mlxMemoryCeilingBytes) "
                + "largest_gate_up_transient_bytes=\(expertFootprint.largestGateUpFusionTransientBytes) "
                + "residency=\(residency)");
            #expect(validatedArtifact.shardCount() > 0);
            #expect(expertFootprint.totalPayloadBytes > 0);
            #expect(elapsedSeconds > 0);
        }

        private static func routedExpertTensorNames(
            validatedArtifact: ValidatedQwen35Artifact
        ) -> Set<String> {
            return Set(validatedArtifact.shardIndex().languageTensorNameToShardFileName()
                .map({ (tensorLocation: (tensorName: String, shardFileName: String)) -> String in
                    return tensorLocation.tensorName;
                })
                .filter({ (tensorName: String) -> Bool in
                    return tensorName.contains(".mlp.switch_mlp.");
                }));
        }

        private static func elapsedSeconds(since start: ContinuousClock.Instant) -> Double {
            let duration: Duration = start.duration(to: ContinuousClock.now);
            return Double(duration.components.seconds)
                + Double(duration.components.attoseconds) * 1e-18;
        }

        private static func emitProgress(_ message: String) -> Void {
            print("[ornith-validation-progress] \(message)");
            fflush(stdout);
        }
    }
}
