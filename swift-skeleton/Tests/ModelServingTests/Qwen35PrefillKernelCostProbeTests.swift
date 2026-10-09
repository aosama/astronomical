import Foundation;

import Testing;

import MLX;
import MLXLLM;
import MLXLMCommon;
import MLXNN;

import IpcProtocol;
import ModelServing;
import ModelServingTestSupport;
import JourneyCategories;

@testable import ModelServing
/**
 * Kernel-granularity prefill attribution at production shapes: times the two
 * dominant per-layer kernels of one 2,048-token prompt chunk — the
 * gated-delta recurrence and the quantized routed-expert projections — with
 * the 35B OptiQ geometry (40 layers: 30 gated-delta, 10 full attention; 256
 * routed experts, top 8). No model loads; the tensors are synthesized. The
 * printed seconds-per-call figures feed the parity work in issue #1091: the
 * per-chunk budget is the measured gate chunk time divided across these
 * calls, so the probe names which kernel family owns the remaining gap
 * before any kernel port starts.
 */
extension MlxGpuJourneyContainer {

    @Suite(.tags(.hermeticMlxJourney))
    final class Qwen35PrefillKernelCostProbeTests {

        private static let CHUNK_TOKEN_COUNT: Int = 2_048;
        private static let GATED_DELTA_KEY_HEAD_COUNT: Int = 16;
        private static let GATED_DELTA_VALUE_HEAD_COUNT: Int = 32;
        private static let GATED_DELTA_HEAD_DIMENSION: Int = 128;
        private static let EXPERT_COUNT: Int = 256;
        private static let EXPERTS_PER_TOKEN: Int = 8;
        private static let HIDDEN_SIZE: Int = 2_048;
        private static let EXPERT_INTERMEDIATE_SIZE: Int = 512;

        init() {
            signal(SIGPIPE, SIG_IGN);
            MLXMetallibLocator.overrideMetallibPathIfNecessary();
        }

        @Test(.timeLimit(.minutes(2)))
        func should_report_gated_delta_ops_seconds_per_production_chunk() throws {
            let elapsedSeconds: Double = try Self.timedGatedDeltaRecurrence(useKernel: false);
            Self.printProbeLine(label: "gated_delta_ops", elapsedSeconds: elapsedSeconds);
            #expect(elapsedSeconds > 0);
        }

        @Test(.timeLimit(.minutes(2)))
        func should_report_gated_delta_kernel_seconds_per_production_chunk() throws {
            let elapsedSeconds: Double = try Self.timedGatedDeltaRecurrence(useKernel: true);
            Self.printProbeLine(label: "gated_delta_kernel", elapsedSeconds: elapsedSeconds);
            #expect(elapsedSeconds > 0);
        }

        @Test(.timeLimit(.minutes(2)))
        func should_compare_warmed_full_moe_blocks_with_resident_gate_up_fusion() throws {
            let (fixtureDirectory, layout): (URL, TinyMoeArtifactFixture.SynthesizedLayout) =
                try TinyMoeArtifactFixture.writeModelDirectory()
            defer { try? FileManager.default.removeItem(at: fixtureDirectory) }
            let configurationText: String = String(decoding: layout.configBytes, as: UTF8.self)
                .replacingOccurrences(of: "\"hidden_size\": 64", with: "\"hidden_size\": 2048")
                .replacingOccurrences(of: "\"num_experts\": 4", with: "\"num_experts\": 256")
                .replacingOccurrences(of: "\"num_experts_per_tok\": 2", with: "\"num_experts_per_tok\": 8")
                .replacingOccurrences(of: "\"moe_intermediate_size\": 64", with: "\"moe_intermediate_size\": 512")
                .replacingOccurrences(of: "\"shared_expert_intermediate_size\": 64", with: "\"shared_expert_intermediate_size\": 512")
            let configuration: Qwen35Configuration = try JSONDecoder().decode(
                Qwen35Configuration.self, from: Data(configurationText.utf8))
            let model = Qwen35MoEModel(configuration)
            let modules: [String: Module] = Dictionary(uniqueKeysWithValues: model.namedModules())
            let original: Module = try #require(modules["language_model.model.layers.0.mlp"])
            MLXNN.quantize(model: original, groupSize: 64, bits: 4)
            original.train(false)
            let children: [String: Module] = Dictionary(uniqueKeysWithValues: original.namedModules())
            let experts: SwitchGLU = try #require(children["switch_mlp"] as? SwitchGLU)
            let fusion: Qwen35ResidentGateUpFusion = try #require(
                try Qwen35ResidentGateUpFusion.make(
                    experts: experts,
                    sortedReduction: Qwen35SortedExpertReduction.probed(attributionEnabled: false)))
            let originalForward: any UnaryLayer = try #require(original as? any UnaryLayer)
            let fused = Qwen35ResidentFusedMlp(
                originalMlp: original, originalForward: originalForward, fusedExperts: fusion,
                router: try #require(children["gate"] as? Linear),
                sharedExpert: try #require(children["shared_expert"] as? any UnaryLayer),
                sharedExpertGate: try #require(children["shared_expert_gate"] as? Linear),
                topK: 8, normalize: true, attributionEnabled: false)
            let hiddenStates: MLXArray = MLXRandom.normal(
                [1, Self.CHUNK_TOKEN_COUNT, Self.HIDDEN_SIZE], dtype: .bfloat16)
            MLX.eval(hiddenStates, original, fused)
            MLX.eval(originalForward(hiddenStates))
            MLX.eval(fused(hiddenStates))
            for repeatIndex in 0..<5 {
                let originalStart: ContinuousClock.Instant = ContinuousClock.now
                let originalOutput: MLXArray = originalForward(hiddenStates)
                MLX.eval(originalOutput)
                let originalSeconds: Double = Self.seconds(since: originalStart)
                let fusedStart: ContinuousClock.Instant = ContinuousClock.now
                let fusedOutput: MLXArray = fused(hiddenStates)
                MLX.eval(fusedOutput)
                let fusedSeconds: Double = Self.seconds(since: fusedStart)
                print("[prefill-fusion-probe] repeat=\(repeatIndex) original_seconds=\(originalSeconds) fused_seconds=\(fusedSeconds)")
                fflush(stdout)
                #expect(MLX.allClose(fusedOutput, originalOutput, rtol: 0.03, atol: 0.01).item(Bool.self))
            }
        }

        @Test(.timeLimit(.minutes(2)))
        func should_report_expert_projection_seconds_per_production_chunk() throws {
            let elapsedSeconds: Double = try Self.timedExpertProjections();
            Self.printProbeLine(label: "expert_projections", elapsedSeconds: elapsedSeconds);
            #expect(elapsedSeconds > 0);
        }

        private static func printProbeLine(
            label: String, elapsedSeconds: Double
        ) -> Void {
            print("[prefill-kernel-probe] chunk_tokens=\(Self.CHUNK_TOKEN_COUNT) "
                + "\(label)_seconds=\(String(format: "%.4f", elapsedSeconds))",
                terminator: "\n");
            fflush(stdout);
        }

        /// One gated-delta prefill recurrence over one full production chunk,
        /// timed end to end with a forced evaluation. Both dispatches (the
        /// fused Metal kernel and the ops composition) are measured so the
        /// probe also documents what the loader's evaluation-mode contract
        /// buys.
        private static func timedGatedDeltaRecurrence(useKernel: Bool) throws -> Double {
            let chunkTokenCount: Int = Self.CHUNK_TOKEN_COUNT;
            let queries: MLXArray = MLXRandom.normal(
                [1, chunkTokenCount, Self.GATED_DELTA_KEY_HEAD_COUNT,
                    Self.GATED_DELTA_HEAD_DIMENSION], dtype: .bfloat16);
            let keys: MLXArray = MLXRandom.normal(
                [1, chunkTokenCount, Self.GATED_DELTA_KEY_HEAD_COUNT,
                    Self.GATED_DELTA_HEAD_DIMENSION], dtype: .bfloat16);
            let values: MLXArray = MLXRandom.normal(
                [1, chunkTokenCount, Self.GATED_DELTA_VALUE_HEAD_COUNT,
                    Self.GATED_DELTA_HEAD_DIMENSION], dtype: .bfloat16);
            let updateRates: MLXArray = MLXRandom.normal(
                [1, chunkTokenCount, Self.GATED_DELTA_VALUE_HEAD_COUNT], dtype: .bfloat16);
            let decayInputs: MLXArray = MLXRandom.normal(
                [1, chunkTokenCount, Self.GATED_DELTA_VALUE_HEAD_COUNT], dtype: .bfloat16);
            let decayLogs: MLXArray = MLXRandom.uniform(
                low: 0, high: 16, [Self.GATED_DELTA_VALUE_HEAD_COUNT]);
            let decayBiases: MLXArray = MLXArray.ones([Self.GATED_DELTA_VALUE_HEAD_COUNT]);
            let recurrentState: MLXArray = MLXArray.zeros(
                [1, Self.GATED_DELTA_VALUE_HEAD_COUNT, Self.GATED_DELTA_HEAD_DIMENSION,
                    Self.GATED_DELTA_HEAD_DIMENSION], dtype: .float32);
            MLX.eval(queries, keys, values, updateRates, decayInputs, decayLogs, decayBiases,
                recurrentState);
            let (warmOutput, warmState) = gatedDeltaUpdate(
                q: queries, k: keys, v: values, a: decayInputs, b: updateRates,
                aLog: decayLogs, dtBias: decayBiases, state: recurrentState,
                useKernel: useKernel);
            MLX.eval(warmOutput, warmState);
            let timedStart: ContinuousClock.Instant = ContinuousClock.now;
            let (recurrenceOutput, nextState) = gatedDeltaUpdate(
                q: queries, k: keys, v: values, a: decayInputs, b: updateRates,
                aLog: decayLogs, dtBias: decayBiases, state: recurrentState,
                useKernel: useKernel);
            MLX.eval(recurrenceOutput, nextState);
            let elapsedSeconds: Double = Self.seconds(since: timedStart);
            return elapsedSeconds;
        }

        /// One MoE layer's routed-expert execution over one full production
        /// chunk through the real upstream composition: the quantized
        /// SwitchGLU with gate, up, and down projections, gather-sort, and
        /// the per-assignment weighted reduction inputs the production model
        /// feeds it.
        private static func timedExpertProjections() throws -> Double {
            let chunkTokenCount: Int = Self.CHUNK_TOKEN_COUNT;
            let expertBlock: SwitchGLU = Self.quantizedExpertBlock();
            let tokenRows: MLXArray = MLXRandom.normal(
                [chunkTokenCount, Self.HIDDEN_SIZE], dtype: .bfloat16);
            let routedExpertIds: MLXArray = MLXRandom.uniform(
                low: 0, high: Self.EXPERT_COUNT,
                [chunkTokenCount, Self.EXPERTS_PER_TOKEN]).asType(.uint32);
            MLX.eval(tokenRows, routedExpertIds);
            let warmOutput: MLXArray = expertBlock(tokenRows, routedExpertIds);
            MLX.eval(warmOutput);
            let timedStart: ContinuousClock.Instant = ContinuousClock.now;
            let expertOutput: MLXArray = expertBlock(tokenRows, routedExpertIds);
            MLX.eval(expertOutput);
            return Self.seconds(since: timedStart);
        }

        private static func quantizedExpertBlock() -> SwitchGLU {
            let seedState: MLXRandom.RandomState = MLXRandom.RandomState(seed: 11);
            return withRandomState(seedState) {
                let expertBlock: SwitchGLU = SwitchGLU(
                    inputDims: Self.HIDDEN_SIZE,
                    hiddenDims: Self.EXPERT_INTERMEDIATE_SIZE,
                    numExperts: Self.EXPERT_COUNT);
                MLXNN.quantize(model: expertBlock) { (_: String, _: Module) -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
                    return (groupSize: 64, bits: 4, mode: .affine);
                };
                return expertBlock;
            };
        }

        private static func seconds(since startedAt: ContinuousClock.Instant) -> Double {
            let elapsed: Duration = ContinuousClock.now.duration(to: startedAt);
            let elapsedSeconds: Double = Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) / 1_000_000_000_000_000_000;
            return abs(elapsedSeconds);
        }
    }
}
