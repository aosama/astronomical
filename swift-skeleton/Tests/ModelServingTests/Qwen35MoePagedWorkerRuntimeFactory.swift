import Foundation;

import IpcProtocol;
import ModelServing;

@testable import ModelServing;

/**
 * Factory pairing the paged Qwen3.5-MoE engine with its journey chat
 * processor: the engine is the pinned in-memory fixture engine with the
 * fixture paging plan installed, and its page source is seeded from a fully
 * resident twin loaded from the same pinned weights, so the worker loop
 * drives a real paged MoE forward pass hermetically.
 */
struct Qwen35MoePagedWorkerRuntimeFactory: ChatModelRuntimeFactory {

    func createChatRuntime(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration
    ) throws -> LoadedChatRuntime {
        return try self.createChatRuntimeCandidate(
            modelDirectory: modelDirectory,
            modelConfiguration: modelConfiguration).load();
    }

    func createChatRuntimeCandidate(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration
    ) throws -> ChatRuntimeCandidate {
        if modelDirectory.hasSuffix(Qwen35MoeWorkerRuntimeFactory.MODEL_DIRECTORY_SUFFIX) == false {
            throw InferenceEngineError.modelLoad(
                reason: "the paged MoE journey factory cannot load \(modelDirectory)");
        }
        return ChatRuntimeCandidate {
            let residentTwinEngine: Qwen35MoeEngine = try Qwen35MoeInMemoryEngineFixture
                .makePinnedEngine();
            let pageSource: EngineSwitchGluPageSourceFixture = try EngineSwitchGluPageSourceFixture(
                residentEngine: residentTwinEngine,
                layerCount: Int(Qwen35MoeInMemoryEngineFixture.FIXTURE_LAYER_COUNT),
                expertCount: Int(Qwen35MoeInMemoryEngineFixture.FIXTURE_EXPERT_COUNT));
            let pagedEngine: Qwen35MoeEngine = try Qwen35MoeInMemoryEngineFixture
                .makePinnedPagedEngine(
                    retainedExpertIdsPerLayer:
                        Qwen35MoeInMemoryEngineFixture.retainedExpertIdsPerLayer(),
                    expertPageMaterializer: pageSource);
            return LoadedChatRuntime(
                processor: Qwen35MoeWorkerChatProcessor(),
                engine: pagedEngine);
        };
    }
}
