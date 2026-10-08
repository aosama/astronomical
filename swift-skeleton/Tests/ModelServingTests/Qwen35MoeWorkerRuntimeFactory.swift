import Foundation;

import IpcProtocol;
import ModelServing;

@testable import ModelServing;

/**
 * Factory pairing the resident Qwen3.5-MoE engine with its journey chat
 * processor: directories that do not name the loadable MoE journey model
 * fail the swap the way real factories fail a rejected selection. The
 * engine is the pinned in-memory fixture engine, so the worker loop drives
 * a real MoE forward pass hermetically.
 */
struct Qwen35MoeWorkerRuntimeFactory: ChatModelRuntimeFactory {

    static let MODEL_DIRECTORY_SUFFIX: String = "/qwen3.5-moe";

    func createChatRuntime(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration
    ) throws -> LoadedChatRuntime {
        if modelDirectory.hasSuffix(Self.MODEL_DIRECTORY_SUFFIX) == false {
            throw InferenceEngineError.modelLoad(
                reason: "the MoE journey factory cannot load \(modelDirectory)");
        }
        let pinnedEngine: Qwen35MoeEngine = try Qwen35MoeInMemoryEngineFixture.makePinnedEngine();
        return LoadedChatRuntime(
            processor: Qwen35MoeWorkerChatProcessor(),
            engine: pinnedEngine);
    }
}
