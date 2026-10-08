import Foundation;

import IpcProtocol;
import ModelServing;

@testable import ModelServing;

/**
 * Request-local translator of the MoE worker journey runtime: the prepared
 * request carries the Romeo and Juliet fixture bytes as model-visible token
 * ids inside the tiny vocabulary, the fixture end tokens close the stream,
 * and every generated token decodes to one fixture word the worker wire
 * events can assert on.
 */
final class Qwen35MoeWorkerActiveGeneration: ActiveChatGeneration {

    private var producedTokenCount: Int = 0;

    let promptTokenCount: Int;
    let preparedRequest: Qwen35PreparedInferenceRequest;

    var inferenceRequest: any PreparedInferenceRequest {
        return self.preparedRequest;
    }

    init(chatGenerationCommand: ChatGenerationCommand) {
        let promptText: String = chatGenerationCommand.messages.reduce("") {
            (promptAccumulator: String, chatMessage: ChatMessage) -> String in
            return promptAccumulator + chatMessage.plainTextContent();
        };
        let promptTokenIds: Array<UInt32> = promptText.utf8.map { (promptByte: UInt8) -> UInt32 in
            return UInt32(promptByte) % 512;
        };
        self.promptTokenCount = promptTokenIds.count;
        self.preparedRequest = Qwen35PreparedInferenceRequest(
            promptTokenIds: promptTokenIds,
            samplingSettings: Qwen35SamplingSettings(
                chatGenerationSettings: chatGenerationCommand.settings));
    }

    func isEndOfSequenceToken(_ generatedTokenId: UInt32) -> Bool {
        return Qwen35MoeInMemoryEngineFixture.FIXTURE_END_TOKEN_IDS.contains(generatedTokenId);
    }

    func translateGeneratedToken(
        _ generatedTokenId: UInt32
    ) throws -> ModelGeneratedTokenTranslation {
        self.producedTokenCount += 1;
        let fixtureWords: Array<String> = EngineBackedWorkerTests.romeoAndJulietOutputWords;
        let fixtureWord: String = fixtureWords[(self.producedTokenCount - 1) % fixtureWords.count];
        return ModelGeneratedTokenTranslation(publicOutputs: [.text(text: fixtureWord)]);
    }

    func finishOutputs() throws -> Array<ChatGenerationOutput> {
        return [];
    }
}
