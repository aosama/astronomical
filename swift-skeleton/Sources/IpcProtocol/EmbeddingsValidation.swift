import Foundation;

private enum EmbeddingsValidation {

    private static let maximumEmbeddingInputBytes: Int = 8_192;
    private static let maximumEmbeddingTotalInputBytes: Int = 1_000_000;

    /// Rejection reasons enforced independently at the worker IPC boundary.
    enum EmbeddingsValidationError: Error, Equatable, CustomStringConvertible {
        case emptyModelId;
        case emptyInputs;
        case inputCountExceeded(actualInputCount: Int, maximumInputCount: Int);
        case inputTextTooLarge(actualInputBytes: Int, maximumInputBytes: Int);
        case totalInputBytesExceeded(actualTotalBytes: Int, maximumTotalBytes: Int);
        case invalidDimensions;

        var description: String {
            switch (self) {
            case .emptyModelId:
                return "model id must not be empty";
            case .emptyInputs:
                return "inputs must contain at least one text string";
            case let .inputCountExceeded(actualInputCount, maximumInputCount):
                return "embedding input count is \(actualInputCount), outside the 1..=\(maximumInputCount) range";
            case let .inputTextTooLarge(actualInputBytes, maximumInputBytes):
                return "one embedding input has \(actualInputBytes) bytes, exceeding the \(maximumInputBytes)-byte limit";
            case let .totalInputBytesExceeded(actualTotalBytes, maximumTotalBytes):
                return "aggregate embedding input has \(actualTotalBytes) bytes, exceeding the \(maximumTotalBytes)-byte limit";
            case .invalidDimensions:
                return "dimensions must be a positive vector width";
            }
        }
    }

    static func validateCommand(_ command: EmbeddingsCommand) throws {
        if command.model.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines).isEmpty {
            throw EmbeddingsValidationError.emptyModelId;
        }
        if command.inputs.isEmpty {
            throw EmbeddingsValidationError.emptyInputs;
        }
        if command.inputs.count > EmbeddingsCommand.maximumEmbeddingInputCount {
            throw EmbeddingsValidationError.inputCountExceeded(
                actualInputCount: command.inputs.count,
                maximumInputCount: EmbeddingsCommand.maximumEmbeddingInputCount);
        }
        var totalBytes: Int = 0;
        for inputText in command.inputs {
            let inputBytes: Int = inputText.utf8.count;
            totalBytes = totalBytes + inputBytes;
            if inputBytes > EmbeddingsValidation.maximumEmbeddingInputBytes {
                throw EmbeddingsValidationError.inputTextTooLarge(
                    actualInputBytes: inputBytes,
                    maximumInputBytes: EmbeddingsValidation.maximumEmbeddingInputBytes);
            }
        }
        if totalBytes > EmbeddingsValidation.maximumEmbeddingTotalInputBytes {
            throw EmbeddingsValidationError.totalInputBytesExceeded(
                actualTotalBytes: totalBytes,
                maximumTotalBytes: EmbeddingsValidation.maximumEmbeddingTotalInputBytes);
        }
        if command.dimensions == 0 {
            throw EmbeddingsValidationError.invalidDimensions;
        }
    }
}

extension EmbeddingsCommand {

    /// Independently validates embeddings input after it crosses the worker trust boundary.
    public func validate() throws {
        return try EmbeddingsValidation.validateCommand(self);
    }
}
