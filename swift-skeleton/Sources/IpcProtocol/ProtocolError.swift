import Foundation;

/// The Swift stand-in for Rust's `std::io::Error` on the IPC frame path; it
/// carries the rendered OS error text so Display-compatible messages survive
/// the port without binding to one platform error representation.
public struct IpcIoError: Error, CustomStringConvertible {
    public let underlyingErrorDescription: String;

    public init(underlyingErrorDescription: String) {
        self.underlyingErrorDescription = underlyingErrorDescription;
    }

    public var description: String {
        return underlyingErrorDescription;
    }
}

/// Errors raised while serializing, transmitting, or deserializing IPC messages.
public enum ProtocolError: Error, CustomStringConvertible {
    /// A decoded chat command violated the worker trust-boundary contract.
    case invalidChatGenerationCommand(ChatGenerationValidationError);
    /// A decoded image command violated the worker trust-boundary contract.
    case invalidImageGenerationCommand(ImageGenerationValidationError);
    /// A worker advertised an impossible or empty capability contract.
    case invalidWorkerModelCapabilities(WorkerModelCapabilitiesValidationError);
    /// Image completion bytes or metadata did not describe a protocol-valid PNG outcome.
    case invalidImageGenerationCompletion(ImageGenerationCompletionValidationError);
    /// The serialized message cannot fit inside one bounded IPC frame.
    case outgoingMessageTooLarge(actualMessageBytes: Int, maximumMessageBytes: Int);
    /// A message supplied by a message-oriented transport exceeded the frame cap.
    case incomingMessageTooLarge(actualMessageBytes: Int, maximumMessageBytes: Int);
    /// Reading a length-delimited frame failed.
    case readFrame(IpcIoError);
    /// Writing a length-delimited frame failed.
    case writeFrame(IpcIoError);
    /// Serializing a typed message into JSON failed.
    case serializeMessage(JsonWireProblem);
    /// Deserializing a JSON frame into a typed message failed.
    case deserializeMessage(JsonWireProblem);

    public var description: String {
        switch (self) {
        case .invalidChatGenerationCommand:
            return "received an invalid chat-generation command";
        case .invalidImageGenerationCommand:
            return "received an invalid image-generation command";
        case .invalidWorkerModelCapabilities:
            return "received invalid worker model capabilities";
        case .invalidImageGenerationCompletion:
            return "received an invalid image-generation completion";
        case let .outgoingMessageTooLarge(actualMessageBytes, maximumMessageBytes):
            return "IPC message is \(actualMessageBytes) bytes, exceeding the \(maximumMessageBytes)-byte limit";
        case let .incomingMessageTooLarge(actualMessageBytes, maximumMessageBytes):
            return "received IPC message is \(actualMessageBytes) bytes, exceeding the \(maximumMessageBytes)-byte limit";
        case .readFrame:
            return "failed to read an IPC frame";
        case .writeFrame:
            return "failed to write an IPC frame";
        case let .serializeMessage(wireProblem):
            return "failed to serialize an IPC message: \(wireProblem)";
        case let .deserializeMessage(wireProblem):
            return "failed to deserialize an IPC message: \(wireProblem)";
        }
    }
}
