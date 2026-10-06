import Foundation;

/// Wire-shaped codec seam shared by the four bounded IPC message enums; it
/// stands in for the Rust codec's `Serialize`/`DeserializeOwned` generic bounds.
private protocol JsonWireMessage {
    func wireValue() -> JsonWireValue;
    static func fromWireValue(_ wireValue: JsonWireValue) throws -> Self;
}

extension WorkerCommand: JsonWireMessage {
}

extension WorkerEvent: JsonWireMessage {
}

extension DaemonRequest: JsonWireMessage {
}

extension DaemonResponse: JsonWireMessage {
}

/// Serializes and deserializes bounded IPC messages, enforcing the shared frame
/// budget in both directions exactly like the Rust codec.
public enum MessageCodec {
    /// Serializes one bounded command sent to the inference worker.
    public static func encodeCommand(_ workerCommand: WorkerCommand) throws -> Data {
        return try MessageCodec.encodeMessage(workerCommand);
    }

    /// Deserializes one bounded command received by the inference worker.
    public static func decodeCommand(_ serializedCommand: Data) throws -> WorkerCommand {
        return try MessageCodec.decodeMessage(serializedCommand);
    }

    /// Serializes one bounded event emitted by the inference worker.
    public static func encodeEvent(_ workerEvent: WorkerEvent) throws -> Data {
        return try MessageCodec.encodeMessage(workerEvent);
    }

    /// Deserializes one bounded event received from the inference worker,
    /// re-checking the semantic capability and completion contracts that the
    /// worker must never breach even over a trusted socket.
    public static func decodeEvent(_ serializedEvent: Data) throws -> WorkerEvent {
        let workerEvent: WorkerEvent = try MessageCodec.decodeMessage(serializedEvent);
        switch (workerEvent) {
        case .ready(_, let workerCapabilities):
            do { try workerCapabilities.validate(); }
            catch let validationError as WorkerModelCapabilitiesValidationError {
                throw ProtocolError.invalidWorkerModelCapabilities(validationError);
            }
        case .modelSwapped(_, let workerCapabilities, _, _):
            do { try workerCapabilities.validate(); }
            catch let validationError as WorkerModelCapabilitiesValidationError {
                throw ProtocolError.invalidWorkerModelCapabilities(validationError);
            }
        case .imageGenerationCompleted(_, let generatedImage, let resultMetadata):
            do { try generatedImage.validateCompletion(resultMetadata: resultMetadata); }
            catch let validationError as ImageGenerationCompletionValidationError {
                throw ProtocolError.invalidImageGenerationCompletion(validationError);
            }
        default: break;
        }
        return workerEvent;
    }

    /// Serializes one bounded request sent by a local CLI process to the daemon.
    public static func encodeDaemonRequest(_ daemonRequest: DaemonRequest) throws -> Data {
        return try MessageCodec.encodeMessage(daemonRequest);
    }

    /// Deserializes one bounded request received by the daemon.
    public static func decodeDaemonRequest(_ serializedRequest: Data) throws -> DaemonRequest {
        return try MessageCodec.decodeMessage(serializedRequest);
    }

    /// Serializes one bounded response sent by the daemon to a local CLI process.
    public static func encodeDaemonResponse(_ daemonResponse: DaemonResponse) throws -> Data {
        return try MessageCodec.encodeMessage(daemonResponse);
    }

    /// Deserializes one bounded response received by a local CLI process.
    public static func decodeDaemonResponse(_ serializedResponse: Data) throws -> DaemonResponse {
        return try MessageCodec.decodeMessage(serializedResponse);
    }

    private static func encodeMessage<Message: JsonWireMessage>(_ message: Message) throws -> Data {
        var wireWriter = JsonWireWriter();
        do { try wireWriter.appendValue(message.wireValue()); }
        catch {
            throw ProtocolError.serializeMessage(
                (error as? JsonWireProblem) ?? JsonWireProblem.malformedDocument(problem: String(describing: error)));
        }
        let serializedMessage: Data = wireWriter.serializedUtf8Bytes;
        if serializedMessage.count > IpcFrameLimits.maximumIpcFrameBytes {
            throw ProtocolError.outgoingMessageTooLarge(
                actualMessageBytes: serializedMessage.count,
                maximumMessageBytes: IpcFrameLimits.maximumIpcFrameBytes);
        }
        return serializedMessage;
    }

    private static func decodeMessage<Message: JsonWireMessage>(_ serializedMessage: Data) throws -> Message {
        if serializedMessage.count > IpcFrameLimits.maximumIpcFrameBytes {
            throw ProtocolError.incomingMessageTooLarge(
                actualMessageBytes: serializedMessage.count,
                maximumMessageBytes: IpcFrameLimits.maximumIpcFrameBytes);
        }
        do {
            let documentWireValue: JsonWireValue = try JsonWireParser.parseDocument(documentBytes: serializedMessage);
            return try Message.fromWireValue(documentWireValue);
        } catch {
            throw ProtocolError.deserializeMessage(
                (error as? JsonWireProblem) ?? JsonWireProblem.malformedDocument(problem: String(describing: error)));
        }
    }
}
