import CoreGraphics;
import Foundation;
import ImageIO;

import Testing;

import JourneyCategories;

@testable import IpcProtocol;

/**
 * Wire-level journeys for `MessageCodec`, ported from the Rust
 * `crates/ipc-protocol` message codec and hermetic minimal-protocol tests:
 * round trips across all four message families, the serde-compatible
 * snake_case wire shape, the 32 MiB shared frame budget in both directions,
 * and the semantic contracts `decodeEvent` re-checks at the wire boundary.
 */
@Suite(.tags(.hermeticJourney))
final class MessageCodecTests {

    /// Local failure signal for fixtures that cannot be synthesized in-process.
    private enum FixtureError: Error {
        case pngEncodingFailed;
    }

    private static let FAKE_MODEL_ID: String = "astronomical/fake-mixture-of-experts";
    private static let FAKE_APPLICATION_NAME: String = "astronomical";
    private static let COMPLETION_IMAGE_WIDTH_PIXELS: UInt32 = 64;
    private static let COMPLETION_IMAGE_HEIGHT_PIXELS: UInt32 = 64;
    /// One ASCII byte above the 32 MiB frame budget (33_554_432 bytes) plus JSON overhead.
    private static let OVERSIZED_PROMPT_BYTE_COUNT: Int = 34_000_000;
    private static let OVERSIZED_PROMPT_TEXT: String =
        String(repeating: "a", count: MessageCodecTests.OVERSIZED_PROMPT_BYTE_COUNT);

    @Test
    func should_round_trip_a_chat_generation_command_through_the_codec() throws -> Void {
        let originalCommand: WorkerCommand = self.makeChatGenerationCommand(promptText: "Explain this Swift function.");
        let serializedCommand: Data = try MessageCodec.encodeCommand(originalCommand);

        let decodedCommand: WorkerCommand = try MessageCodec.decodeCommand(serializedCommand);
        let reserializedCommand: Data = try MessageCodec.encodeCommand(decodedCommand);

        #expect(decodedCommand == originalCommand, "the chat command should survive transport unchanged");
        #expect(reserializedCommand == serializedCommand, "re-encoding the decoded command should be byte-exact");
    }

    @Test
    func should_reject_a_whitespace_only_chat_model_id_before_worker_preprocessing() throws -> Void {
        let commandWithBlankModelId: ChatGenerationCommand = ChatGenerationCommand(
            requestId: RequestId(rawRequestId: 1),
            model: "  ",
            messages: Array<ChatMessage>([ChatMessage.user(content: "Romeo and Juliet", images: Array<ChatImageInput>())]),
            tools: Array<ChatToolDefinition>(),
            toolChoice: ChatToolChoice.none,
            settings: ChatGenerationSettings(
                maxOutputTokens: 1,
                temperatureThousandths: nil,
                topPThousandths: nil,
                seed: nil,
                thinkingBudget: nil),
            qwenThinkingChannelSeed: nil,
            structuredGeneration: nil);
        let serializedCommand: Data = try MessageCodec.encodeCommand(.generate(commandWithBlankModelId));

        let decodedCommand: WorkerCommand = try MessageCodec.decodeCommand(serializedCommand);
        guard case let .generate(chatGenerationCommand) = decodedCommand else {
            Issue.record("the chat command variant should survive transport");
            return;
        }

        do {
            try chatGenerationCommand.validate();
            Issue.record("a whitespace-only model id must fail worker-boundary validation");
        } catch let validationError as ChatGenerationValidationError {
            #expect(validationError == ChatGenerationValidationError.emptyModelId);
        }
    }

    @Test
    func should_round_trip_a_ready_event_with_valid_model_capabilities() throws -> Void {
        let originalEvent: WorkerEvent = self.makeReadyEvent(modelCapabilities: self.makeChatOnlyModelCapabilities());
        let serializedEvent: Data = try MessageCodec.encodeEvent(originalEvent);

        let decodedEvent: WorkerEvent = try MessageCodec.decodeEvent(serializedEvent);

        #expect(decodedEvent == originalEvent, "the ready event should survive transport unchanged");
    }

    @Test
    func should_round_trip_a_model_swapped_event() throws -> Void {
        let originalEvent: WorkerEvent = self.makeModelSwappedEvent(modelCapabilities: self.makeChatOnlyModelCapabilities());
        let serializedEvent: Data = try MessageCodec.encodeEvent(originalEvent);

        let decodedEvent: WorkerEvent = try MessageCodec.decodeEvent(serializedEvent);
        let reserializedEvent: Data = try MessageCodec.encodeEvent(decodedEvent);

        #expect(decodedEvent == originalEvent, "the model-swapped event should survive transport unchanged");
        #expect(reserializedEvent == serializedEvent, "re-encoding the decoded event should be byte-exact");
    }

    @Test
    func should_round_trip_an_image_generation_completed_event_with_a_valid_png_completion() throws -> Void {
        let encodedPngBytes: Array<UInt8> = try self.makeEncodedPngBytes(
            widthPixels: MessageCodecTests.COMPLETION_IMAGE_WIDTH_PIXELS,
            heightPixels: MessageCodecTests.COMPLETION_IMAGE_HEIGHT_PIXELS);
        let originalEvent: WorkerEvent = self.makeImageGenerationCompletedEvent(
            pngBytes: encodedPngBytes,
            resultMetadata: self.makeValidImageCompletionMetadata());
        let serializedEvent: Data = try MessageCodec.encodeEvent(originalEvent);

        let decodedEvent: WorkerEvent = try MessageCodec.decodeEvent(serializedEvent);
        let reserializedEvent: Data = try MessageCodec.encodeEvent(decodedEvent);

        #expect(decodedEvent == originalEvent, "the image completion should survive transport unchanged");
        #expect(reserializedEvent == serializedEvent, "re-encoding the decoded completion should be byte-exact");
    }

    @Test
    func should_round_trip_a_daemon_handshake_request_through_the_codec() throws -> Void {
        let originalRequest: DaemonRequest = DaemonRequest.handshake;
        let serializedRequest: Data = try MessageCodec.encodeDaemonRequest(originalRequest);

        let decodedRequest: DaemonRequest = try MessageCodec.decodeDaemonRequest(serializedRequest);

        #expect(decodedRequest == originalRequest, "the daemon handshake should survive transport unchanged");
    }

    @Test
    func should_round_trip_a_daemon_handshake_response_through_the_codec() throws -> Void {
        let originalResponse: DaemonResponse = DaemonResponse.handshakeAccepted(
            protocolVersion: 1,
            applicationName: MessageCodecTests.FAKE_APPLICATION_NAME);
        let serializedResponse: Data = try MessageCodec.encodeDaemonResponse(originalResponse);

        let decodedResponse: DaemonResponse = try MessageCodec.decodeDaemonResponse(serializedResponse);
        let reserializedResponse: Data = try MessageCodec.encodeDaemonResponse(decodedResponse);

        #expect(decodedResponse == originalResponse, "the daemon handshake response should survive transport unchanged");
        #expect(reserializedResponse == serializedResponse, "re-encoding the decoded response should be byte-exact");
    }

    @Test
    func should_encode_chat_commands_on_the_serde_compatible_snake_case_wire_shape() throws -> Void {
        let serializedCommand: Data = try MessageCodec.encodeCommand(self.makeChatGenerationCommand(promptText: "wire shape"));

        let serializedText: String = try #require(String(data: serializedCommand, encoding: String.Encoding.utf8));
        #expect(serializedText.contains("\"kind\":\"generate\""), "the variant tag must stay the snake_case `kind` discriminator");
        #expect(serializedText.contains("\"request_id\":71"), "field names must stay serde-compatible snake_case");
        #expect(serializedText.contains("\"tool_choice\":{\"kind\":\"none\"}"), "nested enums must keep the internally tagged shape");
    }

    @Test
    func should_reject_invalid_worker_model_capabilities_on_the_ready_wire_boundary() throws -> Void {
        let impossibleCapabilities: WorkerModelCapabilities = WorkerModelCapabilities(
            chat: nil,
            imageGeneration: nil,
            embeddings: nil);
        let serializedEvent: Data = try MessageCodec.encodeEvent(self.makeReadyEvent(modelCapabilities: impossibleCapabilities));

        do {
            _ = try MessageCodec.decodeEvent(serializedEvent);
            Issue.record("a ready event advertising no capability surface must be rejected at the wire boundary");
        } catch ProtocolError.invalidWorkerModelCapabilities(let capabilitiesValidationError) {
            #expect(capabilitiesValidationError == WorkerModelCapabilitiesValidationError.noCapabilities);
        }
    }

    @Test
    func should_reject_invalid_image_completion_metadata_at_the_wire_boundary() throws -> Void {
        let invalidMetadata: ImageGenerationResultMetadata = ImageGenerationResultMetadata(
            widthPixels: MessageCodecTests.COMPLETION_IMAGE_WIDTH_PIXELS,
            heightPixels: MessageCodecTests.COMPLETION_IMAGE_HEIGHT_PIXELS,
            steps: 28,
            guidanceThousandths: 100_001,
            seed: 42,
            elapsedMillis: 1_800);
        let encodedPngBytes: Array<UInt8> = try self.makeEncodedPngBytes(
            widthPixels: MessageCodecTests.COMPLETION_IMAGE_WIDTH_PIXELS,
            heightPixels: MessageCodecTests.COMPLETION_IMAGE_HEIGHT_PIXELS);
        let serializedEvent: Data = try MessageCodec.encodeEvent(self.makeImageGenerationCompletedEvent(
            pngBytes: encodedPngBytes,
            resultMetadata: invalidMetadata));

        do {
            _ = try MessageCodec.decodeEvent(serializedEvent);
            Issue.record("completion metadata outside the protocol bounds must be rejected at the wire boundary");
        } catch ProtocolError.invalidImageGenerationCompletion(let completionValidationError) {
            #expect(
                completionValidationError
                    == ImageGenerationCompletionValidationError.invalidMetadata(
                        metadataError: ImageGenerationValidationError.guidanceOutOfRange(
                            actualGuidanceThousandths: 100_001,
                            maximumGuidanceThousandths: 100_000)));
        }
    }

    @Test
    func should_reject_an_outgoing_command_beyond_the_shared_frame_budget() throws -> Void {
        let oversizedCommand: WorkerCommand = self.makeChatGenerationCommand(promptText: MessageCodecTests.OVERSIZED_PROMPT_TEXT);

        do {
            _ = try MessageCodec.encodeCommand(oversizedCommand);
            Issue.record("a command beyond the frame budget must be rejected before transport");
        } catch ProtocolError.outgoingMessageTooLarge(let actualMessageBytes, let maximumMessageBytes) {
            #expect(actualMessageBytes > maximumMessageBytes, "the rejection must report the real oversize");
            #expect(maximumMessageBytes == IpcFrameLimits.maximumIpcFrameBytes);
        }
    }

    @Test
    func should_reject_an_incoming_frame_beyond_the_shared_frame_budget() throws -> Void {
        let oversizedFrame: Data = Data(count: IpcFrameLimits.maximumIpcFrameBytes + 1);

        do {
            _ = try MessageCodec.decodeEvent(oversizedFrame);
            Issue.record("an inbound frame beyond the budget must be rejected before parsing");
        } catch ProtocolError.incomingMessageTooLarge(let actualMessageBytes, let maximumMessageBytes) {
            #expect(actualMessageBytes == IpcFrameLimits.maximumIpcFrameBytes + 1);
            #expect(maximumMessageBytes == IpcFrameLimits.maximumIpcFrameBytes);
        }
    }

    @Test
    func should_reject_malformed_json_bytes_as_a_deserialization_failure() throws -> Void {
        let malformedFrame: Data = Data("not json".utf8);

        do {
            _ = try MessageCodec.decodeCommand(malformedFrame);
            Issue.record("malformed JSON must surface as a deserialization failure");
        } catch ProtocolError.deserializeMessage {
            // The expected rejection: any other error propagates and fails the journey.
        }
    }

    private func makeChatGenerationCommand(promptText: String) -> WorkerCommand {
        return WorkerCommand.generate(ChatGenerationCommand(
            requestId: RequestId(rawRequestId: 71),
            model: MessageCodecTests.FAKE_MODEL_ID,
            messages: Array<ChatMessage>([ChatMessage.user(content: promptText, images: Array<ChatImageInput>())]),
            tools: Array<ChatToolDefinition>(),
            toolChoice: ChatToolChoice.none,
            settings: ChatGenerationSettings(
                maxOutputTokens: 128,
                temperatureThousandths: nil,
                topPThousandths: nil,
                seed: nil,
                thinkingBudget: nil),
            qwenThinkingChannelSeed: nil,
            structuredGeneration: nil));
    }

    private func makeChatOnlyModelCapabilities() -> WorkerModelCapabilities {
        return WorkerModelCapabilities(
            chat: ChatModelCapabilities(
                supportsReasoning: true,
                supportsToolCalls: false,
                hasVision: false,
                maxInputTokens: 8_192,
                maxOutputTokens: 2_048,
                contextWindow: 32_768),
            imageGeneration: nil,
            embeddings: nil);
    }

    private func makeReadyEvent(modelCapabilities: WorkerModelCapabilities) -> WorkerEvent {
        return WorkerEvent.ready(
            modelId: MessageCodecTests.FAKE_MODEL_ID,
            capabilities: modelCapabilities);
    }

    private func makeModelSwappedEvent(modelCapabilities: WorkerModelCapabilities) -> WorkerEvent {
        return WorkerEvent.modelSwapped(
            modelId: MessageCodecTests.FAKE_MODEL_ID,
            capabilities: modelCapabilities,
            expertMemoryMode: nil,
            minimumMlxMemoryCeilingBytes: 1);
    }

    private func makeValidImageCompletionMetadata() -> ImageGenerationResultMetadata {
        return ImageGenerationResultMetadata(
            widthPixels: MessageCodecTests.COMPLETION_IMAGE_WIDTH_PIXELS,
            heightPixels: MessageCodecTests.COMPLETION_IMAGE_HEIGHT_PIXELS,
            steps: 28,
            guidanceThousandths: 3_500,
            seed: 42,
            elapsedMillis: 1_800);
    }

    private func makeImageGenerationCompletedEvent(
        pngBytes: Array<UInt8>,
        resultMetadata: ImageGenerationResultMetadata
    ) -> WorkerEvent {
        return WorkerEvent.imageGenerationCompleted(
            requestId: RequestId(rawRequestId: 501),
            generatedImage: GeneratedImage(mimeType: "image/png", encodedBytes: pngBytes),
            resultMetadata: resultMetadata);
    }

    /// Encodes a solid-color truecolor (8-bit RGB, no alpha) PNG in memory, the
    /// only pixel format the completion contract accepts.
    private func makeEncodedPngBytes(widthPixels: UInt32, heightPixels: UInt32) throws -> Array<UInt8> {
        let pixelByteCount: Int = Int(widthPixels) * Int(heightPixels) * 4;
        // Quartz supports no 24-bpp no-alpha contexts: RGB bitmaps must be
        // 32 bpp with a skipped alpha byte, which ImageIO still writes as a
        // truecolor (color type 2) 8-bit PNG.
        var solidPixelBytes: Array<UInt8> = Array<UInt8>(repeating: 0x3C, count: pixelByteCount);
        for opaquePixelStartIndex in stride(from: 0, to: pixelByteCount, by: 4) {
            solidPixelBytes[opaquePixelStartIndex + 3] = 0xFF;
        }
        guard let sRgbColorSpace: CGColorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw FixtureError.pngEncodingFailed;
        }
        guard let bitmapContext: CGContext = CGContext(
            data: &solidPixelBytes,
            width: Int(widthPixels),
            height: Int(heightPixels),
            bitsPerComponent: 8,
            bytesPerRow: Int(widthPixels) * 4,
            space: sRgbColorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            throw FixtureError.pngEncodingFailed;
        }
        guard let encodedImage: CGImage = bitmapContext.makeImage() else {
            throw FixtureError.pngEncodingFailed;
        }
        let pngData: NSMutableData = NSMutableData();
        guard let pngDestination: CGImageDestination = CGImageDestinationCreateWithData(
            pngData,
            "public.png" as CFString,
            1,
            nil) else {
            throw FixtureError.pngEncodingFailed;
        }
        CGImageDestinationAddImage(pngDestination, encodedImage, nil);
        if CGImageDestinationFinalize(pngDestination) == false {
            throw FixtureError.pngEncodingFailed;
        }
        return Array<UInt8>(pngData as Data);
    }
}
