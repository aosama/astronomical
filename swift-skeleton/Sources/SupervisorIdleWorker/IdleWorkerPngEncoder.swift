import Foundation

import CoreGraphics
import ImageIO

/**
 * Encodes the solid-color truecolor PNG completions the scripted image
 * behaviors emit. The supervisor structurally validates every completion
 * (chunk CRCs, bit depth, color type, and the advertised geometry), so the
 * fixture must produce a genuine PNG at the requested dimensions.
 */
enum IdleWorkerPngEncoder {

    enum PngEncodingFailure: Error, CustomStringConvertible {
        case encoderUnavailable

        var description: String {
            switch (self) {
            case .encoderUnavailable:
                return "the scripted image fixture could not encode a truecolor PNG"
            }
        }
    }

    /**
     * - Parameters:
     *   - widthPixels: The advertised completion width.
     *   - heightPixels: The advertised completion height.
     * - Returns: A solid-color 8-bit truecolor (no alpha) PNG, the only pixel
     *   format the completion contract accepts.
     * - Throws: `PngEncodingFailure.encoderUnavailable` when Quartz cannot
     *   provide the color space, bitmap context, or image destination.
     */
    static func encodeTruecolorPng(widthPixels: UInt32, heightPixels: UInt32) throws -> Array<UInt8> {
        let pixelByteCount: Int = Int(widthPixels) * Int(heightPixels) * 4
        // Quartz supports no 24-bpp no-alpha contexts: RGB bitmaps must be
        // 32 bpp with a skipped alpha byte, which ImageIO still writes as a
        // truecolor (color type 2) 8-bit PNG.
        var solidPixelBytes: Array<UInt8> = Array<UInt8>(repeating: 0x3C, count: pixelByteCount)
        for opaquePixelStartIndex: Int in stride(from: 0, to: pixelByteCount, by: 4) {
            solidPixelBytes[opaquePixelStartIndex + 3] = 0xFF
        }
        guard let sRgbColorSpace: CGColorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw PngEncodingFailure.encoderUnavailable
        }
        guard let bitmapContext: CGContext = CGContext(
            data: &solidPixelBytes,
            width: Int(widthPixels),
            height: Int(heightPixels),
            bitsPerComponent: 8,
            bytesPerRow: Int(widthPixels) * 4,
            space: sRgbColorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            throw PngEncodingFailure.encoderUnavailable
        }
        guard let encodedImage: CGImage = bitmapContext.makeImage() else {
            throw PngEncodingFailure.encoderUnavailable
        }
        let pngData: NSMutableData = NSMutableData()
        guard let pngDestination: CGImageDestination = CGImageDestinationCreateWithData(
            pngData,
            "public.png" as CFString,
            1,
            nil) else {
            throw PngEncodingFailure.encoderUnavailable
        }
        CGImageDestinationAddImage(pngDestination, encodedImage, nil)
        if (CGImageDestinationFinalize(pngDestination) == false) {
            throw PngEncodingFailure.encoderUnavailable
        }
        return Array<UInt8>(pngData as Data)
    }
}
