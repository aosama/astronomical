import Foundation;

/// serde `with = "crate::base64_bytes"` counterpart for binary payloads that
/// must stay compact on the JSON IPC wire. Rust uses the padded standard
/// alphabet, so decoding additionally rejects non-alphabet characters and text
/// whose length is not a multiple of four — the platform decoder would
/// otherwise tolerate missing padding and embedded whitespace.
internal enum Base64Bytes {
    internal static func encode(imageFileBytes: Array<UInt8>) -> String {
        return Data(imageFileBytes).base64EncodedString();
    }

    internal static func decode(encodedText: String) throws -> Array<UInt8> {
        let textContainsOnlyAllowedCharacters: Bool = encodedText.unicodeScalars.allSatisfy(
            { (scalar: Unicode.Scalar) -> Bool in Base64Bytes.isAllowedBase64Scalar(scalar) });
        if textContainsOnlyAllowedCharacters == false || encodedText.utf8.count % 4 != 0 {
            throw JsonWireProblem.malformedDocument(problem: "invalid base64 image bytes");
        }
        guard let decodedBytes = Data(base64Encoded: encodedText) else {
            throw JsonWireProblem.malformedDocument(problem: "invalid base64 image bytes");
        }
        return Array<UInt8>(decodedBytes);
    }

    private static func isAllowedBase64Scalar(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x41...0x5A, 0x61...0x7A, 0x30...0x39, 0x2B, 0x2F, 0x3D: return true;
        default: return false;
        }
    }
}
