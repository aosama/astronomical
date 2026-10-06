import Foundation;

/// Safetensors dtype and payload-size arithmetic shared by the bounded
/// reader, the validated source, and the raw inventory. Port of
/// crates/model-serving/src/artifact_validation/safetensors_dtype.rs.
public enum SafetensorsDtypeHelpers {

    /// Payload byte count for a tensor, or nil when the element count is not
    /// a whole number of bytes for this dtype.
    public static func checkedSafetensorsPayloadBytes(
        elementCount: UInt64, bitsPerElement: UInt64) throws -> UInt64? {
        // Eight elements always occupy exactly `bits_per_element` bytes. Grouping
        // first avoids overflowing an intermediate bit count when bytes still fit.
        let completeEightElementGroups: UInt64 = elementCount / 8;
        let trailingElementCount: UInt64 = elementCount % 8;
        let (completeGroupBytes, groupBytesOverflowed): (UInt64, Bool) =
            completeEightElementGroups.multipliedReportingOverflow(by: bitsPerElement);
        if groupBytesOverflowed {
            throw ArtifactValidationError.tensorPayloadSizeOverflow;
        }
        let (trailingBits, trailingBitsOverflowed): (UInt64, Bool) =
            trailingElementCount.multipliedReportingOverflow(by: bitsPerElement);
        if trailingBitsOverflowed {
            throw ArtifactValidationError.tensorPayloadSizeOverflow;
        }
        if trailingBits % 8 != 0 {
            return nil;
        }
        let (payloadBytes, payloadBytesOverflowed): (UInt64, Bool) =
            completeGroupBytes.addingReportingOverflow(trailingBits / 8);
        if payloadBytesOverflowed {
            throw ArtifactValidationError.tensorPayloadSizeOverflow;
        }
        return payloadBytes;
    }

    /// Bit width per element for every dtype name the artifact validators
    /// accept, including the sub-byte formats.
    public static func dtypeBitsPerElement(
        dtypeString: String, fileName: String, tensorName: String) throws -> UInt64 {
        switch dtypeString {
        case "F4":
            return 4;
        case "F6_E2M3", "F6_E3M2":
            return 6;
        case "BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3", "F8_E8M0", "F8_E4M3FNUZ", "F8_E5M2FNUZ":
            return 8;
        case "I16", "U16", "F16", "BF16":
            return 16;
        case "I32", "U32", "F32":
            return 32;
        case "C64", "F64", "I64", "U64":
            return 64;
        default:
            throw SafetensorsDtypeHelpers.unknownSafetensorsDtypeError(
                dtypeString: dtypeString, fileName: fileName, tensorName: tensorName);
        }
    }

    /// Parses the dtype names accepted during artifact validation.
    public static func parseSafetensorsDtype(
        dtypeString: String, fileName: String, tensorName: String) throws -> SafetensorsDtype {
        guard let parsedDtype: SafetensorsDtype = SafetensorsDtype.parsed(fromCanonicalName: dtypeString) else {
            throw SafetensorsDtypeHelpers.unknownSafetensorsDtypeError(
                dtypeString: dtypeString, fileName: fileName, tensorName: tensorName);
        }
        return parsedDtype;
    }

    /// Parses every dtype understood by the pinned safetensors format, for
    /// raw inventory readers that must describe any well-formed file.
    public static func parseRawSafetensorsDtype(
        dtypeString: String, fileName: String, tensorName: String) throws -> SafetensorsDtype {
        guard let parsedDtype: SafetensorsDtype = SafetensorsDtype.parsed(fromCanonicalName: dtypeString) else {
            throw SafetensorsDtypeHelpers.unknownSafetensorsDtypeError(
                dtypeString: dtypeString, fileName: fileName, tensorName: tensorName);
        }
        return parsedDtype;
    }

    private static func unknownSafetensorsDtypeError(
        dtypeString: String, fileName: String, tensorName: String) -> ArtifactValidationError {
        return ArtifactValidationError.unknownSafetensorsDtype(
            fileName: fileName, tensorName: tensorName, dtypeString: dtypeString);
    }
}
