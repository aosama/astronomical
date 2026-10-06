import Foundation;

/// Every dtype the pinned safetensors format declares, port of the external
/// `safetensors` crate's `Dtype` enum the Rust side imports. The Swift port
/// owns the format types, so the validated-source and raw-inventory readers
/// share this single exhaustive list.
public enum SafetensorsDtype: Equatable, Sendable, CaseIterable {
    case bool;
    case f4;
    case f6E2m3;
    case f6E3m2;
    case u8;
    case i8;
    case f8E5m2;
    case f8E4m3;
    case f8E8m0;
    case f8E4m3fnuz;
    case f8E5m2fnuz;
    case i16;
    case u16;
    case f16;
    case bf16;
    case i32;
    case u32;
    case f32;
    case f64;
    case i64;
    case c64;
    case u64;

    /// The canonical wire name the format's header JSON uses, also what
    /// Rust's `Dtype` Display renders.
    public var canonicalName: String {
        switch self {
        case .bool: return "BOOL";
        case .f4: return "F4";
        case .f6E2m3: return "F6_E2M3";
        case .f6E3m2: return "F6_E3M2";
        case .u8: return "U8";
        case .i8: return "I8";
        case .f8E5m2: return "F8_E5M2";
        case .f8E4m3: return "F8_E4M3";
        case .f8E8m0: return "F8_E8M0";
        case .f8E4m3fnuz: return "F8_E4M3FNUZ";
        case .f8E5m2fnuz: return "F8_E5M2FNUZ";
        case .i16: return "I16";
        case .u16: return "U16";
        case .f16: return "F16";
        case .bf16: return "BF16";
        case .i32: return "I32";
        case .u32: return "U32";
        case .f32: return "F32";
        case .f64: return "F64";
        case .i64: return "I64";
        case .c64: return "C64";
        case .u64: return "U64";
        }
    }

    /// Bits occupied by one element of this dtype, matching the pinned
    /// safetensors format's `Dtype::bitsize` table.
    public var bitsize: UInt64 {
        switch self {
        case .f4: return 4;
        case .f6E2m3, .f6E3m2: return 6;
        case .bool, .u8, .i8, .f8E5m2, .f8E4m3, .f8E8m0, .f8E4m3fnuz, .f8E5m2fnuz: return 8;
        case .i16, .u16, .f16, .bf16: return 16;
        case .i32, .u32, .f32: return 32;
        case .c64, .f64, .i64, .u64: return 64;
        }
    }

    public static func parsed(fromCanonicalName canonicalName: String) -> SafetensorsDtype? {
        for candidate: SafetensorsDtype in SafetensorsDtype.allCases where candidate.canonicalName == canonicalName {
            return candidate;
        }
        return nil;
    }
}

extension SafetensorsDtype: CustomStringConvertible {
    public var description: String {
        return self.canonicalName;
    }
}
