import Foundation;

/// Cache-owned file vocabulary and byte-exact helpers shared by the disk
/// store's writers and scanners, port of the Rust `disk_store_file` helpers.
public enum PersistentPromptCacheStoreFile {

    public static let BLOCK_MANIFEST_FILE_NAME: String = "manifest.json";
    public static let SEQUENCE_STATE_FILE_NAME: String = "sequence.safetensors";
    public static let BOUNDARY_STATE_FILE_NAME: String = "boundary.safetensors";

    /// The lowercase hexadecimal encoding the writer emits for every block
    /// hash and fingerprint on disk.
    public static func hexEncode(_ bytes: Data) -> String {
        return bytes.map({ (byte: UInt8) -> String in
            return String(format: "%02x", byte);
        }).joined();
    }

    /// Parses a canonical 64-character lowercase hexadecimal block hash back
    /// into binary identity; the writer emits exactly this representation,
    /// and parsing still validates exact width and every byte pair.
    public static func parseBlockHashHex(_ blockHashHex: String) -> Data? {
        let hexCharacters: Array<Character> = Array(blockHashHex);
        if hexCharacters.count != 64 {
            return nil;
        }
        var blockHash: Data = Data(capacity: 32);
        var byteValue: UInt8 = 0;
        for (characterIndex, hexCharacter): (Int, Character) in zip(hexCharacters.indices, hexCharacters) {
            guard let nibbleValue: UInt8 = UInt8(String(hexCharacter), radix: 16) else {
                return nil;
            }
            if characterIndex % 2 == 0 {
                byteValue = nibbleValue << 4;
            } else {
                byteValue |= nibbleValue;
                blockHash.append(byteValue);
            }
        }
        return blockHash;
    }

    /// Removes one cache-owned file, tolerating an already-absent path: the
    /// store's cleanup paths must stay idempotent under concurrent readers.
    public static func removeCacheOwnedFileOrConfirmAbsent(
        filePath: URL
    ) throws {
        do {
            try FileManager.default.removeItem(at: filePath);
        } catch let removeError as NSError where removeError.code == NSFileNoSuchFileError {
            return;
        } catch {
            throw PersistentPromptCacheDiskStoreError.removeCacheOwnedFile(
                filePath: filePath.path, problem: String(describing: error));
        }
    }
}
