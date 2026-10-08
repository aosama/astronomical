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

    /// Re-anchors one enumerated entry onto the caller-provided directory
    /// URL. Foundation enumeration resolves symlinked prefixes (/var becomes
    /// /private/var on macOS) while every path the store derives from its
    /// configuration keeps the caller's form; quota exclusion, ancestry
    /// protection, and index removal all compare these paths as strings, so
    /// the port joins names onto the given directory like Rust's read_dir.
    public static func storeFormEntryURL(
        directory: URL, enumeratedEntry: URL
    ) -> URL {
        return directory.appendingPathComponent(enumeratedEntry.lastPathComponent);
    }

    /// Persists a directory entry after publication: fsyncing file contents
    /// does not guarantee the directory entry survives a crash, so callers
    /// sync the parent directory after every rename.
    public static func synchronizeDirectory(directoryPath: URL) throws {
        // FileHandle cannot open directories on Darwin; the Rust original
        // relied on File::open + sync_all, so the port fsyncs through the
        // raw descriptor, preferring F_FULLFSYNC for the same power-loss
        // durability Rust's sync_all provides.
        let directoryFileDescriptor: Int32 = Darwin.open(directoryPath.path, O_RDONLY);
        if directoryFileDescriptor < 0 {
            throw PersistentPromptCacheDiskStoreError.openBlockFile(
                blockFilePath: directoryPath.path,
                problem: String(cString: strerror(errno)));
        }
        defer { Darwin.close(directoryFileDescriptor); }
        let fullSyncResult: Int32 = Darwin.fcntl(directoryFileDescriptor, F_FULLFSYNC);
        let syncResult: Int32 = fullSyncResult == 0
            ? 0
            : Darwin.fsync(directoryFileDescriptor);
        if syncResult != 0 {
            throw PersistentPromptCacheDiskStoreError.readBlockMetadata(
                blockFilePath: directoryPath.path,
                problem: String(cString: strerror(errno)));
        }
    }

    /// Removes one cache-owned directory tree, tolerating an already-absent
    /// path so cleanup stays idempotent under concurrent scanners.
    public static func removeCacheOwnedDirectoryOrConfirmAbsent(
        directoryPath: URL
    ) throws {
        do {
            try FileManager.default.removeItem(at: directoryPath);
        } catch let removeError as NSError
            where removeError.code == NSFileNoSuchFileError
                || removeError.code == NSFileReadNoSuchFileError {
            return;
        } catch {
            throw PersistentPromptCacheDiskStoreError.removePromptCacheFile(
                filePath: directoryPath.path, problem: String(describing: error));
        }
    }

    /// Removes one cache-owned file, tolerating an already-absent path: the
    /// store's cleanup paths must stay idempotent under concurrent readers.
    public static func removeCacheOwnedFileOrConfirmAbsent(
        filePath: URL
    ) throws {
        do {
            try FileManager.default.removeItem(at: filePath);
        } catch let removeError as NSError
            where removeError.code == NSFileNoSuchFileError
                || removeError.code == NSFileReadNoSuchFileError {
            return;
        } catch {
            throw PersistentPromptCacheDiskStoreError.removeCacheOwnedFile(
                filePath: filePath.path, problem: String(describing: error));
        }
    }
}
