import Foundation;

#if canImport(Glibc)
import Glibc;
#else
import Darwin;
#endif

/// Exclusive process-lifetime ownership of one instance's writable state.
///
/// Mirrors apps/supervisor/src/instance_lock.rs: the lock is an advisory
/// flock(2) on a 0600 lock file inside the state directory, so the kernel
/// releases it when the process dies and a second daemon for the same
/// instance fails immediately instead of corrupting shared state.
public final class AstronomicalInstanceLock {

    private let lockedFileDescriptor: Int32;

    private init(lockedFileDescriptor: Int32) {
        self.lockedFileDescriptor = lockedFileDescriptor;
    }

    public static func acquire(lockFilePath: String) throws -> AstronomicalInstanceLock {
        let stateDirectory: String = (lockFilePath as NSString).deletingLastPathComponent;
        do {
            try FileManager.default.createDirectory(
                atPath: stateDirectory,
                withIntermediateDirectories: true);
        } catch let createError as NSError {
            throw AstronomicalInstanceLockError.createStateDirectory(
                stateDirectory: stateDirectory,
                underlyingDescription: createError.localizedDescription);
        }
        let fileDescriptor: Int32 = open(lockFilePath, O_RDWR | O_CREAT, 0o600);
        guard fileDescriptor >= 0 else {
            throw AstronomicalInstanceLockError.openLockFile(
                lockFilePath: lockFilePath,
                underlyingDescription: String(cString: strerror(errno)));
        }
        let lockResult: Int32 = flock(fileDescriptor, LOCK_EX | LOCK_NB);
        if lockResult == 0 {
            return AstronomicalInstanceLock(lockedFileDescriptor: fileDescriptor);
        }
        let lockErrno: Int32 = errno;
        close(fileDescriptor);
        if lockErrno == EWOULDBLOCK {
            throw AstronomicalInstanceLockError.alreadyRunning;
        }
        throw AstronomicalInstanceLockError.acquireLock(
            underlyingDescription: String(cString: strerror(lockErrno)));
    }

    deinit {
        flock(self.lockedFileDescriptor, LOCK_UN);
        close(self.lockedFileDescriptor);
    }
}

public enum AstronomicalInstanceLockError: Error, Equatable {
    /// Astronomical is already running for the selected instance.
    case alreadyRunning
    case openLockFile(lockFilePath: String, underlyingDescription: String)
    case createStateDirectory(stateDirectory: String, underlyingDescription: String)
    case acquireLock(underlyingDescription: String)
}
