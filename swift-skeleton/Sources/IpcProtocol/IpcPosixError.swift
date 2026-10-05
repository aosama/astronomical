import Foundation;

/// The POSIX errno failure raised by the IPC socket transport. It carries the
/// raw errno so callers can classify failures the way Rust io::Error kinds do
/// (for example ENOENT versus ECONNREFUSED when probing the daemon socket),
/// and it renders the Rust io::Error Display shape so daemon-facing messages
/// survive the port byte for byte.
internal enum IpcPosixError: Error {
    case osError(errnoValue: Int32, underlyingErrorDescription: String);

    /// Captures the current errno before anything else can clobber it and
    /// renders it the way Rust renders OS errors: "No such file or directory
    /// (os error 2)".
    internal static func fromErrno() -> IpcPosixError {
        let capturedErrno: Int32 = errno;
        return IpcPosixError.osError(
            errnoValue: capturedErrno,
            underlyingErrorDescription: "\(String(cString: strerror(capturedErrno))) (os error \(capturedErrno))");
    }

    internal var errnoValue: Int32 {
        switch (self) {
        case .osError(let errnoValue, _): return errnoValue;
        }
    }

    internal var underlyingErrorDescription: String {
        switch (self) {
        case .osError(_, let underlyingErrorDescription): return underlyingErrorDescription;
        }
    }

    /// Converts into the public io::Error stand-in carried by ProtocolError.
    internal var ioError: IpcIoError {
        return IpcIoError(underlyingErrorDescription: self.underlyingErrorDescription);
    }
}
