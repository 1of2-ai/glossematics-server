import Darwin
import Foundation

/// Per-request scratch directories for uploaded media, and the sweep that removes the ones a crash
/// leaves behind.
///
/// Audio and video arrive as bytes but are decoded by AVFoundation from files, so each preparation
/// writes the upload to `$TMPDIR/gloss-media-<pid>-<uuid>/input.<ext>` and removes the directory
/// when preparation finishes. A crash (or `kill -9`) between write and removal would otherwise leave
/// user media on disk indefinitely. Naming the directory with the owning pid lets a later process
/// tell "its owner is gone" from "another live server is still working on it".
enum TemporaryMediaFiles {
    static let directoryPrefix = "gloss-media-"

    /// `gloss-media-<pid>-<uuid>`.
    static func directoryName(pid: Int32 = getpid(), id: UUID = UUID()) -> String {
        "\(directoryPrefix)\(pid)-\(id.uuidString)"
    }

    /// Who a `gloss-media-*` directory name says it belongs to.
    enum Owner: Equatable {
        /// `gloss-media-<pid>-<uuid>`: written by the process with this pid.
        case process(Int32)
        /// `gloss-media-<uuid>`: the pre-pid format of older versions. The owner is unknown.
        case legacy
        /// Anything else that merely starts with the prefix.
        case unrecognized
    }

    /// Classify a directory name; `nil` when it does not start with `gloss-media-` at all.
    static func owner(ofDirectoryNamed name: String) -> Owner? {
        guard name.hasPrefix(directoryPrefix) else { return nil }
        let rest = String(name.dropFirst(directoryPrefix.count))
        if UUID(uuidString: rest) != nil { return .legacy }
        if let dash = rest.firstIndex(of: "-") {
            let pidText = rest[rest.startIndex..<dash]
            let uuidText = String(rest[rest.index(after: dash)...])
            // A pid is a positive decimal that fits pid_t; `Int32` parsing rejects signs and
            // overflow, and the digit check rejects "+5" and other spellings `Int32` would accept.
            if pidText.allSatisfy(\.isASCII), pidText.allSatisfy(\.isNumber),
               let pid = Int32(pidText), pid > 0,
               UUID(uuidString: uuidText) != nil {
                return .process(pid)
            }
        }
        return .unrecognized
    }

    /// Write `data` to a fresh private directory as `input.<extension>`, run `operation` on it, and
    /// remove the directory afterwards — on success, failure, and cancellation alike.
    static func withFile<T>(
        _ data: Data,
        extension fileExtension: String,
        in parent: URL = FileManager.default.temporaryDirectory,
        pid: Int32 = getpid(),
        _ operation: (URL) throws -> T
    ) throws -> T {
        let directory = parent.appendingPathComponent(directoryName(pid: pid), isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("input.\(fileExtension)")
        try data.write(to: file, options: .atomic)
        return try operation(file)
    }

    /// Whether `pid` names a live process, by signal 0: success means it exists; `EPERM` means it
    /// exists but belongs to someone else (still alive); only `ESRCH` means it is gone. A
    /// non-positive pid is never a valid owner, and `kill(0, _)` would signal a whole process
    /// group, so those are reported alive (never swept) without calling `kill`.
    static func processIsAlive(_ pid: Int32) -> Bool {
        guard pid > 0 else { return true }
        if kill(pid, 0) == 0 { return true }
        return errno != ESRCH
    }
}

/// When a stale `gloss-media-*` directory may be removed. See ``OmniSmall/sweepStaleTemporaryMedia(in:now:currentPID:currentUID:policy:isProcessAlive:)``.
@_spi(Server)
public struct TemporaryMediaSweepPolicy: Sendable, Equatable {
    /// How long a directory whose owning process is dead must sit before it is removed. A short
    /// grace keeps the sweep from racing a directory that was just created by a process that is
    /// exiting, and from acting on a transiently misreported liveness.
    public var deadOwnerGrace: TimeInterval
    /// Age beyond which a legacy `gloss-media-<uuid>` directory (owner unknown) is removed. Media
    /// preparation takes seconds, so an hour is far past any live use.
    public var legacyMaximumAge: TimeInterval
    /// Age beyond which any matching directory is removed even if its recorded pid looks alive.
    /// Pids are reused; without this a stale directory whose pid was recycled by an unrelated
    /// long-running process would live forever.
    public var absoluteMaximumAge: TimeInterval

    public init(
        deadOwnerGrace: TimeInterval = 60,
        legacyMaximumAge: TimeInterval = 3_600,
        absoluteMaximumAge: TimeInterval = 86_400
    ) {
        self.deadOwnerGrace = deadOwnerGrace
        self.legacyMaximumAge = legacyMaximumAge
        self.absoluteMaximumAge = absoluteMaximumAge
    }
}

/// What a sweep did, for the startup log.
@_spi(Server)
public struct TemporaryMediaSweepReport: Sendable, Equatable {
    /// Direct children whose names matched `gloss-media-*` (any type).
    public var examined = 0
    /// Directories removed.
    public var removedDirectories = 0
    /// Total size of the regular files inside the removed directories.
    public var removedBytes: Int64 = 0
    /// Matching entries deliberately left alone: not a real directory, not owned by this user, this
    /// process's own, owned by a live process, or not yet old enough.
    public var kept = 0
    /// Directories that qualified for removal but could not be removed.
    public var failed = 0

    public init() {}
}

extension OmniSmall {
    /// Remove `gloss-media-*` scratch directories that a crashed or killed process left behind.
    /// Call once at startup, before serving.
    ///
    /// Only DIRECT children of `directory` are considered, and only those that
    ///
    /// * match `gloss-media-*`,
    /// * are real directories — checked with `lstat`, so a symlink named `gloss-media-x` (to
    ///   anywhere) is never followed or removed, and
    /// * are owned by `currentUID`.
    ///
    /// A qualifying directory is removed when ANY of these holds:
    ///
    /// 1. its name records a pid (`gloss-media-<pid>-<uuid>`) that is not alive and it is older than
    ///    `policy.deadOwnerGrace`;
    /// 2. it uses the legacy `gloss-media-<uuid>` format and is older than `policy.legacyMaximumAge`;
    /// 3. it is older than `policy.absoluteMaximumAge`, whatever its name says — the guard against
    ///    pid reuse making a dead owner look alive.
    ///
    /// A directory owned by `currentPID` is never touched, however old. Age is the directory's
    /// modification time relative to `now`.
    ///
    /// Every input the decision depends on is injectable — the directory, the clock (`now`), the
    /// pids, the uid, and the liveness predicate (default: `kill(pid, 0)`, where only `ESRCH` means
    /// dead) — so it is testable with a temporary directory.
    /// This never throws: an unreadable directory or an entry that cannot be removed is counted, not
    /// fatal, because startup must not fail over scratch-file housekeeping.
    @_spi(Server)
    public static func sweepStaleTemporaryMedia(
        in directory: URL = FileManager.default.temporaryDirectory,
        now: Date = Date(),
        currentPID: Int32 = getpid(),
        currentUID: UInt32 = getuid(),
        policy: TemporaryMediaSweepPolicy = TemporaryMediaSweepPolicy(),
        isProcessAlive: ((Int32) -> Bool)? = nil
    ) -> TemporaryMediaSweepReport {
        let isProcessAlive = isProcessAlive ?? TemporaryMediaFiles.processIsAlive
        var report = TemporaryMediaSweepReport()
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        for name in names.sorted() {
            guard let owner = TemporaryMediaFiles.owner(ofDirectoryNamed: name) else { continue }
            report.examined += 1
            let path = directory.appendingPathComponent(name).path

            var status = stat()
            guard lstat(path, &status) == 0,
                  (status.st_mode & S_IFMT) == S_IFDIR,
                  status.st_uid == currentUID else {
                report.kept += 1
                continue
            }
            if case .process(let pid) = owner, pid == currentPID {
                report.kept += 1
                continue
            }

            let modified = Double(status.st_mtimespec.tv_sec)
                + Double(status.st_mtimespec.tv_nsec) / 1_000_000_000
            let age = now.timeIntervalSince1970 - modified

            let removable: Bool
            if age > policy.absoluteMaximumAge {
                removable = true
            } else {
                switch owner {
                case let .process(pid):
                    removable = age > policy.deadOwnerGrace && !isProcessAlive(pid)
                case .legacy:
                    removable = age > policy.legacyMaximumAge
                case .unrecognized:
                    removable = false
                }
            }
            guard removable else {
                report.kept += 1
                continue
            }

            let bytes = regularFileBytes(under: path)
            do {
                try FileManager.default.removeItem(atPath: path)
                report.removedDirectories += 1
                report.removedBytes += bytes
            } catch {
                report.failed += 1
            }
        }
        return report
    }

    /// Total size of the regular files under `path`, without following symlinks.
    private static func regularFileBytes(under path: String) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(atPath: path) else { return 0 }
        var total: Int64 = 0
        for case let relative as String in enumerator {
            var status = stat()
            if lstat(path + "/" + relative, &status) == 0, (status.st_mode & S_IFMT) == S_IFREG {
                total += Int64(status.st_size)
            }
        }
        return total
    }
}
