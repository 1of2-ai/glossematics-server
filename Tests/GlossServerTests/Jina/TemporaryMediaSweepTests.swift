import Foundation
import Testing
@_spi(Server) @testable import gloss_server

/// Stale `gloss-media-*` scratch directories: naming, and the startup sweep that removes what a
/// crash left behind. Everything the decision depends on is injected (directory, clock, pids, uid,
/// liveness), so these run against a private temporary directory.

private let now = Date(timeIntervalSince1970: 1_800_000_000)
private let currentPID: Int32 = 4_242
private let deadPID: Int32 = 5_001
private let livePID: Int32 = 6_002

private struct Sandbox {
    let root: URL

    init() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("gloss-sweep-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    func remove() { try? FileManager.default.removeItem(at: root) }

    /// A `gloss-media-*` directory holding one file of `bytes` bytes, last modified `age` seconds
    /// before `now`.
    @discardableResult
    func directory(_ name: String, age: TimeInterval, bytes: Int = 100) throws -> URL {
        let url = root.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try Data(count: bytes).write(to: url.appendingPathComponent("input.wav"))
        try FileManager.default.setAttributes(
            [.modificationDate: now.addingTimeInterval(-age)], ofItemAtPath: url.path)
        return url
    }

    func exists(_ name: String) -> Bool {
        FileManager.default.fileExists(atPath: root.appendingPathComponent(name).path)
    }

    func sweep(
        uid: UInt32 = getuid(),
        policy: TemporaryMediaSweepPolicy = TemporaryMediaSweepPolicy(),
        alive: Set<Int32> = [livePID, currentPID]
    ) -> TemporaryMediaSweepReport {
        OmniSmall.sweepStaleTemporaryMedia(
            in: root, now: now, currentPID: currentPID, currentUID: uid,
            policy: policy, isProcessAlive: { alive.contains($0) })
    }
}

private func name(pid: Int32, _ id: UUID = UUID()) -> String { "gloss-media-\(pid)-\(id.uuidString)" }
private func legacyName(_ id: UUID = UUID()) -> String { "gloss-media-\(id.uuidString)" }

@Suite struct TemporaryMediaSweepTests {
    @Test func directoriesAreNamedWithTheOwningPid() throws {
        let id = UUID()
        let directoryName = TemporaryMediaFiles.directoryName(pid: 77, id: id)
        #expect(directoryName == "gloss-media-77-\(id.uuidString)")
        #expect(TemporaryMediaFiles.owner(ofDirectoryNamed: directoryName) == .process(77))
        #expect(TemporaryMediaFiles.owner(ofDirectoryNamed: "gloss-media-\(id.uuidString)") == .legacy)
        for other in ["gloss-media-", "gloss-media-abc", "gloss-media-12-notauuid", "gloss-media-+5-\(id.uuidString)",
                      "gloss-media--5-\(id.uuidString)", "gloss-media-0-\(id.uuidString)",
                      "gloss-media-99999999999-\(id.uuidString)"] {
            #expect(TemporaryMediaFiles.owner(ofDirectoryNamed: other) == .unrecognized, Comment(rawValue: other))
        }
        #expect(TemporaryMediaFiles.owner(ofDirectoryNamed: "unrelated") == nil)

        // The staging helper uses the real pid and removes its directory on every exit path.
        let sandbox = try Sandbox()
        defer { sandbox.remove() }
        var stagedDirectory: URL?
        _ = try TemporaryMediaFiles.withFile(Data([1, 2, 3]), extension: "wav", in: sandbox.root) { file in
            stagedDirectory = file.deletingLastPathComponent()
            #expect(file.lastPathComponent == "input.wav")
            #expect(FileManager.default.fileExists(atPath: file.path))
        }
        let staged = try #require(stagedDirectory)
        #expect(TemporaryMediaFiles.owner(ofDirectoryNamed: staged.lastPathComponent) == .process(getpid()))
        #expect(!FileManager.default.fileExists(atPath: staged.path))

        struct Boom: Error {}
        #expect(throws: Boom.self) {
            try TemporaryMediaFiles.withFile(Data([1]), extension: "mp4", in: sandbox.root) { _ in throw Boom() }
        }
        #expect((try FileManager.default.contentsOfDirectory(atPath: sandbox.root.path)).isEmpty)
    }

    @Test func aDeadOwnerIsSweptOnlyAfterTheGrace() throws {
        let sandbox = try Sandbox()
        defer { sandbox.remove() }
        let old = name(pid: deadPID), fresh = name(pid: deadPID)
        try sandbox.directory(old, age: 120, bytes: 1_000)
        try sandbox.directory(fresh, age: 10)

        let report = sandbox.sweep()
        #expect(!sandbox.exists(old), "a dead owner's directory past the grace must be removed")
        #expect(sandbox.exists(fresh), "a directory inside the grace must be left alone")
        #expect(report.removedDirectories == 1)
        #expect(report.removedBytes == 1_000)
        #expect(report.examined == 2 && report.kept == 1 && report.failed == 0)
    }

    @Test func aLiveOwnerIsKeptUntilTheAbsoluteAgeGuardForPidReuse() throws {
        let sandbox = try Sandbox()
        defer { sandbox.remove() }
        let recent = name(pid: livePID), aged = name(pid: livePID), reusedPid = name(pid: livePID)
        try sandbox.directory(recent, age: 3_600 * 5)        // alive owner, five hours old
        try sandbox.directory(aged, age: 86_400 + 1_000)     // older than 24 h: pid must have been reused
        try sandbox.directory(reusedPid, age: 86_400 - 100)  // just under 24 h
        let report = sandbox.sweep()
        #expect(sandbox.exists(recent))
        #expect(sandbox.exists(reusedPid))
        #expect(!sandbox.exists(aged))
        #expect(report.removedDirectories == 1 && report.kept == 2)
    }

    @Test func legacyDirectoriesAgeOutAfterAnHour() throws {
        let sandbox = try Sandbox()
        defer { sandbox.remove() }
        let young = legacyName(), old = legacyName()
        try sandbox.directory(young, age: 3_000)
        try sandbox.directory(old, age: 3_700)
        let report = sandbox.sweep()
        #expect(sandbox.exists(young))
        #expect(!sandbox.exists(old))
        #expect(report.removedDirectories == 1 && report.kept == 1)
    }

    @Test func unrecognizedNamesOnlyGoAtTheAbsoluteAge() throws {
        let sandbox = try Sandbox()
        defer { sandbox.remove() }
        try sandbox.directory("gloss-media-scratch", age: 3 * 3_600)
        try sandbox.directory("gloss-media-old-thing", age: 86_400 + 60)
        let report = sandbox.sweep()
        #expect(sandbox.exists("gloss-media-scratch"))
        #expect(!sandbox.exists("gloss-media-old-thing"))
        #expect(report.removedDirectories == 1)
    }

    @Test func theCurrentProcessesOwnDirectoriesAreNeverTouched() throws {
        let sandbox = try Sandbox()
        defer { sandbox.remove() }
        let own = name(pid: currentPID), ancient = name(pid: currentPID)
        try sandbox.directory(own, age: 10_000)
        try sandbox.directory(ancient, age: 10 * 86_400)
        // Even with a liveness predicate that calls every pid dead.
        let report = sandbox.sweep(alive: [])
        #expect(sandbox.exists(own) && sandbox.exists(ancient))
        #expect(report.removedDirectories == 0 && report.kept == 2)
    }

    @Test func onlyRealDirectoriesOwnedByTheCurrentUserAreRemoved() throws {
        let sandbox = try Sandbox()
        defer { sandbox.remove() }

        // A symlink named like a scratch directory, pointing at a directory that must survive.
        let victim = sandbox.root.appendingPathComponent("precious", isDirectory: true)
        try FileManager.default.createDirectory(at: victim, withIntermediateDirectories: true)
        try Data("keep".utf8).write(to: victim.appendingPathComponent("file.txt"))
        let linkName = name(pid: deadPID)
        try FileManager.default.createSymbolicLink(
            at: sandbox.root.appendingPathComponent(linkName), withDestinationURL: victim)
        try FileManager.default.setAttributes(
            [.modificationDate: now.addingTimeInterval(-10_000)],
            ofItemAtPath: victim.path)

        // A regular file with a matching name, and an unrelated directory.
        let fileName = name(pid: deadPID)
        try Data("x".utf8).write(to: sandbox.root.appendingPathComponent(fileName))
        try sandbox.directory("unrelated-directory", age: 10 * 86_400)

        // A genuine stale directory — but reported as owned by someone else via the uid.
        let real = name(pid: deadPID)
        try sandbox.directory(real, age: 10_000)

        var report = sandbox.sweep(uid: getuid() &+ 1)
        #expect(sandbox.exists(real), "a directory owned by another user must not be removed")
        #expect(report.removedDirectories == 0)

        report = sandbox.sweep()
        #expect(!sandbox.exists(real))
        #expect(sandbox.exists(linkName), "a symlink must never be followed or removed")
        #expect(FileManager.default.fileExists(atPath: victim.appendingPathComponent("file.txt").path),
                "the symlink target must survive")
        #expect(sandbox.exists(fileName), "a regular file is not a scratch directory")
        #expect(sandbox.exists("unrelated-directory"), "non-matching names are never considered")
        #expect(report.removedDirectories == 1)
    }

    @Test func aMissingDirectoryIsAnEmptySweepNotAFailure() {
        let report = OmniSmall.sweepStaleTemporaryMedia(
            in: URL(fileURLWithPath: "/definitely/not/a/directory-\(UUID().uuidString)"))
        #expect(report == TemporaryMediaSweepReport())
    }

    @Test func realLivenessTreatsOnlyESRCHAsDead() {
        #expect(TemporaryMediaFiles.processIsAlive(getpid()))
        #expect(TemporaryMediaFiles.processIsAlive(1), "launchd exists; EPERM must count as alive")
        #expect(TemporaryMediaFiles.processIsAlive(0), "pid 0 is never a valid owner and must not reach kill()")
        #expect(TemporaryMediaFiles.processIsAlive(-1))
        // A pid that is certainly free: spawn and reap a child, then ask about it.
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/true")
        do {
            try process.run()
            process.waitUntilExit()
            #expect(!TemporaryMediaFiles.processIsAlive(process.processIdentifier))
        } catch {
            Issue.record("could not spawn a child to test liveness: \(error)")
        }
    }
}
