import Foundation
import Testing
@_spi(Server) @testable import gloss_server

/// Startup verification of every Core ML function the validated bundle declares.
///
/// The golden fixture always runs (every function returns 1/32 * ones, so every consistency
/// cosine is exactly 1). The real bundle is env-gated:
/// `GLOSS_JINA_BUNDLE=<real bundle> swift test --filter verifyAllFunctionsOnTheRealBundle`.

private let goldenFixture = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .appendingPathComponent("Fixtures/JinaV5OmniSmall.w8a16.dummy.bundle")

private let testSpace = "glossematics:omni-small:sha256:" + String(repeating: "a", count: 64)
private let testArtifact = "sha256:" + String(repeating: "c", count: 64)

private func realBundle() -> URL? {
    ProcessInfo.processInfo.environment["GLOSS_JINA_BUNDLE"].map { URL(fileURLWithPath: $0, isDirectory: true) }
}

/// Every function the pinned Omni Small contract declares, as `model.function`.
private let expectedFunctions: [String] = {
    var names = [String]()
    names += [32, 64, 128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768].map { "text.bucket_\($0)" }
    names += [(64, 32), (32, 64), (16, 128), (8, 256), (4, 512)].map { "text.bucket_\($0.1)_b\($0.0)" }
    names += [1_024, 1_600, 2_304, 3_072, 4_032, 5_120].map { "image.f\($0)" }
    names += [256, 512, 1_024, 2_048].map { "video.f\($0)" }
    names += [200, 400, 800, 1_600, 3_200].map { "audio.f\($0)" }
    names += [128, 256, 512, 1_024, 2_048].map { "embed.f\($0)" }
    names += [128, 256, 512, 1_024, 2_048].map { "decoder.f\($0)" }
    return names
}()

/// A model over the golden fixture whose verification outputs can be corrupted per function.
private func fixtureModel(
    faults: OmniSmallVerificationFaultInjector? = nil
) throws -> OmniSmall {
    let bundle = try GlossModelBundle(url: goldenFixture)
    return OmniSmall(
        backend: OmniSmallProductionBackend(bundle: bundle, verificationFaults: faults),
        dimensions: .d1024,
        space: testSpace,
        artifactFingerprint: testArtifact)
}

@Suite(.serialized) struct FunctionVerification {
    @Test func goldenFixtureVerifiesEveryDeclaredFunction() async throws {
        let model = try await OmniSmall.load(from: goldenFixture)
        let seen = ProgressLog()
        let report = try await model.verifyAllFunctions { seen.append($0.name) }

        #expect(report.functions.map(\.name) == expectedFunctions,
                "verification must cover every declared function, in a stable order")
        #expect(seen.names == expectedFunctions, "progress must report each function as it completes")
        #expect(report.allPassed && report.failedCount == 0)
        #expect(report.passedCount == expectedFunctions.count)
        // The fixture returns a constant, so every sibling comparison is exact.
        for check in report.functions where check.referenceFunction != nil {
            let cosine = try #require(check.minimumCosine)
            #expect(abs(cosine - 1) < 1e-6, "\(check.name) cosine \(cosine)")
            #expect(check.cosineThreshold != nil)
        }
        // Anchors have no reference; everything else names the function it was compared with.
        let anchors = report.functions.filter { $0.referenceFunction == nil }.map(\.name)
        #expect(anchors == ["text.bucket_32", "image.f1024", "video.f256", "audio.f200", "embed.f128", "decoder.f128"])
        #expect(report.functions.filter { $0.name.hasPrefix("text.") && $0.function.contains("_b") }
            .allSatisfy { $0.referenceFunction == "bucket_" + $0.function.split(separator: "_")[1] })
        #expect(report.functions.allSatisfy { $0.computeUnits != nil })
        #expect(report.functions.first { $0.name == "text.bucket_32" }?.computeUnits == "cpu+ane")
        #expect(report.functions.first { $0.name == "text.bucket_32768" }?.computeUnits == "cpu+gpu")
        #expect(report.functions.first { $0.name == "image.f1024" }?.computeUnits == "cpu+gpu")
        #expect(report.wallMilliseconds > 0)
        #expect(report.loadMilliseconds > 0 && report.runMilliseconds > 0)
        #expect(report.residentBytesAfter > 0 && report.footprintBytesAfter > 0)

        // Verification leaves every function loaded, so a second run finds them warm.
        let again = try await model.verifyAllFunctions()
        #expect(again.allPassed)
        #expect(again.functions.allSatisfy { $0.loadMilliseconds < 100 },
                "a warm function must not reload: \(again.functions.filter { $0.loadMilliseconds >= 100 }.map(\.name))")
    }

    @Test func reportIsEncodableForHealth() async throws {
        let model = try await OmniSmall.load(from: goldenFixture)
        let report = try await model.verifyAllFunctions()
        let json = try JSONEncoder().encode(report)
        let decoded = try JSONDecoder().decode(OmniSmallVerificationReport.self, from: json)
        #expect(decoded == report)
        let object = try #require(JSONSerialization.jsonObject(with: json) as? [String: Any])
        for key in ["functions", "wallMilliseconds", "loadMilliseconds", "runMilliseconds",
                    "residentBytesBefore", "residentBytesAfter", "footprintBytesBefore", "footprintBytesAfter"] {
            #expect(object[key] != nil, "missing \(key)")
        }
    }

    @Test func nonFiniteFeaturesFailNamingTheFunction() async throws {
        let model = try fixtureModel { function, values in
            if function == "image.f1600", !values.isEmpty { values[values.count / 2] = .nan }
        }
        do {
            _ = try await model.verifyAllFunctions()
            Issue.record("a NaN in an encoder output must fail verification")
        } catch let error as OmniSmallVerificationError {
            #expect(error.function == "image.f1600")
            guard case let .invalidOutput(reason) = error.reason else {
                Issue.record("unexpected reason \(error.reason)")
                return
            }
            #expect(reason.contains("non-finite"))
            // The run completes: every other function is still checked and reported.
            #expect(error.report.functions.count == expectedFunctions.count)
            #expect(error.report.failedCount == 1)
            #expect(error.description.contains("image.f1600"))
        }
    }

    @Test func finiteButWrongBucketFailsTheConsistencyCheck() async throws {
        // Scale-free corruption: rotate one text bucket's unit embedding so it stays finite,
        // correctly shaped, and unit norm — only its agreement with bucket_32 exposes it.
        let model = try fixtureModel { function, values in
            if function == "text.bucket_2048" {
                values = [Float](repeating: 0, count: values.count)
                values[0] = 1
            }
        }
        do {
            _ = try await model.verifyAllFunctions()
            Issue.record("a bucket that disagrees with bucket_32 must fail verification")
        } catch let error as OmniSmallVerificationError {
            #expect(error.function == "text.bucket_2048")
            guard case let .inconsistent(cosine, threshold, reference) = error.reason else {
                Issue.record("unexpected reason \(error.reason)")
                return
            }
            #expect(reference == "bucket_32")
            #expect(cosine < threshold)
            #expect(error.report.failedCount == 1)
        }
    }

    @Test func batchRowThatDisagreesWithTheSingleRowFunctionFails() async throws {
        let model = try fixtureModel { function, values in
            if function == "text.bucket_128_b16", values.count == 1_024 {
                // Damage every row: the last row processed is what a naive check would keep.
                values = [Float](repeating: 0, count: 1_024)
                values[7] = 1
            }
        }
        do {
            _ = try await model.verifyAllFunctions()
            Issue.record("a batch function that disagrees with its single-row twin must fail")
        } catch let error as OmniSmallVerificationError {
            #expect(error.function == "text.bucket_128_b16")
            guard case let .inconsistent(_, _, reference) = error.reason else {
                Issue.record("unexpected reason \(error.reason)")
                return
            }
            #expect(reference == "bucket_128")
        }
    }

    @Test func nonUnitEmbeddingsAndZeroVectorsFail() async throws {
        let scaled = try fixtureModel { function, values in
            if function == "decoder.f512" { values = values.map { $0 * 2 } }
        }
        do {
            _ = try await scaled.verifyAllFunctions()
            Issue.record("a non-unit-norm embedding must fail verification")
        } catch let error as OmniSmallVerificationError {
            #expect(error.function == "decoder.f512")
            guard case let .invalidOutput(reason) = error.reason else {
                Issue.record("unexpected reason \(error.reason)")
                return
            }
            #expect(reason.contains("L2-normalized"))
        }

        let zero = try fixtureModel { function, values in
            if function == "embed.f256" { values = [Float](repeating: 0, count: values.count) }
        }
        do {
            _ = try await zero.verifyAllFunctions()
            Issue.record("an all-zero embed output must fail its consistency check")
        } catch let error as OmniSmallVerificationError {
            // Zero vectors have no defined cosine; that must fail, never pass by comparison to NaN.
            #expect(error.function == "embed.f256")
            guard case .inconsistent = error.reason else {
                Issue.record("unexpected reason \(error.reason)")
                return
            }
        }
    }

    @Test func minimumRowCosineFindsTheWorstRow() {
        let good: [Float] = [1, 0, 0, 1, 1, 1]
        let same = OmniSmallFunctionVerifier.minimumRowCosine(good, good, width: 2)
        #expect(abs(same - 1) < 1e-12)
        let worse: [Float] = [1, 0, 1, 0, 1, 1]   // second row orthogonal to [0, 1]
        #expect(OmniSmallFunctionVerifier.minimumRowCosine(good, worse, width: 2) < 1e-12)
        #expect(OmniSmallFunctionVerifier.minimumRowCosine(good, [1, 0], width: 2).isNaN)
    }

    @Test func aBackendWithoutNativeFunctionsReportsSetupFailure() async throws {
        let model = OmniSmall(
            backend: NoNativeFunctionsBackend(),
            dimensions: .d32, space: testSpace, artifactFingerprint: testArtifact)
        do {
            _ = try await model.verifyAllFunctions()
            Issue.record("a backend with nothing to verify must not report success")
        } catch let error as OmniSmallVerificationError {
            #expect(error.function == "setup")
            guard case .setupFailed = error.reason else {
                Issue.record("unexpected reason \(error.reason)")
                return
            }
        }
    }

    @Test func cancellationStopsVerification() async throws {
        let model = try await OmniSmall.load(from: goldenFixture)
        let task = Task { try await model.verifyAllFunctions() }
        task.cancel()
        do {
            _ = try await task.value
            Issue.record("a cancelled verification must not complete")
        } catch is CancellationError {
            // Expected.
        }
    }

    /// Real bundle: every function, with the measured numbers printed for the record. Gated on
    /// `GLOSS_JINA_BUNDLE`; takes minutes on a cold machine.
    @Test(.enabled(if: realBundle() != nil, "Set GLOSS_JINA_BUNDLE to the production bundle"))
    func verifyAllFunctionsOnTheRealBundle() async throws {
        let bundle = try #require(realBundle())
        let model = try await OmniSmall.load(from: bundle)
        let report: OmniSmallVerificationReport
        do {
            report = try await model.verifyAllFunctions { check in
                print("  verified \(check.name) [\(check.computeUnits ?? "?")] load \(String(format: "%.0f", check.loadMilliseconds)) ms, run \(String(format: "%.0f", check.runMilliseconds)) ms\(check.minimumCosine.map { ", cosine \($0)" } ?? "")\(check.failure.map { " FAILED: \($0)" } ?? "")")
            }
        } catch let error as OmniSmallVerificationError {
            Issue.record("\(error)")
            report = error.report
        }
        let mib = { (bytes: Int64) in String(format: "%.0f", Double(bytes) / 1_048_576) }
        print("""
        verification: \(report.passedCount)/\(report.functions.count) functions passed in \(String(format: "%.1f", report.wallMilliseconds / 1000)) s \
        (load \(String(format: "%.1f", report.loadMilliseconds / 1000)) s, run \(String(format: "%.1f", report.runMilliseconds / 1000)) s); \
        resident \(mib(Int64(report.residentBytesBefore))) -> \(mib(Int64(report.residentBytesAfter))) MiB (delta \(mib(report.residentDeltaBytes))), \
        footprint \(mib(Int64(report.footprintBytesBefore))) -> \(mib(Int64(report.footprintBytesAfter))) MiB (delta \(mib(report.footprintDeltaBytes)))
        """)
        #expect(report.functions.map(\.name) == expectedFunctions)
        #expect(report.allPassed)
    }
}

private final class ProgressLog: @unchecked Sendable {
    private let lock = NSLock()
    private var stored = [String]()
    func append(_ name: String) { lock.lock(); stored.append(name); lock.unlock() }
    var names: [String] { lock.lock(); defer { lock.unlock() }; return stored }
}

/// A backend with no native functions: the protocol default for verification applies.
private actor NoNativeFunctionsBackend: OmniSmallBackend {
    func prepareTexts(_ texts: [String], role: OmniSmallRole) async throws -> [ValidatedText] { [] }
    func embedTexts(_ rows: [ValidatedText], dimensions: OmniSmall.Dimensions) async throws -> [[Float]] { [] }
    func embedMedia(_ input: OmniSmall.Input, role: OmniSmallRole, dimensions: OmniSmall.Dimensions) async throws -> [Float] { [] }
}
