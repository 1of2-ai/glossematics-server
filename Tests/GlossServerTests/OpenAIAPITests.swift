import Foundation
import XCTest
@testable import gloss_server

final class OpenAIAPITests: XCTestCase {
    func testMessageInputInterleavesTextImageAndAudio() throws {
        let payload = Data([1, 2, 3]).base64EncodedString()
        let data = Data(#"{"model":"m","input":{"type":"message","role":"user","content":[{"type":"input_text","text":"a photo of"},{"type":"input_image","image_url":"data:image/png;base64,\#(payload)"},{"type":"input_audio","audio_url":"data:audio/wav;base64,\#(payload)"}]}}"#.utf8)
        let body = try JSONDecoder().decode(EmbeddingsRequestBody.self, from: data)
        guard case let .message(parts) = body.items.first else { return XCTFail("expected a message") }
        XCTAssertEqual(parts.count, 3)
        guard case let .text(text) = parts[0], case .image = parts[1], case .audio = parts[2] else {
            return XCTFail("unexpected message parts")
        }
        XCTAssertEqual(text, "a photo of")
    }

    func testMessageRejectsOtherRolesNestedMessagesAndVideo() {
        let payload = Data([1, 2, 3]).base64EncodedString()
        for input in [
            #"{"type":"message","role":"assistant","content":[{"type":"input_text","text":"x"}]}"#,
            #"{"type":"message","content":[]}"#,
            #"{"type":"message","content":[{"type":"message","content":[{"type":"input_text","text":"x"}]}]}"#,
            #"{"type":"message","content":[{"type":"input_video","video_url":"data:video/mp4;base64,\#(payload)"}]}"#,
        ] {
            let data = Data(#"{"model":"m","input":\#(input)}"#.utf8)
            XCTAssertThrowsError(try JSONDecoder().decode(EmbeddingsRequestBody.self, from: data), input)
        }
    }

    func testStandardStringInputIsAccepted() throws {
        let data = Data(#"{"model":"m","input":["one","two"],"dimensions":512,"encoding_format":"base64","user":"u"}"#.utf8)
        let body = try JSONDecoder().decode(EmbeddingsRequestBody.self, from: data)
        XCTAssertEqual(body.items.count, 2)
        guard case let .text(first) = body.items[0],
              case let .text(second) = body.items[1] else {
            return XCTFail("expected string inputs")
        }
        XCTAssertEqual(first, "one")
        XCTAssertEqual(second, "two")
        XCTAssertEqual(body.dimensions, 512)
        XCTAssertEqual(body.encodingFormat, "base64")
        XCTAssertEqual(body.user, "u")
    }

    func testTokenInputsAreAccepted() throws {
        let single = try JSONDecoder().decode(
            EmbeddingsRequestBody.self,
            from: Data(#"{"model":"m","input":[12,34,56]}"#.utf8))
        guard case let .tokenIDs(ids) = single.items.first else {
            return XCTFail("expected one token-ID input")
        }
        XCTAssertEqual(ids, [12, 34, 56])

        let batch = try JSONDecoder().decode(
            EmbeddingsRequestBody.self,
            from: Data(#"{"model":"m","input":[[12,34],[56,78]]}"#.utf8))
        XCTAssertEqual(batch.items.count, 2)
        guard case let .tokenIDs(first) = batch.items[0],
              case let .tokenIDs(second) = batch.items[1] else {
            return XCTFail("expected token-ID batch")
        }
        XCTAssertEqual(first, [12, 34])
        XCTAssertEqual(second, [56, 78])
    }

    func testResponsesShapedImageDataURLIsAccepted() throws {
        let bytes = Data([0x89, 0x50, 0x4E, 0x47])
        let dataURL = "data:image/png;base64,\(bytes.base64EncodedString())"
        let data = Data(#"{"model":"m","input":{"type":"input_image","image_url":"\#(dataURL)"}}"#.utf8)
        let body = try JSONDecoder().decode(EmbeddingsRequestBody.self, from: data)
        guard case let .image(decoded) = body.items.first else {
            return XCTFail("expected image data")
        }
        XCTAssertEqual(decoded, bytes)
    }

    func testAudioAndVideoDataURLsAreAccepted() throws {
        let bytes = Data([1, 2, 3, 4])
        let payload = bytes.base64EncodedString()
        let data = Data(#"{"model":"m","input":[{"type":"input_audio","audio_url":"data:audio/wav;base64,\#(payload)"},{"type":"input_video","video_url":"data:video/mp4;base64,\#(payload)"}]}"#.utf8)
        let body = try JSONDecoder().decode(EmbeddingsRequestBody.self, from: data)
        guard case let .audio(audio) = body.items[0],
              case let .video(video) = body.items[1] else {
            return XCTFail("expected audio and video data")
        }
        XCTAssertEqual(audio, bytes)
        XCTAssertEqual(video, bytes)
    }

    func testMediaDataURLsRejectRemoteAndUnsupportedFormats() {
        for input in [
            #"{"type":"input_audio","audio_url":"https://example.com/sound.wav"}"#,
            #"{"type":"input_audio","audio_url":"data:audio/mp3;base64,AQID"}"#,
            #"{"type":"input_video","video_url":"data:video/quicktime;base64,AQID"}"#,
            #"{"type":"input_video","video_url":"data:video/mp4;base64,AQI D"}"#,
        ] {
            let data = Data(#"{"model":"m","input":\#(input)}"#.utf8)
            XCTAssertThrowsError(try JSONDecoder().decode(EmbeddingsRequestBody.self, from: data))
        }
    }

    func testRetrievalRoleFieldsDecode() throws {
        func role(_ json: String) throws -> (RetrievalRole?, String?) {
            let body = try JSONDecoder().decode(EmbeddingsRequestBody.self, from: Data(json.utf8))
            return (body.retrievalRole, body.retrievalRoleField)
        }
        XCTAssertEqual(try role(#"{"model":"m","input":"x"}"#).0, nil)
        XCTAssertEqual(try role(#"{"model":"m","input":"x","task":"retrieval.query"}"#).0, .query)
        XCTAssertEqual(try role(#"{"model":"m","input":"x","task":"Retrieval.Passage"}"#).0, .document)
        XCTAssertEqual(try role(#"{"model":"m","input":"x","role":"document"}"#).0, .document)
        XCTAssertEqual(try role(#"{"model":"m","input":"x","input_type":"query"}"#).1, "input_type")
        XCTAssertEqual(try role(#"{"model":"m","input":"x","task":"query","role":"query"}"#).0, .query)
        for json in [
            #"{"model":"m","input":"x","task":"text-matching"}"#,
            #"{"model":"m","input":"x","task":"query","role":"document"}"#,
            #"{"model":"m","input":"x","input_type":7}"#,
        ] {
            XCTAssertThrowsError(try JSONDecoder().decode(EmbeddingsRequestBody.self, from: Data(json.utf8)), json)
        }
    }

    func testModelFamilyIsDetectedFromTheManifest() throws {
        XCTAssertEqual(try ModelFamily.detect(manifest: ["format": "bidirlm-omni-ane-v2"]), .bidirlm)
        XCTAssertEqual(try ModelFamily.detect(manifest: ["format": "bidirlm-omni-streamed-v2"]), .bidirlm)
        XCTAssertEqual(try ModelFamily.detect(manifest: [
            "formatVersion": 2, "modelID": "jinaai/jina-embeddings-v5-omni-small"]), .jinaOmniSmall)
        XCTAssertThrowsError(try ModelFamily.detect(manifest: [
            "formatVersion": 2, "modelID": "jinaai/jina-embeddings-v5-omni-nano"]))
        XCTAssertThrowsError(try ModelFamily.detect(manifest: ["formatVersion": 1, "modelID": "x"]))
        XCTAssertThrowsError(try ModelFamily.detect(manifest: [:]))
    }

    func testNonstandardOrUnsafeInputIsRejected() {
        let nonstandard = Data(#"{"model":"m","input":"x","instruction":"Represent this"}"#.utf8)
        XCTAssertThrowsError(try JSONDecoder().decode(EmbeddingsRequestBody.self, from: nonstandard))

        let path = Data(#"{"model":"m","input":{"image_path":"/tmp/frame.png"}}"#.utf8)
        XCTAssertThrowsError(try JSONDecoder().decode(EmbeddingsRequestBody.self, from: path))

        let remoteURL = Data(#"{"model":"m","input":{"type":"input_image","image_url":"https://example.com/frame.png"}}"#.utf8)
        XCTAssertThrowsError(try JSONDecoder().decode(EmbeddingsRequestBody.self, from: remoteURL))
    }

    func testInvalidTokenIDsAreRejected() {
        let data = Data(#"{"model":"m","input":[-1]}"#.utf8)
        XCTAssertThrowsError(try JSONDecoder().decode(EmbeddingsRequestBody.self, from: data))
    }
}
