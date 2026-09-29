// swift-tools-version: 6.0
import PackageDescription

// OpenAI-compatible /v1/embeddings server daemon for local model bundles.
// The Core ML inference and media preprocessing implementation is owned by this executable.
// HTTP lives on Network.framework; swift-transformers supplies local tokenization.
let package = Package(
    name: "GlossematicsServer",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "gloss-server", targets: ["gloss-server"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.17"),
    ],
    targets: [
        .executableTarget(
            name: "gloss-server",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/gloss-server",
            resources: [
                // The /docs page. Edited as a real HTML file in the repo; packaged into the
                // executable's resource bundle at build time and templated by DocsPage.swift.
                .copy("Resources/docs.html"),
                .copy("Resources/mel_filters.f32"),
                .copy("Resources/mel_window.f32"),
            ]
        ),
        .testTarget(
            name: "GlossServerTests",
            dependencies: ["gloss-server"],
            path: "Tests/GlossServerTests"
        ),
    ],
    swiftLanguageModes: [.v6]
)
