// swift-tools-version: 6.0
import PackageDescription

// OpenAI-compatible /v1/embeddings server daemon for local model bundles.
// Builds against the local `Glossematics` library product and drives the same public API
// (OmniSmall + GlossModelBundle) that host applications use. HTTP lives on Network.framework,
// so the only external dependency is the tokenizer (already a dependency of the library).
let package = Package(
    name: "GlossematicsServer",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "gloss-server", targets: ["gloss-server"]),
    ],
    dependencies: [
        .package(name: "Glossematics", path: "../../Models/GlossematicsSDK/SwiftPackages/Glossematics"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.17"),
    ],
    targets: [
        .executableTarget(
            name: "gloss-server",
            dependencies: [
                .product(name: "Glossematics", package: "Glossematics"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/gloss-server",
            resources: [
                // The /docs page. Edited as a real HTML file in the repo; packaged into the
                // executable's resource bundle at build time and templated by DocsPage.swift.
                .copy("Resources/docs.html"),
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
