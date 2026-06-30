// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "LadaMac",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "LadaMac", targets: ["LadaMac"])
    ],
    targets: [
        .executableTarget(
            name: "LadaMac",
            path: "Sources/LadaMac",
            exclude: ["Resources"]
        ),
        .testTarget(
            name: "LadaMacTests",
            dependencies: ["LadaMac"],
            path: "Tests/LadaMacTests"
        )
    ]
)
