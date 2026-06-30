import Foundation

enum NativeModelBundleCatalog {
    static let detectorModelName = "LadaMosaicDetector"
    static let restorerModelName = "LadaMosaicRestorer"

    static func modelURL(
        named name: String,
        in bundle: Bundle = .main
    ) -> URL? {
        bundle.url(forResource: name, withExtension: "mlmodelc")
    }

    static func coreAIModelURL(
        named name: String,
        in bundle: Bundle = .main
    ) -> URL? {
        bundle.url(forResource: name, withExtension: "aimodel")
    }

    static func expectedModelNames() -> [String] {
        [
            detectorModelName,
            restorerModelName
        ]
    }

    static func expectedCoreAIModelNames() -> [String] {
        expectedModelNames()
    }
}
