import Foundation

struct NativeCoreMLCapabilities: Equatable, Sendable {
    let detectorModelURL: URL?
    let restorerModelURL: URL?

    var isDetectorAvailable: Bool {
        isCompiledModelPresent(detectorModelURL)
    }

    var isRestorerAvailable: Bool {
        isCompiledModelPresent(restorerModelURL)
    }

    var statusTitle: String {
        if isDetectorAvailable && isRestorerAvailable {
            return "Detector + restorer ready"
        }
        if isDetectorAvailable {
            return "Detector ready"
        }
        if isRestorerAvailable {
            return "Restorer ready"
        }
        return "Waiting"
    }

    var statusDetail: String {
        let detector = isDetectorAvailable
            ? "Core ML detector bundled"
            : "Core ML detector missing"
        let restorer = isRestorerAvailable
            ? "Core ML restorer bundled"
            : "Core ML restorer missing"
        return "\(detector) · \(restorer)"
    }

    static func current(bundle: Bundle = .main) -> NativeCoreMLCapabilities {
        NativeCoreMLCapabilities(
            detectorModelURL: NativeModelBundleCatalog.modelURL(
                named: NativeModelBundleCatalog.detectorModelName,
                in: bundle
            ),
            restorerModelURL: NativeModelBundleCatalog.modelURL(
                named: NativeModelBundleCatalog.restorerModelName,
                in: bundle
            )
        )
    }

    private func isCompiledModelPresent(_ url: URL?) -> Bool {
        guard let url else {
            return false
        }
        return url.pathExtension == "mlmodelc" &&
            FileManager.default.fileExists(atPath: url.path)
    }
}
