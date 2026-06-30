import Foundation

#if canImport(CoreAI)
import CoreAI
#endif

struct NativeCoreAICapabilities: Equatable, Sendable {
    struct ModelAssetSummary: Equatable, Sendable {
        let name: String
        let isValid: Bool
        let functions: [String]
        let computeTypes: [String]
        let storageTypes: [String]
        let error: String?

        var shortStatus: String {
            if let error {
                return "\(name): invalid · \(error)"
            }
            let functionText = functions.isEmpty ? "no functions reported" : functions.joined(separator: ", ")
            let computeText = computeTypes.isEmpty ? "compute unknown" : computeTypes.joined(separator: "+")
            return "\(name): \(functionText) · \(computeText)"
        }
    }

    let isFrameworkPresent: Bool
    let isSwiftModuleAvailable: Bool
    let deviceArchitectureName: String?
    let availableComputeUnits: [String]
    let detectorAssetURL: URL?
    let restorerAssetURL: URL?
    let detectorAssetSummary: ModelAssetSummary?
    let restorerAssetSummary: ModelAssetSummary?

    var hasAnyAsset: Bool {
        detectorAssetURL != nil || restorerAssetURL != nil
    }

    var hasInvalidAsset: Bool {
        [
            detectorAssetSummary,
            restorerAssetSummary
        ]
        .compactMap(\.self)
        .contains { !$0.isValid }
    }

    var isReadyForAssets: Bool {
        isFrameworkPresent && isSwiftModuleAvailable && hasAnyAsset && !hasInvalidAsset
    }

    var statusDetail: String {
        guard isFrameworkPresent else {
            return "Core AI framework unavailable on this macOS runtime"
        }
        guard isSwiftModuleAvailable else {
            return "Core AI framework found; build with macOS 27 SDK / Swift 6.4+ to enable typed runtime integration"
        }
        let units = availableComputeUnits.isEmpty
            ? "compute units unknown"
            : availableComputeUnits.joined(separator: "+")
        let architecture = deviceArchitectureName ?? "unknown architecture"
        if hasAnyAsset {
            let assets = assetSummaryText
            if assets.isEmpty {
                return "Core AI ready · \(units) · \(architecture)"
            }
            if hasInvalidAsset {
                return "Core AI asset found but needs attention · \(units) · \(architecture) · \(assets)"
            }
            return "Core AI ready · \(units) · \(architecture) · \(assets)"
        }
        return "Core AI runtime ready · waiting for .aimodel assets · \(units) · \(architecture)"
    }

    private var assetSummaryText: String {
        [
            detectorAssetSummary?.shortStatus,
            restorerAssetSummary?.shortStatus
        ]
        .compactMap(\.self)
        .joined(separator: " · ")
    }

    static func current(
        bundle: Bundle = .main,
        fileManager: FileManager = .default
    ) -> NativeCoreAICapabilities {
        let detector = NativeModelBundleCatalog.coreAIModelURL(
            named: NativeModelBundleCatalog.detectorModelName,
            in: bundle
        )
        let restorer = NativeModelBundleCatalog.coreAIModelURL(
            named: NativeModelBundleCatalog.restorerModelName,
            in: bundle
        )

        #if canImport(CoreAI)
        if #available(macOS 27.0, *) {
            return NativeCoreAICapabilities(
                isFrameworkPresent: true,
                isSwiftModuleAvailable: true,
                deviceArchitectureName: AIModel.deviceArchitectureName,
                availableComputeUnits: ComputeUnitKind.availableKinds
                    .map(\.displayName)
                    .sorted(),
                detectorAssetURL: detector,
                restorerAssetURL: restorer,
                detectorAssetSummary: detector.map {
                    NativeCoreAIAssetInspector.inspect(
                        name: NativeModelBundleCatalog.detectorModelName,
                        url: $0
                    )
                },
                restorerAssetSummary: restorer.map {
                    NativeCoreAIAssetInspector.inspect(
                        name: NativeModelBundleCatalog.restorerModelName,
                        url: $0
                    )
                }
            )
        }
        #endif

        return NativeCoreAICapabilities(
            isFrameworkPresent: fileManager.fileExists(
                atPath: "/System/Library/Frameworks/CoreAI.framework"
            ),
            isSwiftModuleAvailable: false,
            deviceArchitectureName: nil,
            availableComputeUnits: [],
            detectorAssetURL: detector,
            restorerAssetURL: restorer,
            detectorAssetSummary: nil,
            restorerAssetSummary: nil
        )
    }
}

#if canImport(CoreAI)
@available(macOS 27.0, *)
private enum NativeCoreAIAssetInspector {
    static func inspect(name: String, url: URL) -> NativeCoreAICapabilities.ModelAssetSummary {
        guard AIModelAsset.isValid(at: url) else {
            return NativeCoreAICapabilities.ModelAssetSummary(
                name: name,
                isValid: false,
                functions: [],
                computeTypes: [],
                storageTypes: [],
                error: "AIModelAsset.isValid returned false"
            )
        }

        do {
            let asset = try AIModelAsset(contentsOf: url)
            let summary = try asset.summary(includingStatistics: false)
            return NativeCoreAICapabilities.ModelAssetSummary(
                name: name,
                isValid: true,
                functions: summary?.functions.map(\.signatureText).sorted() ?? [],
                computeTypes: summary?.computeTypes.sorted() ?? [],
                storageTypes: summary?.storageTypes
                    .map { "\($0.typeName):\($0.count)" }
                    .sorted() ?? [],
                error: nil
            )
        } catch {
            return NativeCoreAICapabilities.ModelAssetSummary(
                name: name,
                isValid: false,
                functions: [],
                computeTypes: [],
                storageTypes: [],
                error: String(describing: error)
            )
        }
    }
}

@available(macOS 27.0, *)
private extension AIModelAsset.FunctionDescriptor {
    var signatureText: String {
        let inputText = inputs.map(\.typedName).joined(separator: ", ")
        let outputText = outputs.map(\.typedName).joined(separator: ", ")
        return "\(name)(\(inputText)) -> \(outputText)"
    }
}

@available(macOS 27.0, *)
private extension AIModelAsset.ValueDescriptor {
    var typedName: String {
        "\(name):\(typeName)"
    }
}

@available(macOS 27.0, *)
private extension ComputeUnitKind {
    var displayName: String {
        switch self {
        case .cpu: "CPU"
        case .gpu: "GPU"
        case .neuralEngine: "Neural Engine"
        @unknown default: "Unknown"
        }
    }
}
#endif
