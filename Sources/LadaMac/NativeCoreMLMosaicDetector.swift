import CoreML
import CoreVideo
import Foundation

enum NativeCoreMLDetectorAvailability: Sendable {
    case missing(expectedURL: URL?)
    case available(modelURL: URL)

    var isAvailable: Bool {
        switch self {
        case .available:
            true
        case .missing:
            false
        }
    }
}

struct NativeCoreMLMosaicDetector: NativePixelBufferMosaicDetector {
    static let defaultModelName = NativeModelBundleCatalog.detectorModelName
    static let inputFeatureName = "images"
    static let imageInputFeatureName = "image"
    static let outputFeatureName = "output0"
    static let prototypeFeatureName = "output1"
    static let ultralyticsOutputFeatureName = "var_1324"
    static let ultralyticsPrototypeFeatureName = "var_1362"

    let modelURL: URL?
    let postprocessor: NativeYOLOPostprocessor
    private let modelCache: NativeCoreMLModelCache

    init(
        modelURL: URL? = NativeModelBundleCatalog.modelURL(named: Self.defaultModelName),
        postprocessor: NativeYOLOPostprocessor = NativeYOLOPostprocessor(),
        modelCache: NativeCoreMLModelCache = NativeCoreMLModelCache()
    ) {
        self.modelURL = modelURL
        self.postprocessor = postprocessor
        self.modelCache = modelCache
    }

    var availability: NativeCoreMLDetectorAvailability {
        guard let modelURL else {
            return .missing(expectedURL: nil)
        }
        let exists = FileManager.default.fileExists(atPath: modelURL.path)
        let isCompiledModel = modelURL.pathExtension == "mlmodelc"
        guard exists, isCompiledModel else {
            return .missing(expectedURL: modelURL)
        }
        return .available(modelURL: modelURL)
    }

    func detections(for frame: NativeBGRAFrame) throws -> [NativeMosaicDetection] {
        guard case .available(let modelURL) = availability else {
            return []
        }

        let model = try modelCache.model(
            at: modelURL,
            configuration: makeModelConfiguration()
        )
        let input = try makeInputFeatureProvider(from: frame, model: model)
        let prediction = try model.prediction(from: input)
        return try detections(
            from: prediction,
            frameWidth: frame.width,
            frameHeight: frame.height
        )
    }

    func detections(for pixelBuffer: CVPixelBuffer) throws -> [NativeMosaicDetection] {
        guard case .available(let modelURL) = availability else {
            return []
        }

        let model = try modelCache.model(
            at: modelURL,
            configuration: makeModelConfiguration()
        )
        let input = try makeInputFeatureProvider(from: pixelBuffer, model: model)
        let prediction = try model.prediction(from: input)
        return try detections(
            from: prediction,
            frameWidth: CVPixelBufferGetWidth(pixelBuffer),
            frameHeight: CVPixelBufferGetHeight(pixelBuffer)
        )
    }

    func makeModelConfiguration() -> MLModelConfiguration {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        return configuration
    }

    func makeInputFeatureProvider(
        from frame: NativeBGRAFrame,
        model: MLModel
    ) throws -> MLFeatureProvider {
        if model.modelDescription.inputDescriptionsByName[Self.imageInputFeatureName]?.type == .image {
            return try makeImageInputFeatureProvider(from: frame)
        }
        return try makeInputFeatureProvider(from: frame)
    }

    func makeInputFeatureProvider(
        from pixelBuffer: CVPixelBuffer,
        model: MLModel
    ) throws -> MLFeatureProvider {
        if model.modelDescription.inputDescriptionsByName[Self.imageInputFeatureName]?.type == .image {
            return try makeImageInputFeatureProvider(from: pixelBuffer)
        }
        return try makeInputFeatureProvider(from: pixelBuffer)
    }

    func makeInputFeatureProvider(from frame: NativeBGRAFrame) throws -> MLFeatureProvider {
        guard frame.width > 0,
              frame.height > 0,
              frame.bytes.count == frame.width * frame.height * 4
        else {
            throw NativeCoreMLDetectorError.invalidFrameData
        }

        let inputSize = postprocessor.modelInputSize
        guard let layout = NativeYOLOInputLayout.aspectFit(
            modelInputSize: inputSize,
            frameWidth: frame.width,
            frameHeight: frame.height
        ) else {
            throw NativeCoreMLDetectorError.invalidFrameData
        }
        let array = try MLMultiArray(
            shape: [1, 3, NSNumber(value: inputSize), NSNumber(value: inputSize)],
            dataType: .float32
        )
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        let planeSize = inputSize * inputSize
        let letterboxValue = Float(114) / 255

        for y in 0..<inputSize {
            for x in 0..<inputSize {
                let destinationIndex = y * inputSize + x
                let unpaddedX = Float(x) - layout.padX
                let unpaddedY = Float(y) - layout.padY
                guard unpaddedX >= 0,
                      unpaddedY >= 0,
                      unpaddedX < Float(layout.scaledWidth),
                      unpaddedY < Float(layout.scaledHeight)
                else {
                    pointer[destinationIndex] = letterboxValue
                    pointer[planeSize + destinationIndex] = letterboxValue
                    pointer[planeSize * 2 + destinationIndex] = letterboxValue
                    continue
                }

                let sourceX = min(
                    frame.width - 1,
                    Int((unpaddedX / layout.scale).rounded(.down))
                )
                let sourceY = min(
                    frame.height - 1,
                    Int((unpaddedY / layout.scale).rounded(.down))
                )
                let sourceIndex = (sourceY * frame.width + sourceX) * 4
                pointer[destinationIndex] = Float(frame.bytes[sourceIndex + 2]) / 255
                pointer[planeSize + destinationIndex] = Float(frame.bytes[sourceIndex + 1]) / 255
                pointer[planeSize * 2 + destinationIndex] = Float(frame.bytes[sourceIndex]) / 255
            }
        }

        return try MLDictionaryFeatureProvider(dictionary: [
            Self.inputFeatureName: MLFeatureValue(multiArray: array)
        ])
    }

    func makeInputFeatureProvider(from pixelBuffer: CVPixelBuffer) throws -> MLFeatureProvider {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA else {
            throw NativeCoreMLDetectorError.invalidFrameData
        }
        let frameWidth = CVPixelBufferGetWidth(pixelBuffer)
        let frameHeight = CVPixelBufferGetHeight(pixelBuffer)
        guard frameWidth > 0, frameHeight > 0 else {
            throw NativeCoreMLDetectorError.invalidFrameData
        }

        let inputSize = postprocessor.modelInputSize
        guard let layout = NativeYOLOInputLayout.aspectFit(
            modelInputSize: inputSize,
            frameWidth: frameWidth,
            frameHeight: frameHeight
        ) else {
            throw NativeCoreMLDetectorError.invalidFrameData
        }
        let array = try MLMultiArray(
            shape: [1, 3, NSNumber(value: inputSize), NSNumber(value: inputSize)],
            dataType: .float32
        )
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        let planeSize = inputSize * inputSize
        let letterboxValue = Float(114) / 255

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw NativePixelBufferBridgeError.baseAddressUnavailable
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let source = baseAddress.assumingMemoryBound(to: UInt8.self)

        for y in 0..<inputSize {
            for x in 0..<inputSize {
                let destinationIndex = y * inputSize + x
                let unpaddedX = Float(x) - layout.padX
                let unpaddedY = Float(y) - layout.padY
                guard unpaddedX >= 0,
                      unpaddedY >= 0,
                      unpaddedX < Float(layout.scaledWidth),
                      unpaddedY < Float(layout.scaledHeight)
                else {
                    pointer[destinationIndex] = letterboxValue
                    pointer[planeSize + destinationIndex] = letterboxValue
                    pointer[planeSize * 2 + destinationIndex] = letterboxValue
                    continue
                }

                let sourceX = min(
                    frameWidth - 1,
                    Int((unpaddedX / layout.scale).rounded(.down))
                )
                let sourceY = min(
                    frameHeight - 1,
                    Int((unpaddedY / layout.scale).rounded(.down))
                )
                let sourceIndex = sourceY * bytesPerRow + sourceX * 4
                pointer[destinationIndex] = Float(source[sourceIndex + 2]) / 255
                pointer[planeSize + destinationIndex] = Float(source[sourceIndex + 1]) / 255
                pointer[planeSize * 2 + destinationIndex] = Float(source[sourceIndex]) / 255
            }
        }

        return try MLDictionaryFeatureProvider(dictionary: [
            Self.inputFeatureName: MLFeatureValue(multiArray: array)
        ])
    }

    func makeImageInputFeatureProvider(from frame: NativeBGRAFrame) throws -> MLFeatureProvider {
        let pixelBuffer = try makeLetterboxedPixelBuffer(from: frame)
        return try MLDictionaryFeatureProvider(dictionary: [
            Self.imageInputFeatureName: MLFeatureValue(pixelBuffer: pixelBuffer)
        ])
    }

    func makeImageInputFeatureProvider(from pixelBuffer: CVPixelBuffer) throws -> MLFeatureProvider {
        let pixelBuffer = try makeLetterboxedPixelBuffer(from: pixelBuffer)
        return try MLDictionaryFeatureProvider(dictionary: [
            Self.imageInputFeatureName: MLFeatureValue(pixelBuffer: pixelBuffer)
        ])
    }

    private func makeLetterboxedPixelBuffer(from frame: NativeBGRAFrame) throws -> CVPixelBuffer {
        guard frame.width > 0,
              frame.height > 0,
              frame.bytes.count == frame.width * frame.height * 4
        else {
            throw NativeCoreMLDetectorError.invalidFrameData
        }

        let inputSize = postprocessor.modelInputSize
        guard let layout = NativeYOLOInputLayout.aspectFit(
            modelInputSize: inputSize,
            frameWidth: frame.width,
            frameHeight: frame.height
        ) else {
            throw NativeCoreMLDetectorError.invalidFrameData
        }

        var maybePixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            inputSize,
            inputSize,
            kCVPixelFormatType_32BGRA,
            [
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true
            ] as CFDictionary,
            &maybePixelBuffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer = maybePixelBuffer else {
            throw NativePixelBufferBridgeError.pixelBufferCreationFailed(status)
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw NativePixelBufferBridgeError.baseAddressUnavailable
        }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let destination = baseAddress.assumingMemoryBound(to: UInt8.self)
        for y in 0..<inputSize {
            for x in 0..<inputSize {
                let destinationIndex = y * bytesPerRow + x * 4
                let unpaddedX = Float(x) - layout.padX
                let unpaddedY = Float(y) - layout.padY
                guard unpaddedX >= 0,
                      unpaddedY >= 0,
                      unpaddedX < Float(layout.scaledWidth),
                      unpaddedY < Float(layout.scaledHeight)
                else {
                    destination[destinationIndex] = 114
                    destination[destinationIndex + 1] = 114
                    destination[destinationIndex + 2] = 114
                    destination[destinationIndex + 3] = 255
                    continue
                }

                let sourceX = min(
                    frame.width - 1,
                    Int((unpaddedX / layout.scale).rounded(.down))
                )
                let sourceY = min(
                    frame.height - 1,
                    Int((unpaddedY / layout.scale).rounded(.down))
                )
                let sourceIndex = (sourceY * frame.width + sourceX) * 4
                destination[destinationIndex] = frame.bytes[sourceIndex]
                destination[destinationIndex + 1] = frame.bytes[sourceIndex + 1]
                destination[destinationIndex + 2] = frame.bytes[sourceIndex + 2]
                destination[destinationIndex + 3] = 255
            }
        }

        return pixelBuffer
    }

    private func makeLetterboxedPixelBuffer(from sourcePixelBuffer: CVPixelBuffer) throws -> CVPixelBuffer {
        guard CVPixelBufferGetPixelFormatType(sourcePixelBuffer) == kCVPixelFormatType_32BGRA else {
            throw NativeCoreMLDetectorError.invalidFrameData
        }
        let frameWidth = CVPixelBufferGetWidth(sourcePixelBuffer)
        let frameHeight = CVPixelBufferGetHeight(sourcePixelBuffer)
        guard frameWidth > 0, frameHeight > 0 else {
            throw NativeCoreMLDetectorError.invalidFrameData
        }

        let inputSize = postprocessor.modelInputSize
        guard let layout = NativeYOLOInputLayout.aspectFit(
            modelInputSize: inputSize,
            frameWidth: frameWidth,
            frameHeight: frameHeight
        ) else {
            throw NativeCoreMLDetectorError.invalidFrameData
        }

        var maybePixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            inputSize,
            inputSize,
            kCVPixelFormatType_32BGRA,
            [
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true,
                kCVPixelBufferMetalCompatibilityKey: true,
                kCVPixelBufferIOSurfacePropertiesKey: [:]
            ] as CFDictionary,
            &maybePixelBuffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer = maybePixelBuffer else {
            throw NativePixelBufferBridgeError.pixelBufferCreationFailed(status)
        }

        CVPixelBufferLockBaseAddress(sourcePixelBuffer, .readOnly)
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
            CVPixelBufferUnlockBaseAddress(sourcePixelBuffer, .readOnly)
        }

        guard let sourceBaseAddress = CVPixelBufferGetBaseAddress(sourcePixelBuffer),
              let destinationBaseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
        else {
            throw NativePixelBufferBridgeError.baseAddressUnavailable
        }

        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(sourcePixelBuffer)
        let destinationBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let source = sourceBaseAddress.assumingMemoryBound(to: UInt8.self)
        let destination = destinationBaseAddress.assumingMemoryBound(to: UInt8.self)
        for y in 0..<inputSize {
            for x in 0..<inputSize {
                let destinationIndex = y * destinationBytesPerRow + x * 4
                let unpaddedX = Float(x) - layout.padX
                let unpaddedY = Float(y) - layout.padY
                guard unpaddedX >= 0,
                      unpaddedY >= 0,
                      unpaddedX < Float(layout.scaledWidth),
                      unpaddedY < Float(layout.scaledHeight)
                else {
                    destination[destinationIndex] = 114
                    destination[destinationIndex + 1] = 114
                    destination[destinationIndex + 2] = 114
                    destination[destinationIndex + 3] = 255
                    continue
                }

                let sourceX = min(
                    frameWidth - 1,
                    Int((unpaddedX / layout.scale).rounded(.down))
                )
                let sourceY = min(
                    frameHeight - 1,
                    Int((unpaddedY / layout.scale).rounded(.down))
                )
                let sourceIndex = sourceY * sourceBytesPerRow + sourceX * 4
                destination[destinationIndex] = source[sourceIndex]
                destination[destinationIndex + 1] = source[sourceIndex + 1]
                destination[destinationIndex + 2] = source[sourceIndex + 2]
                destination[destinationIndex + 3] = 255
            }
        }

        return pixelBuffer
    }

    func detections(
        from featureProvider: MLFeatureProvider,
        frameWidth: Int,
        frameHeight: Int
    ) throws -> [NativeMosaicDetection] {
        let output0Name = [
            Self.outputFeatureName,
            Self.ultralyticsOutputFeatureName
        ].first {
            featureProvider.featureValue(for: $0)?.multiArrayValue != nil
        }
        let output1Name = [
            Self.prototypeFeatureName,
            Self.ultralyticsPrototypeFeatureName
        ].first {
            featureProvider.featureValue(for: $0)?.multiArrayValue != nil
        }

        guard let output0Name,
              let output0 = featureProvider.featureValue(for: output0Name)?.multiArrayValue
        else {
            throw NativeCoreMLDetectorError.missingOutput(Self.outputFeatureName)
        }
        guard let output1Name,
              let output1 = featureProvider.featureValue(for: output1Name)?.multiArrayValue
        else {
            throw NativeCoreMLDetectorError.missingOutput(Self.prototypeFeatureName)
        }

        return postprocessor.detections(
            output0: floats(from: output0),
            output1: floats(from: output1),
            frameWidth: frameWidth,
            frameHeight: frameHeight
        )
    }

    private func floats(from multiArray: MLMultiArray) -> [Float] {
        guard multiArray.dataType == .float32 else {
            return (0..<multiArray.count).map { Float(truncating: multiArray[$0]) }
        }

        let pointer = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
        return Array(
            UnsafeBufferPointer(
                start: pointer,
                count: multiArray.count
            )
        )
    }
}

enum NativeCoreMLDetectorError: LocalizedError {
    case invalidFrameData
    case missingOutput(String)

    var errorDescription: String? {
        switch self {
        case .invalidFrameData:
            "The Core ML detector received frame bytes that do not match the frame size."
        case .missingOutput(let name):
            "The Core ML detector did not receive the expected model output named \(name)."
        }
    }
}

final class NativeCoreMLModelCache: @unchecked Sendable {
    private let lock = NSLock()
    private var cachedURL: URL?
    private var cachedModel: MLModel?

    func model(
        at url: URL,
        configuration: MLModelConfiguration
    ) throws -> MLModel {
        lock.lock()
        defer { lock.unlock() }

        if cachedURL == url, let cachedModel {
            return cachedModel
        }

        let model = try MLModel(
            contentsOf: url,
            configuration: configuration
        )
        cachedURL = url
        cachedModel = model
        return model
    }
}
