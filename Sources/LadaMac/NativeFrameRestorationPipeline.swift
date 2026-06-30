import CoreVideo
import Foundation
import Metal

struct NativeRestorationRegion: Sendable {
    let x: Int
    let y: Int
    let width: Int
    let height: Int
}

protocol NativeRestorationRegionProvider: Sendable {
    func regions(for frame: NativeBGRAFrame) throws -> [NativeRestorationRegion]
}

protocol NativeGeometryRestorationRegionProvider: NativeRestorationRegionProvider {
    func regions(frameWidth: Int, frameHeight: Int) throws -> [NativeRestorationRegion]
}

protocol NativeRegionRestorer: Sendable {
    func restore(
        modelInput: [UInt8],
        width: Int,
        height: Int
    ) throws -> [UInt8]
}

struct NativeRestorationClip: Sendable {
    let frames: [NativeBGRAFrame]
    let frameRate: Double?

    init(
        frames: [NativeBGRAFrame],
        frameRate: Double? = nil
    ) {
        self.frames = frames
        self.frameRate = frameRate
    }
}

protocol NativeTemporalRegionRestorer: Sendable {
    func restore(clip: NativeRestorationClip) throws -> NativeRestorationClip
}

struct NativeSingleFrameRestorerAdapter: NativeTemporalRegionRestorer {
    let restorer: any NativeRegionRestorer

    init(restorer: any NativeRegionRestorer) {
        self.restorer = restorer
    }

    func restore(clip: NativeRestorationClip) throws -> NativeRestorationClip {
        let restoredFrames = try clip.frames.map { frame in
            let restoredBytes = try restorer.restore(
                modelInput: frame.bytes,
                width: frame.width,
                height: frame.height
            )
            return NativeBGRAFrame(
                width: frame.width,
                height: frame.height,
                bytes: restoredBytes
            )
        }
        return NativeRestorationClip(
            frames: restoredFrames,
            frameRate: clip.frameRate
        )
    }
}

struct PlaceholderNativeRegionRestorer: NativeRegionRestorer {
    func restore(
        modelInput: [UInt8],
        width: Int,
        height: Int
    ) throws -> [UInt8] {
        var output = modelInput
        for index in stride(from: 0, to: output.count, by: 4) {
            output[index] = UInt8(min(Int(output[index]) + 40, 255))
            output[index + 1] = UInt8(min(Int(output[index + 1]) + 40, 255))
            output[index + 2] = UInt8(min(Int(output[index + 2]) + 40, 255))
        }
        return output
    }
}

struct NativeCoreMLRestorerAvailability: Sendable {
    let modelURL: URL?

    var isAvailable: Bool {
        guard let modelURL else {
            return false
        }
        return FileManager.default.fileExists(atPath: modelURL.path) &&
            modelURL.pathExtension == "mlmodelc"
    }
}

struct NativeCoreMLRegionRestorer: NativeRegionRestorer {
    static let defaultModelName = NativeModelBundleCatalog.restorerModelName

    let modelURL: URL?
    let fallback: any NativeRegionRestorer

    init(
        modelURL: URL? = NativeModelBundleCatalog.modelURL(named: Self.defaultModelName),
        fallback: any NativeRegionRestorer = PlaceholderNativeRegionRestorer()
    ) {
        self.modelURL = modelURL
        self.fallback = fallback
    }

    var availability: NativeCoreMLRestorerAvailability {
        NativeCoreMLRestorerAvailability(modelURL: modelURL)
    }

    func restore(
        modelInput: [UInt8],
        width: Int,
        height: Int
    ) throws -> [UInt8] {
        guard availability.isAvailable else {
            return try fallback.restore(
                modelInput: modelInput,
                width: width,
                height: height
            )
        }

        // The real BasicVSR++ replacement is temporal and will need a sequence
        // model interface. Until `LadaMosaicRestorer.mlmodelc` exists and its
        // contract is known, keep the native engine safe by falling back to the
        // placeholder restorer instead of pretending single-frame inference is
        // equivalent.
        return try fallback.restore(
            modelInput: modelInput,
            width: width,
            height: height
        )
    }
}

struct NativeCoreMLTemporalRestorer: NativeTemporalRegionRestorer {
    static let defaultModelName = NativeModelBundleCatalog.restorerModelName

    let modelURL: URL?
    let fallback: any NativeTemporalRegionRestorer

    init(
        modelURL: URL? = NativeModelBundleCatalog.modelURL(named: Self.defaultModelName),
        fallback: any NativeTemporalRegionRestorer = NativeSingleFrameRestorerAdapter(
            restorer: PlaceholderNativeRegionRestorer()
        )
    ) {
        self.modelURL = modelURL
        self.fallback = fallback
    }

    var availability: NativeCoreMLRestorerAvailability {
        NativeCoreMLRestorerAvailability(modelURL: modelURL)
    }

    func restore(clip: NativeRestorationClip) throws -> NativeRestorationClip {
        guard availability.isAvailable else {
            return try fallback.restore(clip: clip)
        }

        // BasicVSR++ is a temporal model: real inference should consume and
        // produce a clip, not unrelated single frames. Keep this scaffold
        // conservative until the compiled Core ML model contract is known.
        return try fallback.restore(clip: clip)
    }
}

struct CenterNativeRestorationRegionProvider: NativeGeometryRestorationRegionProvider {
    func regions(for frame: NativeBGRAFrame) throws -> [NativeRestorationRegion] {
        try regions(frameWidth: frame.width, frameHeight: frame.height)
    }

    func regions(frameWidth: Int, frameHeight: Int) throws -> [NativeRestorationRegion] {
        let regionWidth = max(2, frameWidth / 2)
        let regionHeight = max(2, frameHeight / 2)
        return [
            NativeRestorationRegion(
                x: max(0, (frameWidth - regionWidth) / 2),
                y: max(0, (frameHeight - regionHeight) / 2),
                width: min(regionWidth, frameWidth),
                height: min(regionHeight, frameHeight)
            )
        ]
    }
}

struct FixedNativeRestorationRegionProvider: NativeGeometryRestorationRegionProvider {
    let regions: [NativeRestorationRegion]

    func regions(for frame: NativeBGRAFrame) throws -> [NativeRestorationRegion] {
        try regions(frameWidth: frame.width, frameHeight: frame.height)
    }

    func regions(frameWidth: Int, frameHeight: Int) throws -> [NativeRestorationRegion] {
        regions.filter { region in
            region.x >= 0 &&
            region.y >= 0 &&
            region.width > 0 &&
            region.height > 0 &&
            region.x + region.width <= frameWidth &&
            region.y + region.height <= frameHeight
        }
    }
}

final class NativeFrameRestorationPipeline: @unchecked Sendable {
    private let imageProcessor: NativeMetalImageProcessor
    private let regionProvider: any NativeRestorationRegionProvider
    private let restorer: any NativeRegionRestorer
    private let modelInputSize: Int
    private let blendStrength: Float

    init(
        imageProcessor: NativeMetalImageProcessor,
        regionProvider: any NativeRestorationRegionProvider = CenterNativeRestorationRegionProvider(),
        restorer: any NativeRegionRestorer = PlaceholderNativeRegionRestorer(),
        modelInputSize: Int = 32,
        blendStrength: Float = 0.25
    ) {
        self.imageProcessor = imageProcessor
        self.regionProvider = regionProvider
        self.restorer = restorer
        self.modelInputSize = modelInputSize
        self.blendStrength = blendStrength
    }

    func process(
        frame: NativeBGRAFrame,
        region: NativeRestorationRegion? = nil
    ) throws -> NativeBGRAFrame {
        let regions = if let region {
            [region]
        } else {
            try regionProvider.regions(for: frame)
        }
        var output = frame
        for target in regions {
            output = try process(frame: output, target: target)
        }
        return output
    }

    func process(pixelBuffer: CVPixelBuffer) throws -> CVPixelBuffer {
        guard let geometryProvider = regionProvider as? any NativeGeometryRestorationRegionProvider else {
            let sourceFrame = try NativePixelBufferBridge.copyBGRAFrame(from: pixelBuffer)
            let processedFrame = try process(frame: sourceFrame)
            return try NativePixelBufferBridge.makePixelBuffer(from: processedFrame)
        }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let regions = try geometryProvider.regions(frameWidth: width, frameHeight: height)
        guard !regions.isEmpty else {
            return pixelBuffer
        }

        let wrappedSource = try imageProcessor.pixelBufferBridge.texture(from: pixelBuffer)
        var outputTexture = wrappedSource.texture
        for target in regions {
            outputTexture = try process(
                sourceTexture: outputTexture,
                target: target
            )
        }
        let outputBytes = try imageProcessor.readBGRABytes(from: outputTexture)
        return try NativePixelBufferBridge.makePixelBuffer(
            from: NativeBGRAFrame(width: width, height: height, bytes: outputBytes)
        )
    }

    private func process(
        frame: NativeBGRAFrame,
        target: NativeRestorationRegion
    ) throws -> NativeBGRAFrame {
        let sourceTexture = try imageProcessor.makeTexture(
            fromBGRABytes: frame.bytes,
            width: frame.width,
            height: frame.height
        )
        let modelInputCommandBuffer = try imageProcessor.makeCommandBuffer()
        let cropped = try imageProcessor.crop(
            sourceTexture,
            region: target,
            commandBuffer: modelInputCommandBuffer
        )
        let modelInputTexture = try imageProcessor.resize(
            cropped,
            toWidth: modelInputSize,
            height: modelInputSize,
            commandBuffer: modelInputCommandBuffer
        )
        try imageProcessor.submitAndWait(modelInputCommandBuffer)

        let modelInput = try imageProcessor.readBGRABytes(from: modelInputTexture)
        let restoredModelOutput = try restorer.restore(
            modelInput: modelInput,
            width: modelInputSize,
            height: modelInputSize
        )
        let restoredTexture = try imageProcessor.makeTexture(
            fromBGRABytes: restoredModelOutput,
            width: modelInputSize,
            height: modelInputSize
        )
        let compositeCommandBuffer = try imageProcessor.makeCommandBuffer()
        let restoredRegion = try imageProcessor.resize(
            restoredTexture,
            toWidth: target.width,
            height: target.height,
            commandBuffer: compositeCommandBuffer
        )
        let compositedTexture = try imageProcessor.compositeRegion(
            source: sourceTexture,
            restoredRegion: restoredRegion,
            origin: (target.x, target.y),
            alpha: blendStrength,
            commandBuffer: compositeCommandBuffer
        )
        try imageProcessor.submitAndWait(compositeCommandBuffer)

        let composited = try imageProcessor.readBGRABytes(from: compositedTexture)
        return NativeBGRAFrame(width: frame.width, height: frame.height, bytes: composited)
    }

    private func process(
        sourceTexture: MTLTexture,
        target: NativeRestorationRegion
    ) throws -> MTLTexture {
        let modelInputCommandBuffer = try imageProcessor.makeCommandBuffer()
        let cropped = try imageProcessor.crop(
            sourceTexture,
            region: target,
            commandBuffer: modelInputCommandBuffer
        )
        let modelInputTexture = try imageProcessor.resize(
            cropped,
            toWidth: modelInputSize,
            height: modelInputSize,
            commandBuffer: modelInputCommandBuffer
        )
        try imageProcessor.submitAndWait(modelInputCommandBuffer)

        let modelInput = try imageProcessor.readBGRABytes(from: modelInputTexture)
        let restoredModelOutput = try restorer.restore(
            modelInput: modelInput,
            width: modelInputSize,
            height: modelInputSize
        )
        let restoredTexture = try imageProcessor.makeTexture(
            fromBGRABytes: restoredModelOutput,
            width: modelInputSize,
            height: modelInputSize
        )
        let compositeCommandBuffer = try imageProcessor.makeCommandBuffer()
        let restoredRegion = try imageProcessor.resize(
            restoredTexture,
            toWidth: target.width,
            height: target.height,
            commandBuffer: compositeCommandBuffer
        )
        let compositedTexture = try imageProcessor.compositeRegion(
            source: sourceTexture,
            restoredRegion: restoredRegion,
            origin: (target.x, target.y),
            alpha: blendStrength,
            commandBuffer: compositeCommandBuffer
        )
        try imageProcessor.submitAndWait(compositeCommandBuffer)
        return compositedTexture
    }
}
