import Foundation

struct NativeRestorationRegion: Sendable {
    let x: Int
    let y: Int
    let width: Int
    let height: Int
}

protocol NativeRestorationRegionProvider: Sendable {
    func regions(for frame: NativeBGRAFrame) throws -> [NativeRestorationRegion]
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

struct CenterNativeRestorationRegionProvider: NativeRestorationRegionProvider {
    func regions(for frame: NativeBGRAFrame) throws -> [NativeRestorationRegion] {
        let regionWidth = max(2, frame.width / 2)
        let regionHeight = max(2, frame.height / 2)
        return [
            NativeRestorationRegion(
                x: max(0, (frame.width - regionWidth) / 2),
                y: max(0, (frame.height - regionHeight) / 2),
                width: min(regionWidth, frame.width),
                height: min(regionHeight, frame.height)
            )
        ]
    }
}

struct FixedNativeRestorationRegionProvider: NativeRestorationRegionProvider {
    let regions: [NativeRestorationRegion]

    func regions(for frame: NativeBGRAFrame) throws -> [NativeRestorationRegion] {
        regions.filter { region in
            region.x >= 0 &&
            region.y >= 0 &&
            region.width > 0 &&
            region.height > 0 &&
            region.x + region.width <= frame.width &&
            region.y + region.height <= frame.height
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

    private func process(
        frame: NativeBGRAFrame,
        target: NativeRestorationRegion
    ) throws -> NativeBGRAFrame {
        let cropped = try imageProcessor.cropBGRA(
            source: frame.bytes,
            width: frame.width,
            height: frame.height,
            x: target.x,
            y: target.y,
            cropWidth: target.width,
            cropHeight: target.height
        )
        let modelInput = try imageProcessor.resizeBGRANearest(
            source: cropped,
            width: target.width,
            height: target.height,
            outputWidth: modelInputSize,
            outputHeight: modelInputSize
        )
        let restoredModelOutput = try restorer.restore(
            modelInput: modelInput,
            width: modelInputSize,
            height: modelInputSize
        )
        let restoredRegion = try imageProcessor.resizeBGRANearest(
            source: restoredModelOutput,
            width: modelInputSize,
            height: modelInputSize,
            outputWidth: target.width,
            outputHeight: target.height
        )
        let mask = [Float](
            repeating: min(max(blendStrength, 0), 1),
            count: target.width * target.height
        )
        let composited = try imageProcessor.compositeBGRARegion(
            source: frame.bytes,
            restoredRegion: restoredRegion,
            mask: mask,
            width: frame.width,
            height: frame.height,
            x: target.x,
            y: target.y,
            regionWidth: target.width,
            regionHeight: target.height
        )
        return NativeBGRAFrame(width: frame.width, height: frame.height, bytes: composited)
    }
}
