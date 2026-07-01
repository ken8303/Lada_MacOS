import Foundation
import AVFoundation
import CoreML
import CoreVideo
import Testing
@testable import LadaMac

@Test
func outputFormatsMapToAppleHardwarePresets() {
    #expect(OutputFormat.hevc.encodingPreset.contains("apple"))
    #expect(OutputFormat.h264.encodingPreset.contains("apple"))
}

@Test
func qualityPresetsMapToAppleEncoderPresets() {
    #expect(OutputFormat.hevc.encodingPreset(quality: .balanced) == "hevc-apple-gpu-fast")
    #expect(OutputFormat.hevc.encodingPreset(quality: .high) == "hevc-apple-gpu-balanced")
    #expect(OutputFormat.hevc.encodingPreset(quality: .maximum) == "hevc-apple-gpu-hq")
}

@Test
func restorationEngineSelectionDefaultsToProductionPythonPath() {
    let mode = RestorationEngineSelection.mode(
        environment: [:],
        arguments: ["LadaMac"]
    )

    #expect(mode == .python)
    #expect(mode.title == "Python Lada")
}

@Test
func restorationEngineSelectionCanUseNativeModesForExplicitTesting() {
    #expect(RestorationEngineSelection.mode(
        environment: ["LADA_ENGINE": "native-metal"],
        arguments: ["LadaMac"]
    ) == .nativeMetal)

    #expect(RestorationEngineSelection.mode(
        environment: ["LADA_ENGINE": "python"],
        arguments: ["LadaMac", "--engine=native-coreai"]
    ) == .nativeCoreAI)
}

@Test
func progressStabilizerIgnoresWorkerProgressResetAndHugeETA() {
    let start = Date(timeIntervalSince1970: 0)
    var stabilizer = RestorationProgressStabilizer(startedAt: start)

    let first = stabilizer.update(
        rawProgress: 0.02,
        rawRemaining: 10_200,
        now: start.addingTimeInterval(120)
    )
    let second = stabilizer.update(
        rawProgress: 0.06,
        rawRemaining: 36_000,
        now: start.addingTimeInterval(600)
    )
    let reset = stabilizer.update(
        rawProgress: 0,
        rawRemaining: 252_000,
        now: start.addingTimeInterval(899)
    )

    #expect(first.progress == 0.02)
    #expect(second.progress == 0.06)
    #expect(reset.progress == 0.06)
    #expect(reset.estimatedSecondsRemaining ?? 0 < 15_000)
}

@Test
func progressStabilizerCapsETAWhileProgressIsStalled() {
    let start = Date(timeIntervalSince1970: 0)
    var stabilizer = RestorationProgressStabilizer(startedAt: start)

    let moving = stabilizer.update(
        rawProgress: 0.06,
        rawRemaining: nil,
        now: start.addingTimeInterval(600)
    )
    let stalled = stabilizer.update(
        rawProgress: 0.06,
        rawRemaining: nil,
        now: start.addingTimeInterval(720)
    )

    #expect(stalled.progress == moving.progress)
    #expect(moving.estimatedSecondsRemaining != nil)
    #expect(stalled.estimatedSecondsRemaining != nil)
}

@Test
func progressStabilizerHidesETAWhenProgressIsStale() {
    let start = Date(timeIntervalSince1970: 0)
    var stabilizer = RestorationProgressStabilizer(startedAt: start)

    let moving = stabilizer.update(
        rawProgress: 0.079227,
        rawRemaining: 41_307,
        now: start.addingTimeInterval(2_115)
    )
    let stale = stabilizer.update(
        rawProgress: 0.020214,
        rawRemaining: 269_516,
        now: start.addingTimeInterval(7_718)
    )

    #expect(moving.estimatedSecondsRemaining != nil)
    #expect(stale.progress == moving.progress)
    #expect(stale.estimatedSecondsRemaining == nil)
}

@Test
func progressStabilizerShowsCredibleHigherETAWhenCurrentSpeedIsSlower() {
    let start = Date(timeIntervalSince1970: 0)
    var stabilizer = RestorationProgressStabilizer(startedAt: start)

    let earlier = stabilizer.update(
        rawProgress: 0.042775,
        rawRemaining: 11_052,
        now: start.addingTimeInterval(612)
    )
    let later = stabilizer.update(
        rawProgress: 0.089004,
        rawRemaining: 40_392,
        now: start.addingTimeInterval(2_718)
    )

    #expect(later.progress > earlier.progress)
    #expect((later.estimatedSecondsRemaining ?? 0) >= 40_392)
}

@Test
func progressStabilizerDoesNotShowOverlyOptimisticETAFromOldBestSpeed() {
    let start = Date(timeIntervalSince1970: 0)
    var stabilizer = RestorationProgressStabilizer(startedAt: start)

    _ = stabilizer.update(
        rawProgress: 0.02,
        rawRemaining: 11_198,
        now: start.addingTimeInterval(300)
    )
    let current = stabilizer.update(
        rawProgress: 0.07649,
        rawRemaining: 41_925,
        now: start.addingTimeInterval(2_150)
    )

    #expect(current.progress == 0.07649)
    #expect((current.estimatedSecondsRemaining ?? 0) >= 41_925)
}

@Test
func progressStabilizerIgnoresImplausiblyHugeRawETA() {
    let start = Date(timeIntervalSince1970: 0)
    var stabilizer = RestorationProgressStabilizer(startedAt: start)

    let update = stabilizer.update(
        rawProgress: 0.000004,
        rawRemaining: 123_948,
        now: start.addingTimeInterval(12)
    )

    #expect(update.estimatedSecondsRemaining == nil)
}

@Test
func progressUpdateThrottlerReducesHighFrequencyUpdates() {
    let start = Date(timeIntervalSince1970: 0)
    let throttler = ProgressUpdateThrottler(
        minimumInterval: 1,
        minimumProgressDelta: 0.01
    )

    #expect(throttler.shouldEmit(progress: 0.01, now: start))
    #expect(!throttler.shouldEmit(progress: 0.011, now: start.addingTimeInterval(0.2)))
    #expect(throttler.shouldEmit(progress: 0.011, now: start.addingTimeInterval(1.1)))
    #expect(throttler.shouldEmit(progress: 0.03, now: start.addingTimeInterval(1.2)))
}

@Test
func progressDebugLoggerWritesJSONLines() throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("LadaProgressDebug-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    let url = directory.appendingPathComponent("job.progress-debug.jsonl")
    let logger = try #require(ProgressDebugLogger(url: url))
    logger.log(ProgressDebugEvent(
        event: "progress",
        jobID: UUID(),
        source: directory.appendingPathComponent("input.mp4"),
        output: directory.appendingPathComponent("output.mp4"),
        rawProgress: 0,
        stableProgress: 0.06,
        rawRemainingSeconds: 252_000,
        stableRemainingSeconds: 14_100,
        elapsedSeconds: 900,
        note: "test"
    ))
    logger.close()

    let contents = try String(contentsOf: url, encoding: .utf8)
    #expect(contents.contains("\"event\":\"progress\""))
    #expect(contents.contains("\"rawProgress\":0"))
    #expect(contents.contains("\"stableProgress\":0.06"))
    #expect(contents.contains("\"rawRemainingSeconds\":252000"))
}

@Test
func restorationProfilesHaveUsefulClipLengths() {
    #expect(RestorationProfile.fast.maxClipLength < RestorationProfile.accurate.maxClipLength)
    #expect(RestorationProfile.standard.detectionModel == "v4-fast")
}

@Test
func restorationProfilesScaleClipLengthByMemoryMode() {
    #expect(RestorationProfile.standard.maxClipLength(memoryMode: .longVideo) == 75)
    #expect(RestorationProfile.standard.maxClipLength(memoryMode: .conservative) == 45)
    #expect(RestorationProfile.standard.maxClipLength(memoryMode: .automatic) == 60)
    #expect(RestorationProfile.standard.maxClipLength(memoryMode: .performance) == 75)
}

@Test
func newJobsDefaultToSpeedOrientedSettings() {
    let job = RestorationJob(
        sourceURL: URL(fileURLWithPath: "/tmp/source.mp4"),
        destinationURL: URL(fileURLWithPath: "/tmp/output.mp4")
    )

    #expect(job.profile == .standard)
    #expect(job.quality == .balanced)
    #expect(job.memoryMode == .longVideo)
    #expect(job.outputFormat.encodingPreset(quality: job.quality) == "hevc-apple-gpu-fast")
    #expect(job.profile.maxClipLength(memoryMode: job.memoryMode) == 75)
}

@Test
func engineStatusNamesMetal4WhenReady() {
    let status = EngineStatus.ready("Metal 4 / MPS on Apple Silicon")
    #expect(status.title == "Metal 4 ready")
    #expect(status.detail.contains("Metal 4"))
}

@Test
func nativeMetalEngineStartsAsScaffold() async {
    let status = await NativeMetalEngine().probe()
    #expect(status.title == "Metal 4 ready")
    #expect(status.detail.contains("Native AVFoundation + Metal image pipeline ready"))
}

@Test
func nativeMetalImageProcessorBlendsBGRAFrames() throws {
    let processor = try NativeMetalImageProcessor()
    let source: [UInt8] = [
        0, 0, 0, 255,
        10, 20, 30, 200,
        100, 100, 100, 255,
        200, 150, 100, 128
    ]
    let restored: [UInt8] = [
        100, 50, 25, 255,
        250, 220, 210, 255,
        20, 30, 40, 255,
        0, 0, 0, 255
    ]
    let mask: [Float] = [0, 0.5, 1, 0.25]

    let output = try processor.blendBGRA(
        source: source,
        restored: restored,
        mask: mask,
        width: 2,
        height: 2
    )

    #expect(output == [
        0, 0, 0, 255,
        130, 120, 120, 200,
        20, 30, 40, 255,
        150, 113, 75, 128
    ])
}

@Test
func nativeMetalImageProcessorResizesBGRAFrames() throws {
    let processor = try NativeMetalImageProcessor()
    let source: [UInt8] = [
        10, 20, 30, 255,
        40, 50, 60, 255,
        70, 80, 90, 255,
        100, 110, 120, 255
    ]

    let output = try processor.resizeBGRANearest(
        source: source,
        width: 2,
        height: 2,
        outputWidth: 4,
        outputHeight: 4
    )

    #expect(output == [
        10, 20, 30, 255, 10, 20, 30, 255, 40, 50, 60, 255, 40, 50, 60, 255,
        10, 20, 30, 255, 10, 20, 30, 255, 40, 50, 60, 255, 40, 50, 60, 255,
        70, 80, 90, 255, 70, 80, 90, 255, 100, 110, 120, 255, 100, 110, 120, 255,
        70, 80, 90, 255, 70, 80, 90, 255, 100, 110, 120, 255, 100, 110, 120, 255
    ])
}

@Test
func nativeMetalImageProcessorCropsBGRAFrames() throws {
    let processor = try NativeMetalImageProcessor()
    let source: [UInt8] = [
        1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255,
        10, 11, 12, 255, 13, 14, 15, 255, 16, 17, 18, 255,
        19, 20, 21, 255, 22, 23, 24, 255, 25, 26, 27, 255
    ]

    let output = try processor.cropBGRA(
        source: source,
        width: 3,
        height: 3,
        x: 1,
        y: 1,
        cropWidth: 2,
        cropHeight: 2
    )

    #expect(output == [
        13, 14, 15, 255, 16, 17, 18, 255,
        22, 23, 24, 255, 25, 26, 27, 255
    ])
}

@Test
func nativeMetalImageProcessorCompositesBGRARegion() throws {
    let processor = try NativeMetalImageProcessor()
    let source: [UInt8] = Array(repeating: [0, 0, 0, 255], count: 9).flatMap { $0 }
    let restoredRegion: [UInt8] = Array(repeating: [100, 80, 60, 255], count: 4).flatMap { $0 }
    let output = try processor.compositeBGRARegion(
        source: source,
        restoredRegion: restoredRegion,
        mask: [1, 0, 0.5, 1],
        width: 3,
        height: 3,
        x: 1,
        y: 1,
        regionWidth: 2,
        regionHeight: 2
    )

    #expect(output == [
        0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255,
        0, 0, 0, 255, 100, 80, 60, 255, 0, 0, 0, 255,
        0, 0, 0, 255, 50, 40, 30, 255, 100, 80, 60, 255
    ])
}

@Test
func nativeFrameRestorationPipelineScaffoldsCropResizeComposite() throws {
    let source: [UInt8] = Array(repeating: [0, 0, 0, 255], count: 64).flatMap { $0 }
    let pipeline = try NativeFrameRestorationPipeline(
        imageProcessor: NativeMetalImageProcessor(),
        regionProvider: FixedNativeRestorationRegionProvider(
            regions: [
                NativeRestorationRegion(x: 2, y: 2, width: 4, height: 4)
            ]
        ),
        modelInputSize: 4,
        blendStrength: 1
    )
    let output = try pipeline.process(
        frame: NativeBGRAFrame(width: 8, height: 8, bytes: source)
    )

    #expect(output.width == 8)
    #expect(output.height == 8)
    for y in 0..<8 {
        for x in 0..<8 {
            let base = (y * 8 + x) * 4
            let insideRegion = x >= 2 && x < 6 && y >= 2 && y < 6
            #expect(output.bytes[base] == (insideRegion ? 40 : 0))
            #expect(output.bytes[base + 1] == (insideRegion ? 40 : 0))
            #expect(output.bytes[base + 2] == (insideRegion ? 40 : 0))
            #expect(output.bytes[base + 3] == 255)
        }
    }
}

@Test
func nativeFrameRestorationPipelineUsesMultipleProvidedRegions() throws {
    let source: [UInt8] = Array(repeating: [0, 0, 0, 255], count: 64).flatMap { $0 }
    let pipeline = try NativeFrameRestorationPipeline(
        imageProcessor: NativeMetalImageProcessor(),
        regionProvider: FixedNativeRestorationRegionProvider(
            regions: [
                NativeRestorationRegion(x: 0, y: 0, width: 2, height: 2),
                NativeRestorationRegion(x: 6, y: 6, width: 2, height: 2)
            ]
        ),
        modelInputSize: 2,
        blendStrength: 1
    )
    let output = try pipeline.process(
        frame: NativeBGRAFrame(width: 8, height: 8, bytes: source)
    )

    for y in 0..<8 {
        for x in 0..<8 {
            let base = (y * 8 + x) * 4
            let insideFirst = x < 2 && y < 2
            let insideSecond = x >= 6 && y >= 6
            #expect(output.bytes[base] == ((insideFirst || insideSecond) ? 40 : 0))
            #expect(output.bytes[base + 1] == ((insideFirst || insideSecond) ? 40 : 0))
            #expect(output.bytes[base + 2] == ((insideFirst || insideSecond) ? 40 : 0))
            #expect(output.bytes[base + 3] == 255)
        }
    }
}

@Test
func nativeFrameRestorationPipelineUsesTextureBackedPixelBufferPathForGeometryRegions() throws {
    let frame = NativeBGRAFrame(
        width: 8,
        height: 8,
        bytes: Array(repeating: [0, 0, 0, 255], count: 64).flatMap { $0 }
    )
    let pixelBuffer = try NativePixelBufferBridge.makePixelBuffer(from: frame)
    let provider = CountingGeometryNativeRestorationRegionProvider(
        regions: [
            NativeRestorationRegion(x: 2, y: 2, width: 4, height: 4)
        ]
    )
    let pipeline = try NativeFrameRestorationPipeline(
        imageProcessor: NativeMetalImageProcessor(),
        regionProvider: provider,
        modelInputSize: 4,
        blendStrength: 1
    )

    let outputPixelBuffer = try pipeline.process(pixelBuffer: pixelBuffer)
    let outputTexture = try NativeMetalImageProcessor().pixelBufferBridge.texture(from: outputPixelBuffer)
    let output = try NativePixelBufferBridge.copyBGRAFrame(from: outputPixelBuffer)

    #expect(provider.geometryCallCount == 1)
    #expect(provider.frameCallCount == 0)
    #expect(outputTexture.texture.width == 8)
    #expect(outputTexture.texture.height == 8)
    #expect(output.width == 8)
    #expect(output.height == 8)
    for y in 0..<8 {
        for x in 0..<8 {
            let base = (y * 8 + x) * 4
            let insideRegion = x >= 2 && x < 6 && y >= 2 && y < 6
            #expect(output.bytes[base] == (insideRegion ? 40 : 0))
            #expect(output.bytes[base + 1] == (insideRegion ? 40 : 0))
            #expect(output.bytes[base + 2] == (insideRegion ? 40 : 0))
            #expect(output.bytes[base + 3] == 255)
        }
    }
}

@Test
func nativeFrameRestorationPipelineUsesInjectedRegionRestorer() throws {
    let source: [UInt8] = Array(repeating: [0, 0, 0, 255], count: 16).flatMap { $0 }
    let restorer = FixedNativeRegionRestorer(
        restoredPixel: [12, 34, 56, 255]
    )
    let pipeline = try NativeFrameRestorationPipeline(
        imageProcessor: NativeMetalImageProcessor(),
        regionProvider: FixedNativeRestorationRegionProvider(
            regions: [
                NativeRestorationRegion(x: 1, y: 1, width: 2, height: 2)
            ]
        ),
        restorer: restorer,
        modelInputSize: 2,
        blendStrength: 1
    )

    let output = try pipeline.process(
        frame: NativeBGRAFrame(width: 4, height: 4, bytes: source)
    )

    #expect(restorer.callCount == 1)
    for y in 0..<4 {
        for x in 0..<4 {
            let base = (y * 4 + x) * 4
            let insideRegion = x >= 1 && x < 3 && y >= 1 && y < 3
            #expect(output.bytes[base] == (insideRegion ? 12 : 0))
            #expect(output.bytes[base + 1] == (insideRegion ? 34 : 0))
            #expect(output.bytes[base + 2] == (insideRegion ? 56 : 0))
            #expect(output.bytes[base + 3] == 255)
        }
    }
}

@Test
func nativeCoreMLRegionRestorerFallsBackWhenModelMissing() throws {
    let missingURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("MissingLadaMosaicRestorer-\(UUID().uuidString).mlmodelc")
    let fallback = FixedNativeRegionRestorer(
        restoredPixel: [5, 6, 7, 255]
    )
    let restorer = NativeCoreMLRegionRestorer(
        modelURL: missingURL,
        fallback: fallback
    )
    let output = try restorer.restore(
        modelInput: Array(repeating: [0, 0, 0, 255], count: 4).flatMap { $0 },
        width: 2,
        height: 2
    )

    #expect(restorer.availability.isAvailable == false)
    #expect(fallback.callCount == 1)
    #expect(output == Array(repeating: [5, 6, 7, 255], count: 4).flatMap { $0 })
}

@Test
func nativeSingleFrameRestorerAdapterPreservesTemporalClipShape() throws {
    let restorer = FixedNativeRegionRestorer(
        restoredPixel: [9, 8, 7, 255]
    )
    let adapter = NativeSingleFrameRestorerAdapter(restorer: restorer)
    let clip = NativeRestorationClip(
        frames: [
            NativeBGRAFrame(
                width: 2,
                height: 2,
                bytes: Array(repeating: [1, 2, 3, 255], count: 4).flatMap { $0 }
            ),
            NativeBGRAFrame(
                width: 2,
                height: 2,
                bytes: Array(repeating: [4, 5, 6, 255], count: 4).flatMap { $0 }
            )
        ],
        frameRate: 30
    )

    let output = try adapter.restore(clip: clip)

    #expect(restorer.callCount == 2)
    #expect(output.frames.count == 2)
    #expect(output.frameRate == 30)
    #expect(output.frames.allSatisfy { $0.width == 2 && $0.height == 2 })
    #expect(output.frames.allSatisfy {
        $0.bytes == Array(repeating: [9, 8, 7, 255], count: 4).flatMap { $0 }
    })
}

@Test
func nativeCoreMLTemporalRestorerFallsBackWhenModelMissing() throws {
    let missingURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("MissingLadaMosaicTemporalRestorer-\(UUID().uuidString).mlmodelc")
    let fallback = FixedNativeTemporalRestorer(
        replacementPixel: [20, 30, 40, 255]
    )
    let restorer = NativeCoreMLTemporalRestorer(
        modelURL: missingURL,
        fallback: fallback
    )
    let clip = NativeRestorationClip(
        frames: [
            NativeBGRAFrame(
                width: 1,
                height: 1,
                bytes: [0, 0, 0, 255]
            )
        ],
        frameRate: 24
    )

    let output = try restorer.restore(clip: clip)

    #expect(restorer.availability.isAvailable == false)
    #expect(fallback.callCount == 1)
    #expect(output.frameRate == 24)
    #expect(output.frames.first?.bytes == [20, 30, 40, 255])
}

@Test
func nativeDetectorRegionProviderFiltersAndClampsModelDetections() throws {
    let frame = NativeBGRAFrame(
        width: 8,
        height: 8,
        bytes: Array(repeating: [0, 0, 0, 255], count: 64).flatMap { $0 }
    )
    let detector = FixedNativeMosaicDetector(
        detections: [
            NativeMosaicDetection(
                confidence: 0.9,
                boundingBox: NativeRestorationRegion(x: -2, y: 1, width: 5, height: 4),
                mask: NativeDetectionMaskMetadata(
                    width: 16,
                    height: 16,
                    coordinateSpace: .modelInput
                )
            ),
            NativeMosaicDetection(
                confidence: 0.1,
                boundingBox: NativeRestorationRegion(x: 4, y: 4, width: 2, height: 2),
                mask: nil
            ),
            NativeMosaicDetection(
                confidence: 0.8,
                boundingBox: NativeRestorationRegion(x: 20, y: 20, width: 2, height: 2),
                mask: NativeDetectionMaskMetadata(
                    width: 8,
                    height: 8,
                    coordinateSpace: .sourceFrame
                )
            )
        ]
    )
    let provider = NativeDetectorRegionProvider(
        detector: detector,
        minimumConfidence: 0.5
    )

    let regions = try provider.regions(for: frame)

    #expect(regions.count == 1)
    #expect(regions.first?.x == 0)
    #expect(regions.first?.y == 1)
    #expect(regions.first?.width == 3)
    #expect(regions.first?.height == 4)
    #expect(detector.detections.first?.mask?.width == 16)
    #expect(detector.detections.first?.mask?.height == 16)
    #expect(detector.detections.first?.mask?.coordinateSpace == .modelInput)
}

@Test
func nativeDetectorReferenceFixtureLoadsPythonYOLOBaseline() throws {
    let fixtureURL = try projectRootURL()
        .appendingPathComponent("native-models/reference-detections/smoke-input-yolo-reference.json")
    let fixture = try NativeDetectorReferenceFixture.load(from: fixtureURL)

    #expect(fixture.schemaVersion == 1)
    #expect(fixture.imgsz == 640)
    #expect(fixture.stride == 32)
    #expect(fixture.conf == 0.15)
    #expect(fixture.iou == 0.7)
    #expect(fixture.letterbox.kind == "ultralytics")
    #expect(fixture.letterbox.auto)
    #expect(fixture.frames.count == 3)
    #expect(fixture.frames.allSatisfy { $0.width == 320 && $0.height == 240 })
    #expect(fixture.frames.allSatisfy { $0.detections.count == 1 })
}

@Test
func nativeDetectorReferenceFixtureMapsBoxesToNativeDetections() throws {
    let fixtureURL = try projectRootURL()
        .appendingPathComponent("native-models/reference-detections/smoke-input-yolo-reference.json")
    let fixture = try NativeDetectorReferenceFixture.load(from: fixtureURL)
    let firstDetection = try #require(fixture.frames.first?.nativeDetections.first)

    #expect(firstDetection.confidence > 0.87)
    #expect(firstDetection.boundingBox.x == 3)
    #expect(firstDetection.boundingBox.y == 0)
    #expect(firstDetection.boundingBox.width == 315)
    #expect(firstDetection.boundingBox.height == 240)
    #expect(firstDetection.mask?.width == 640)
    #expect(firstDetection.mask?.height == 480)
    #expect(firstDetection.mask?.coordinateSpace == .modelInput)
}

@Test
func nativeDetectorReferenceComparatorPassesMatchingFixtureDetections() throws {
    let fixtureURL = try projectRootURL()
        .appendingPathComponent("native-models/reference-detections/smoke-input-yolo-reference.json")
    let fixture = try NativeDetectorReferenceFixture.load(from: fixtureURL)
    let reference = try #require(fixture.frames.first?.nativeDetections)
    let comparison = NativeDetectorReferenceComparator(
        minimumIoU: 0.95,
        maximumConfidenceDelta: 0.01
    ).compare(
        reference: reference,
        candidate: reference
    )

    #expect(comparison.passed)
    #expect(comparison.referenceCount == 1)
    #expect(comparison.candidateCount == 1)
    #expect(comparison.matches.count == 1)
    #expect(comparison.minimumIoU == 1)
    #expect(comparison.maximumConfidenceDelta == 0)
}

@Test
func nativeDetectorReferenceComparatorReportsMissingAndExtraDetections() {
    let reference = [
        NativeMosaicDetection(
            confidence: 0.9,
            boundingBox: NativeRestorationRegion(x: 10, y: 10, width: 20, height: 20),
            mask: nil
        )
    ]
    let extraCandidate = NativeMosaicDetection(
        confidence: 0.9,
        boundingBox: NativeRestorationRegion(x: 200, y: 200, width: 20, height: 20),
        mask: nil
    )

    let comparison = NativeDetectorReferenceComparator(
        minimumIoU: 0.5,
        maximumConfidenceDelta: 0.1
    ).compare(
        reference: reference,
        candidate: [extraCandidate]
    )

    #expect(comparison.passed == false)
    #expect(comparison.matches.isEmpty)
    #expect(comparison.missingReferenceCount == 1)
    #expect(comparison.extraCandidateCount == 1)
}

@Test
func nativeDetectorReferenceComparatorRejectsConfidenceDrift() {
    let reference = NativeMosaicDetection(
        confidence: 0.9,
        boundingBox: NativeRestorationRegion(x: 10, y: 10, width: 20, height: 20),
        mask: nil
    )
    let candidate = NativeMosaicDetection(
        confidence: 0.4,
        boundingBox: NativeRestorationRegion(x: 10, y: 10, width: 20, height: 20),
        mask: nil
    )

    let comparison = NativeDetectorReferenceComparator(
        minimumIoU: 0.5,
        maximumConfidenceDelta: 0.1
    ).compare(
        reference: [reference],
        candidate: [candidate]
    )

    #expect(comparison.passed == false)
    #expect(comparison.matches.isEmpty)
    #expect(comparison.missingReferenceCount == 1)
    #expect(comparison.extraCandidateCount == 1)
}

@Test
func nativeRestorationRegionComputesIoU() {
    let first = NativeRestorationRegion(x: 0, y: 0, width: 10, height: 10)
    let second = NativeRestorationRegion(x: 5, y: 5, width: 10, height: 10)

    #expect(first.iou(with: first) == 1)
    #expect(abs(first.iou(with: second) - 25.0 / 175.0) < 0.0001)
}

@Test
func nativeFrameRestorationPipelineAcceptsDetectorBackedRegions() throws {
    let source: [UInt8] = Array(repeating: [0, 0, 0, 255], count: 64).flatMap { $0 }
    let detector = FixedNativeMosaicDetector(
        detections: [
            NativeMosaicDetection(
                confidence: 0.95,
                boundingBox: NativeRestorationRegion(x: 1, y: 1, width: 3, height: 3),
                mask: NativeDetectionMaskMetadata(
                    width: 32,
                    height: 32,
                    coordinateSpace: .modelInput
                )
            )
        ]
    )
    let pipeline = try NativeFrameRestorationPipeline(
        imageProcessor: NativeMetalImageProcessor(),
        regionProvider: NativeDetectorRegionProvider(detector: detector),
        modelInputSize: 4,
        blendStrength: 1
    )

    let output = try pipeline.process(
        frame: NativeBGRAFrame(width: 8, height: 8, bytes: source)
    )

    for y in 0..<8 {
        for x in 0..<8 {
            let base = (y * 8 + x) * 4
            let insideDetection = x >= 1 && x < 4 && y >= 1 && y < 4
            #expect(output.bytes[base] == (insideDetection ? 40 : 0))
            #expect(output.bytes[base + 1] == (insideDetection ? 40 : 0))
            #expect(output.bytes[base + 2] == (insideDetection ? 40 : 0))
            #expect(output.bytes[base + 3] == 255)
        }
    }
}

@Test
func nativeCoreMLDetectorReportsMissingModel() throws {
    let missingURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("MissingLadaMosaicDetector-\(UUID().uuidString).mlmodelc")
    let detector = NativeCoreMLMosaicDetector(modelURL: missingURL)

    #expect(detector.availability.isAvailable == false)
    #expect(try detector.detections(for: NativeBGRAFrame(
        width: 2,
        height: 2,
        bytes: Array(repeating: [0, 0, 0, 255], count: 4).flatMap { $0 }
    )).isEmpty)
}

@Test
func nativeModelBundleCatalogNamesExpectedCoreMLBundles() {
    #expect(NativeModelBundleCatalog.detectorModelName == "LadaMosaicDetector")
    #expect(NativeModelBundleCatalog.restorerModelName == "LadaMosaicRestorer")
    #expect(NativeModelBundleCatalog.expectedModelNames() == [
        "LadaMosaicDetector",
        "LadaMosaicRestorer"
    ])
    #expect(NativeModelBundleCatalog.expectedCoreAIModelNames() == [
        "LadaMosaicDetector",
        "LadaMosaicRestorer"
    ])
}

@Test
func nativeModelBundleCatalogFindsCoreAIAssetsInBundleResources() throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("LadaCoreAIBundle-\(UUID().uuidString)", isDirectory: true)
    let resources = directory
        .appendingPathComponent("Test.bundle", isDirectory: true)
        .appendingPathComponent("Contents/Resources", isDirectory: true)
    try FileManager.default.createDirectory(at: resources, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    let infoPlist = directory
        .appendingPathComponent("Test.bundle", isDirectory: true)
        .appendingPathComponent("Contents/Info.plist")
    try """
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0"><dict><key>CFBundleIdentifier</key><string>test.lada.coreai</string></dict></plist>
    """.write(to: infoPlist, atomically: true, encoding: .utf8)

    let asset = resources.appendingPathComponent("LadaMosaicDetector.aimodel")
    try Data("placeholder".utf8).write(to: asset)

    let bundle = try #require(Bundle(url: directory.appendingPathComponent("Test.bundle", isDirectory: true)))
    #expect(NativeModelBundleCatalog.coreAIModelURL(named: "LadaMosaicDetector", in: bundle) == asset)
}

@Test
func nativeCoreMLCapabilitiesReportBundledModelStatus() throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("LadaCoreMLModels-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    let detector = directory.appendingPathComponent("LadaMosaicDetector.mlmodelc", isDirectory: true)
    try FileManager.default.createDirectory(at: detector, withIntermediateDirectories: true)

    let capabilities = NativeCoreMLCapabilities(
        detectorModelURL: detector,
        restorerModelURL: directory.appendingPathComponent("LadaMosaicRestorer.mlmodelc", isDirectory: true)
    )

    #expect(capabilities.isDetectorAvailable)
    #expect(!capabilities.isRestorerAvailable)
    #expect(capabilities.statusTitle == "Detector ready")
    #expect(capabilities.statusDetail == "Core ML detector bundled · Core ML restorer missing")
}

@Test
func nativeCoreAICapabilitiesReportRuntimeAndAssetReadiness() {
    let capabilities = NativeCoreAICapabilities.current()

    #expect(capabilities.statusDetail.isEmpty == false)
    #expect(capabilities.isReadyForAssets == (
        capabilities.isFrameworkPresent &&
        capabilities.isSwiftModuleAvailable &&
        capabilities.hasAnyAsset &&
        !capabilities.hasInvalidAsset
    ))
}

@Test
func nativeCoreAICapabilitiesSummarizeDetectedAimodelAssets() {
    let capabilities = NativeCoreAICapabilities(
        isFrameworkPresent: true,
        isSwiftModuleAvailable: true,
        deviceArchitectureName: "test-apple-silicon",
        availableComputeUnits: ["GPU", "Neural Engine"],
        detectorAssetURL: URL(fileURLWithPath: "/tmp/LadaMosaicDetector.aimodel"),
        restorerAssetURL: nil,
        detectorAssetSummary: NativeCoreAICapabilities.ModelAssetSummary(
            name: "LadaMosaicDetector",
            isValid: true,
            functions: ["main(input:float32) -> output:float32"],
            computeTypes: ["float16"],
            storageTypes: ["float16:12"],
            error: nil
        ),
        restorerAssetSummary: nil
    )

    #expect(capabilities.isReadyForAssets)
    #expect(capabilities.statusDetail.contains("Core AI ready"))
    #expect(capabilities.statusDetail.contains("LadaMosaicDetector"))
    #expect(capabilities.statusDetail.contains("main(input:float32) -> output:float32"))
}

@Test
func nativeCoreAICapabilitiesDoNotMarkInvalidAimodelReady() {
    let capabilities = NativeCoreAICapabilities(
        isFrameworkPresent: true,
        isSwiftModuleAvailable: true,
        deviceArchitectureName: "test-apple-silicon",
        availableComputeUnits: ["GPU"],
        detectorAssetURL: URL(fileURLWithPath: "/tmp/LadaMosaicDetector.aimodel"),
        restorerAssetURL: nil,
        detectorAssetSummary: NativeCoreAICapabilities.ModelAssetSummary(
            name: "LadaMosaicDetector",
            isValid: false,
            functions: [],
            computeTypes: [],
            storageTypes: [],
            error: "invalid test asset"
        ),
        restorerAssetSummary: nil
    )

    #expect(!capabilities.isReadyForAssets)
    #expect(capabilities.statusDetail.contains("needs attention"))
    #expect(capabilities.statusDetail.contains("invalid test asset"))
}

@Test
func nativeCoreAIEngineProbeReportsMissingAssetsUntilAimodelsAreBundled() async {
    let engine = NativeCoreAIEngine(
        capabilitiesProvider: {
            NativeCoreAICapabilities(
                isFrameworkPresent: true,
                isSwiftModuleAvailable: true,
                deviceArchitectureName: "test-apple-silicon",
                availableComputeUnits: ["CPU", "GPU", "Neural Engine"],
                detectorAssetURL: nil,
                restorerAssetURL: nil,
                detectorAssetSummary: nil,
                restorerAssetSummary: nil
            )
        }
    )

    let status = await engine.probe()
    #expect(status.title == "Engine unavailable")
    #expect(status.detail.contains("waiting for .aimodel assets"))
}

@Test
func nativeRestorerModelContractMatchesBasicVSRClipLayout() {
    #expect(NativeRestorerModelContract.modelName == "LadaMosaicRestorer")
    #expect(NativeRestorerModelContract.inputFeatureName == "frames")
    #expect(NativeRestorerModelContract.outputFeatureName == "restored_frames")
    #expect(NativeRestorerModelContract.layout == "BTCHW")
    #expect(NativeRestorerModelContract.colorOrder == "BGR")
    #expect(NativeRestorerModelContract.inputShape() == [1, 16, 3, 256, 256])
    #expect(NativeRestorerModelContract.outputShape() == [1, 16, 3, 256, 256])
    #expect(NativeRestorerModelContract.expectedElementCount() == 3_145_728)
}

@Test
func nativeCoreMLDetectorRecognizesBundledCompiledModelPlaceholder() throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("LadaMosaicDetector-\(UUID().uuidString).mlmodelc", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    let detector = NativeCoreMLMosaicDetector(modelURL: directory)

    #expect(detector.availability.isAvailable)
    #expect(detector.makeModelConfiguration().computeUnits == .cpuAndNeuralEngine)
}

@Test
func nativeCoreMLDetectorBuildsYOLOInputTensorFromBGRAFrame() throws {
    let postprocessor = NativeYOLOPostprocessor(modelInputSize: 2)
    let detector = NativeCoreMLMosaicDetector(
        modelURL: nil,
        postprocessor: postprocessor
    )
    let frame = NativeBGRAFrame(
        width: 2,
        height: 2,
        bytes: [
            10, 20, 30, 255,
            40, 50, 60, 255,
            70, 80, 90, 255,
            100, 110, 120, 255
        ]
    )

    let provider = try detector.makeInputFeatureProvider(from: frame)
    let input = try #require(
        provider.featureValue(
            for: NativeCoreMLMosaicDetector.inputFeatureName
        )?.multiArrayValue
    )

    #expect(input.shape.map(\.intValue) == [1, 3, 2, 2])
    #expect(Float(truncating: input[0]) == Float(30) / 255)
    #expect(Float(truncating: input[4]) == Float(20) / 255)
    #expect(Float(truncating: input[8]) == Float(10) / 255)
    #expect(Float(truncating: input[3]) == Float(120) / 255)
    #expect(Float(truncating: input[7]) == Float(110) / 255)
    #expect(Float(truncating: input[11]) == Float(100) / 255)
}

@Test
func nativeCoreMLDetectorBuildsYOLOInputTensorFromPixelBuffer() throws {
    let postprocessor = NativeYOLOPostprocessor(modelInputSize: 2)
    let detector = NativeCoreMLMosaicDetector(
        modelURL: nil,
        postprocessor: postprocessor
    )
    let frame = NativeBGRAFrame(
        width: 2,
        height: 2,
        bytes: [
            10, 20, 30, 255,
            40, 50, 60, 255,
            70, 80, 90, 255,
            100, 110, 120, 255
        ]
    )
    let pixelBuffer = try NativePixelBufferBridge.makePixelBuffer(from: frame)

    let provider = try detector.makeInputFeatureProvider(from: pixelBuffer)
    let input = try #require(
        provider.featureValue(
            for: NativeCoreMLMosaicDetector.inputFeatureName
        )?.multiArrayValue
    )

    #expect(input.shape.map(\.intValue) == [1, 3, 2, 2])
    #expect(Float(truncating: input[0]) == Float(30) / 255)
    #expect(Float(truncating: input[4]) == Float(20) / 255)
    #expect(Float(truncating: input[8]) == Float(10) / 255)
    #expect(Float(truncating: input[3]) == Float(120) / 255)
    #expect(Float(truncating: input[7]) == Float(110) / 255)
    #expect(Float(truncating: input[11]) == Float(100) / 255)
}

@Test
func nativeCoreMLDetectorLetterboxesWideFrameInputTensor() throws {
    let postprocessor = NativeYOLOPostprocessor(modelInputSize: 4)
    let detector = NativeCoreMLMosaicDetector(
        modelURL: nil,
        postprocessor: postprocessor
    )
    let frame = NativeBGRAFrame(
        width: 4,
        height: 2,
        bytes: [
            10, 20, 30, 255,
            40, 50, 60, 255,
            70, 80, 90, 255,
            100, 110, 120, 255,
            130, 140, 150, 255,
            160, 170, 180, 255,
            190, 200, 210, 255,
            220, 230, 240, 255
        ]
    )

    let provider = try detector.makeInputFeatureProvider(from: frame)
    let input = try #require(
        provider.featureValue(
            for: NativeCoreMLMosaicDetector.inputFeatureName
        )?.multiArrayValue
    )
    let planeSize = 4 * 4
    let topLeft = 0
    let firstContent = 4
    let lastContent = 11
    let bottomRight = 15

    #expect(Float(truncating: input[topLeft]) == Float(114) / 255)
    #expect(Float(truncating: input[planeSize + topLeft]) == Float(114) / 255)
    #expect(Float(truncating: input[planeSize * 2 + topLeft]) == Float(114) / 255)
    #expect(Float(truncating: input[firstContent]) == Float(30) / 255)
    #expect(Float(truncating: input[planeSize + firstContent]) == Float(20) / 255)
    #expect(Float(truncating: input[planeSize * 2 + firstContent]) == Float(10) / 255)
    #expect(Float(truncating: input[lastContent]) == Float(240) / 255)
    #expect(Float(truncating: input[planeSize + lastContent]) == Float(230) / 255)
    #expect(Float(truncating: input[planeSize * 2 + lastContent]) == Float(220) / 255)
    #expect(Float(truncating: input[bottomRight]) == Float(114) / 255)
}

@Test
func nativeCoreMLDetectorBuildsUltralyticsImageInput() throws {
    let postprocessor = NativeYOLOPostprocessor(modelInputSize: 4)
    let detector = NativeCoreMLMosaicDetector(
        modelURL: nil,
        postprocessor: postprocessor
    )
    let frame = NativeBGRAFrame(
        width: 4,
        height: 2,
        bytes: [
            10, 20, 30, 255,
            40, 50, 60, 255,
            70, 80, 90, 255,
            100, 110, 120, 255,
            130, 140, 150, 255,
            160, 170, 180, 255,
            190, 200, 210, 255,
            220, 230, 240, 255
        ]
    )

    let provider = try detector.makeImageInputFeatureProvider(from: frame)
    let pixelBuffer = try #require(
        provider.featureValue(
            for: NativeCoreMLMosaicDetector.imageInputFeatureName
        )?.imageBufferValue
    )
    let roundTripped = try NativePixelBufferBridge.copyBGRAFrame(from: pixelBuffer)

    #expect(roundTripped.width == 4)
    #expect(roundTripped.height == 4)
    #expect(roundTripped.bytes[0] == 114)
    #expect(roundTripped.bytes[1] == 114)
    #expect(roundTripped.bytes[2] == 114)
    #expect(roundTripped.bytes[16] == 10)
    #expect(roundTripped.bytes[17] == 20)
    #expect(roundTripped.bytes[18] == 30)
    #expect(roundTripped.bytes[44] == 220)
    #expect(roundTripped.bytes[45] == 230)
    #expect(roundTripped.bytes[46] == 240)
    #expect(roundTripped.bytes[60] == 114)
}

@Test
func nativeCoreMLDetectorBuildsUltralyticsImageInputFromPixelBuffer() throws {
    let postprocessor = NativeYOLOPostprocessor(modelInputSize: 4)
    let detector = NativeCoreMLMosaicDetector(
        modelURL: nil,
        postprocessor: postprocessor
    )
    let frame = NativeBGRAFrame(
        width: 4,
        height: 2,
        bytes: [
            10, 20, 30, 255,
            40, 50, 60, 255,
            70, 80, 90, 255,
            100, 110, 120, 255,
            130, 140, 150, 255,
            160, 170, 180, 255,
            190, 200, 210, 255,
            220, 230, 240, 255
        ]
    )
    let sourcePixelBuffer = try NativePixelBufferBridge.makePixelBuffer(from: frame)

    let provider = try detector.makeImageInputFeatureProvider(from: sourcePixelBuffer)
    let pixelBuffer = try #require(
        provider.featureValue(
            for: NativeCoreMLMosaicDetector.imageInputFeatureName
        )?.imageBufferValue
    )
    let roundTripped = try NativePixelBufferBridge.copyBGRAFrame(from: pixelBuffer)

    #expect(roundTripped.width == 4)
    #expect(roundTripped.height == 4)
    #expect(roundTripped.bytes[0] == 114)
    #expect(roundTripped.bytes[1] == 114)
    #expect(roundTripped.bytes[2] == 114)
    #expect(roundTripped.bytes[16] == 10)
    #expect(roundTripped.bytes[17] == 20)
    #expect(roundTripped.bytes[18] == 30)
    #expect(roundTripped.bytes[44] == 220)
    #expect(roundTripped.bytes[45] == 230)
    #expect(roundTripped.bytes[46] == 240)
    #expect(roundTripped.bytes[60] == 114)
}

@Test
func nativeCoreMLDetectorParsesYOLOFeatureProviderOutputs() throws {
    let postprocessor = NativeYOLOPostprocessor(
        confidenceThreshold: 0.25,
        iouThreshold: 0.7
    )
    let detector = NativeCoreMLMosaicDetector(
        modelURL: nil,
        postprocessor: postprocessor
    )
    var output0 = emptyYOLOOutput()
    setYOLOAnchor(
        &output0,
        anchor: 11,
        centerX: 320,
        centerY: 320,
        width: 64,
        height: 128,
        confidence: 0.75
    )
    let outputProvider = try MLDictionaryFeatureProvider(dictionary: [
        NativeCoreMLMosaicDetector.outputFeatureName: MLFeatureValue(
            multiArray: try makeYOLOMultiArray(
                values: output0,
                shape: [
                    1,
                    NativeYOLOOutputShape.ladaMosaicDetector.channels,
                    NativeYOLOOutputShape.ladaMosaicDetector.anchors
                ]
            )
        ),
        NativeCoreMLMosaicDetector.prototypeFeatureName: MLFeatureValue(
            multiArray: try makeYOLOMultiArray(
                values: emptyYOLOPrototypes(),
                shape: [
                    1,
                    NativeYOLOPrototypeShape.ladaMosaicDetector.channels,
                    NativeYOLOPrototypeShape.ladaMosaicDetector.height,
                    NativeYOLOPrototypeShape.ladaMosaicDetector.width
                ]
            )
        )
    ])

    let detections = try detector.detections(
        from: outputProvider,
        frameWidth: 640,
        frameHeight: 640
    )

    #expect(detections.count == 1)
    #expect(detections.first?.confidence == 0.75)
    #expect(detections.first?.boundingBox.x == 288)
    #expect(detections.first?.boundingBox.y == 256)
    #expect(detections.first?.boundingBox.width == 64)
    #expect(detections.first?.boundingBox.height == 128)
}

@Test
func nativeCoreMLDetectorParsesUltralyticsFeatureProviderOutputs() throws {
    let postprocessor = NativeYOLOPostprocessor(
        confidenceThreshold: 0.25,
        iouThreshold: 0.7
    )
    let detector = NativeCoreMLMosaicDetector(
        modelURL: nil,
        postprocessor: postprocessor
    )
    var output0 = emptyYOLOOutput()
    setYOLOAnchor(
        &output0,
        anchor: 5,
        centerX: 320,
        centerY: 320,
        width: 64,
        height: 64,
        confidence: 0.8
    )
    let outputProvider = try MLDictionaryFeatureProvider(dictionary: [
        NativeCoreMLMosaicDetector.ultralyticsOutputFeatureName: MLFeatureValue(
            multiArray: try makeYOLOMultiArray(
                values: output0,
                shape: [
                    1,
                    NativeYOLOOutputShape.ladaMosaicDetector.channels,
                    NativeYOLOOutputShape.ladaMosaicDetector.anchors
                ]
            )
        ),
        NativeCoreMLMosaicDetector.ultralyticsPrototypeFeatureName: MLFeatureValue(
            multiArray: try makeYOLOMultiArray(
                values: emptyYOLOPrototypes(),
                shape: [
                    1,
                    NativeYOLOPrototypeShape.ladaMosaicDetector.channels,
                    NativeYOLOPrototypeShape.ladaMosaicDetector.height,
                    NativeYOLOPrototypeShape.ladaMosaicDetector.width
                ]
            )
        )
    ])

    let detections = try detector.detections(
        from: outputProvider,
        frameWidth: 640,
        frameHeight: 640
    )

    #expect(detections.count == 1)
    #expect(detections.first?.confidence == 0.8)
    #expect(detections.first?.boundingBox.x == 288)
    #expect(detections.first?.boundingBox.y == 288)
    #expect(detections.first?.boundingBox.width == 64)
    #expect(detections.first?.boundingBox.height == 64)
}

@Test
func nativeCoreMLDetectorReportsMissingYOLOOutputs() throws {
    let detector = NativeCoreMLMosaicDetector(modelURL: nil)
    let provider = try MLDictionaryFeatureProvider(dictionary: [:])

    #expect(throws: NativeCoreMLDetectorError.self) {
        _ = try detector.detections(
            from: provider,
            frameWidth: 640,
            frameHeight: 640
        )
    }
}

@Test
func nativeCoreMLDetectorRunsCompiledUltralyticsModelWhenPresent() throws {
    let modelURL = try projectRootURL()
        .appendingPathComponent("native-models/LadaMosaicDetector.mlmodelc")
    guard FileManager.default.fileExists(atPath: modelURL.path) else {
        return
    }

    let detector = NativeCoreMLMosaicDetector(modelURL: modelURL)
    let frame = NativeBGRAFrame(
        width: 320,
        height: 240,
        bytes: Array(repeating: [114, 114, 114, 255], count: 320 * 240).flatMap { $0 }
    )
    let detections = try detector.detections(for: frame)

    #expect(detector.availability.isAvailable)
    #expect(detections.count >= 0)
}

@Test
func nativeCoreMLDetectorMatchesPythonReferenceOnSmokeFrameWhenPresent() async throws {
    let modelURL = try projectRootURL()
        .appendingPathComponent("native-models/LadaMosaicDetector.mlmodelc")
    guard FileManager.default.fileExists(atPath: modelURL.path) else {
        return
    }

    let fixtureURL = try projectRootURL()
        .appendingPathComponent("native-models/reference-detections/smoke-input-yolo-reference.json")
    let fixture = try NativeDetectorReferenceFixture.load(from: fixtureURL)
    let reference = try #require(fixture.frames.first?.nativeDetections)
    let input = try projectRootURL()
        .appendingPathComponent("work/smoke-input.mp4")
    let frame = try await firstDecodedBGRAFrame(from: input)
    let detector = NativeCoreMLMosaicDetector(modelURL: modelURL)

    let candidate = try detector.detections(for: frame)
    let comparison = NativeDetectorReferenceComparator(
        minimumIoU: 0.85,
        maximumConfidenceDelta: 0.1
    ).compare(
        reference: reference,
        candidate: candidate
    )

    #expect(comparison.passed)
    #expect(comparison.minimumIoU > 0.9)
    #expect(comparison.maximumConfidenceDelta < 0.1)
}

@Test
func nativeYOLOPostprocessorMapsOutputShapeToDetections() {
    let postprocessor = NativeYOLOPostprocessor(
        confidenceThreshold: 0.25,
        iouThreshold: 0.7
    )
    var output0 = emptyYOLOOutput()
    let output1 = emptyYOLOPrototypes()
    setYOLOAnchor(
        &output0,
        anchor: 7,
        centerX: 320,
        centerY: 320,
        width: 160,
        height: 80,
        confidence: 0.8
    )

    let detections = postprocessor.detections(
        output0: output0,
        output1: output1,
        frameWidth: 1280,
        frameHeight: 720
    )

    #expect(detections.count == 1)
    #expect(detections.first?.confidence == 0.8)
    #expect(detections.first?.boundingBox.x == 480)
    #expect(detections.first?.boundingBox.y == 280)
    #expect(detections.first?.boundingBox.width == 320)
    #expect(detections.first?.boundingBox.height == 160)
    #expect(detections.first?.mask?.width == 160)
    #expect(detections.first?.mask?.height == 160)
    #expect(detections.first?.mask?.coordinateSpace == .modelInput)
}

@Test
func nativeYOLOPostprocessorRemovesLetterboxPaddingFromWideFrames() {
    let postprocessor = NativeYOLOPostprocessor(
        modelInputSize: 640,
        confidenceThreshold: 0.25,
        iouThreshold: 0.7
    )
    var output0 = emptyYOLOOutput()
    let output1 = emptyYOLOPrototypes()
    setYOLOAnchor(
        &output0,
        anchor: 9,
        centerX: 320,
        centerY: 320,
        width: 100,
        height: 100,
        confidence: 0.9
    )

    let detections = postprocessor.detections(
        output0: output0,
        output1: output1,
        frameWidth: 1280,
        frameHeight: 720
    )

    #expect(detections.count == 1)
    #expect(detections.first?.boundingBox.x == 540)
    #expect(detections.first?.boundingBox.y == 260)
    #expect(detections.first?.boundingBox.width == 200)
    #expect(detections.first?.boundingBox.height == 200)
}

@Test
func nativeYOLOPostprocessorSuppressesOverlappingDetections() {
    let postprocessor = NativeYOLOPostprocessor(
        confidenceThreshold: 0.25,
        iouThreshold: 0.5
    )
    var output0 = emptyYOLOOutput()
    let output1 = emptyYOLOPrototypes()
    setYOLOAnchor(
        &output0,
        anchor: 1,
        centerX: 320,
        centerY: 320,
        width: 100,
        height: 100,
        confidence: 0.9
    )
    setYOLOAnchor(
        &output0,
        anchor: 2,
        centerX: 322,
        centerY: 322,
        width: 100,
        height: 100,
        confidence: 0.7
    )

    let detections = postprocessor.detections(
        output0: output0,
        output1: output1,
        frameWidth: 640,
        frameHeight: 640
    )

    #expect(detections.count == 1)
    #expect(detections.first?.confidence == 0.9)
}

@Test
func nativeYOLOPostprocessorRejectsUnexpectedShapes() {
    let postprocessor = NativeYOLOPostprocessor()
    let detections = postprocessor.detections(
        output0: [0],
        output1: emptyYOLOPrototypes(),
        frameWidth: 640,
        frameHeight: 640
    )

    #expect(detections.isEmpty)
}

@Test
func nativeMetalPipelineProcessesDecodedVideoFrame() async throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("LadaMacNativeFramePipelineTest-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    let input = directory.appendingPathComponent("input.mp4")
    try makeTestVideo(at: input)

    let frame = try await firstDecodedBGRAFrame(from: input)
    let processor = try NativeMetalImageProcessor()

    let crop = try processor.cropBGRA(
        source: frame.bytes,
        width: frame.width,
        height: frame.height,
        x: 8,
        y: 8,
        cropWidth: 16,
        cropHeight: 16
    )
    let resized = try processor.resizeBGRANearest(
        source: crop,
        width: 16,
        height: 16,
        outputWidth: 8,
        outputHeight: 8
    )
    let restored = [UInt8](repeating: 255, count: resized.count)
    let mask = [Float](repeating: 0.25, count: 8 * 8)
    let blended = try processor.blendBGRA(
        source: resized,
        restored: restored,
        mask: mask,
        width: 8,
        height: 8
    )
    let processedFrame = NativeBGRAFrame(width: 8, height: 8, bytes: blended)
    let processedPixelBuffer = try NativePixelBufferBridge.makePixelBuffer(from: processedFrame)
    let roundTrippedFrame = try NativePixelBufferBridge.copyBGRAFrame(from: processedPixelBuffer)
    let processedOutput = directory.appendingPathComponent("processed-output.mp4")
    try writeTestVideoFrame(processedPixelBuffer, width: 8, height: 8, to: processedOutput)
    let processedMetadata = await VideoInspector.metadata(for: processedOutput)

    #expect(frame.width == 64)
    #expect(frame.height == 64)
    #expect(frame.bytes.count == 64 * 64 * 4)
    #expect(crop.count == 16 * 16 * 4)
    #expect(resized.count == 8 * 8 * 4)
    #expect(blended.count == 8 * 8 * 4)
    #expect(roundTrippedFrame.width == 8)
    #expect(roundTrippedFrame.height == 8)
    #expect(roundTrippedFrame.bytes == blended)
    #expect(FileManager.default.fileExists(atPath: processedOutput.path))
    #expect(processedMetadata.dimensions.width == 8)
    #expect(processedMetadata.dimensions.height == 8)
    #expect(stride(from: 3, to: blended.count, by: 4).allSatisfy { blended[$0] == resized[$0] })
}

@Test
@MainActor
func completedJobCanBeMarkedForPreview() throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("LadaMacPreviewTest-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    let source = directory.appendingPathComponent("source.mp4")
    let restored = directory.appendingPathComponent("source.restored.mp4")
    FileManager.default.createFile(atPath: source.path, contents: Data())
    FileManager.default.createFile(atPath: restored.path, contents: Data())

    let queue = RestorationQueue()
    let job = RestorationJob(sourceURL: source, destinationURL: directory)
    job.state = .completed
    queue.jobs = [job]

    queue.preview(job)

    #expect(queue.previewJobID == job.id)
    #expect(queue.previewJob === job)
}

@Test
func nativeMetalEngineTranscodesVideoSmoke() async throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("LadaMacNativeVideoTest-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    let input = directory.appendingPathComponent("input.mp4")
    let output = directory.appendingPathComponent("output.mp4")
    try makeTestVideo(at: input)

    let progress = ProgressRecorder()
    try await NativeMetalEngine().restore(
        request: RestorationRequest(
            input: input,
            output: output,
            device: "native",
            detectionModel: "native-smoke",
            maxClipLength: 1,
            memoryMode: MemoryMode.automatic.rawValue,
            encodingPreset: OutputFormat.h264.encodingPreset
        ),
        progress: { value, _ in
        progress.record(value)
        },
        diagnostic: { _ in }
    )

    let metadata = await VideoInspector.metadata(for: output)
    #expect(FileManager.default.fileExists(atPath: output.path))
    #expect(metadata.duration > 0)
    #expect(metadata.dimensions.width == 64)
    #expect(metadata.dimensions.height == 64)
    #expect(progress.maximum == 1)
}

@Test
func nativeMetalEngineProcessesFramesWhenExperimentalPathEnabled() async throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("LadaMacNativeProcessedVideoTest-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    let input = directory.appendingPathComponent("input.mp4")
    let output = directory.appendingPathComponent("processed-output.mp4")
    try makeTestVideo(at: input)

    let progress = ProgressRecorder()
    try await NativeMetalEngine(experimentalProcessedFrames: true).restore(
        request: RestorationRequest(
            input: input,
            output: output,
            device: "native",
            detectionModel: "native-processed-smoke",
            maxClipLength: 1,
            memoryMode: MemoryMode.automatic.rawValue,
            encodingPreset: OutputFormat.h264.encodingPreset
        ),
        progress: { value, _ in
        progress.record(value)
        },
        diagnostic: { _ in }
    )

    let metadata = await VideoInspector.metadata(for: output)
    #expect(FileManager.default.fileExists(atPath: output.path))
    #expect(metadata.duration > 0)
    #expect(metadata.dimensions.width == 64)
    #expect(metadata.dimensions.height == 64)
    #expect(progress.maximum == 1)
}

@Test
func nativeMetalEngineUsesDetectorBackedRegionProviderWhenProcessingFrames() async throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("LadaMacNativeDetectorProviderVideoTest-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    let input = directory.appendingPathComponent("input.mp4")
    let output = directory.appendingPathComponent("processed-output.mp4")
    try makeTestVideo(at: input)

    let provider = CountingNativeRestorationRegionProvider(
        regions: [
            NativeRestorationRegion(x: 8, y: 8, width: 16, height: 16)
        ]
    )
    try await NativeMetalEngine(
        experimentalProcessedFrames: true,
        regionProvider: provider
    ).restore(
        request: RestorationRequest(
            input: input,
            output: output,
            device: "native",
            detectionModel: "native-detector-provider-smoke",
            maxClipLength: 1,
            memoryMode: MemoryMode.automatic.rawValue,
            encodingPreset: OutputFormat.h264.encodingPreset
        ),
        progress: { _, _ in },
        diagnostic: { _ in }
    )

    let metadata = await VideoInspector.metadata(for: output)
    #expect(FileManager.default.fileExists(atPath: output.path))
    #expect(metadata.duration > 0)
    #expect(provider.callCount > 0)
}

@Test
func nativeMetalEnginePreservesAudioTrackSmoke() async throws {
    let root = try projectRootURL()
    let input = root.appendingPathComponent("work/smoke-audio-input.mp4")
    #expect(FileManager.default.fileExists(atPath: input.path))

    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("LadaMacNativeAudioTest-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    let output = directory.appendingPathComponent("output-with-audio.mp4")
    try await NativeMetalEngine().restore(
        request: RestorationRequest(
            input: input,
            output: output,
            device: "native",
            detectionModel: "native-audio-smoke",
            maxClipLength: 1,
            memoryMode: MemoryMode.automatic.rawValue,
            encodingPreset: OutputFormat.h264.encodingPreset
        ),
        progress: { _, _ in },
        diagnostic: { _ in }
    )

    let inputTrackCounts = await trackCounts(for: input)
    let outputTrackCounts = await trackCounts(for: output)
    #expect(inputTrackCounts.video == 1)
    #expect(inputTrackCounts.audio == 1)
    #expect(outputTrackCounts.video == 1)
    #expect(outputTrackCounts.audio == 1)
}

private func trackCounts(for url: URL) async -> (video: Int, audio: Int) {
    let asset = AVURLAsset(url: url)
    return (
        video: ((try? await asset.loadTracks(withMediaType: .video)) ?? []).count,
        audio: ((try? await asset.loadTracks(withMediaType: .audio)) ?? []).count
    )
}

private func firstDecodedBGRAFrame(from url: URL) async throws -> NativeBGRAFrame {
    let asset = AVURLAsset(url: url)
    guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else {
        throw LadaEngineError.failed("Could not find test video track")
    }
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(
        track: videoTrack,
        outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
    )
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else {
        throw LadaEngineError.failed("Could not add test video frame output")
    }
    reader.add(output)
    guard reader.startReading() else {
        throw reader.error ?? LadaEngineError.failed("Could not start test video reader")
    }
    guard let sample = output.copyNextSampleBuffer(),
          let imageBuffer = CMSampleBufferGetImageBuffer(sample)
    else {
        throw reader.error ?? LadaEngineError.failed("Could not decode first test video frame")
    }
    return try NativePixelBufferBridge.copyBGRAFrame(from: imageBuffer)
}

private func emptyYOLOOutput() -> [Float] {
    [Float](
        repeating: 0,
        count: NativeYOLOOutputShape.ladaMosaicDetector.channels *
            NativeYOLOOutputShape.ladaMosaicDetector.anchors
    )
}

private func emptyYOLOPrototypes() -> [Float] {
    [Float](
        repeating: 0,
        count: NativeYOLOPrototypeShape.ladaMosaicDetector.channels *
            NativeYOLOPrototypeShape.ladaMosaicDetector.height *
            NativeYOLOPrototypeShape.ladaMosaicDetector.width
    )
}

private func setYOLOAnchor(
    _ output: inout [Float],
    anchor: Int,
    centerX: Float,
    centerY: Float,
    width: Float,
    height: Float,
    confidence: Float
) {
    let anchors = NativeYOLOOutputShape.ladaMosaicDetector.anchors
    output[0 * anchors + anchor] = centerX
    output[1 * anchors + anchor] = centerY
    output[2 * anchors + anchor] = width
    output[3 * anchors + anchor] = height
    output[4 * anchors + anchor] = confidence
}

private func makeYOLOMultiArray(
    values: [Float],
    shape: [Int]
) throws -> MLMultiArray {
    let array = try MLMultiArray(
        shape: shape.map(NSNumber.init(value:)),
        dataType: .float32
    )
    let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
    values.withUnsafeBufferPointer { source in
        guard let baseAddress = source.baseAddress else {
            return
        }
        pointer.update(from: baseAddress, count: values.count)
    }
    return array
}

private final class ProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Double] = []

    var maximum: Double {
        lock.withLock {
            values.max() ?? 0
        }
    }

    func record(_ value: Double) {
        lock.withLock {
            values.append(value)
        }
    }
}

private final class CountingNativeRestorationRegionProvider: NativeRestorationRegionProvider, @unchecked Sendable {
    private let lock = NSLock()
    private var calls = 0
    private let suppliedRegions: [NativeRestorationRegion]

    init(regions: [NativeRestorationRegion]) {
        self.suppliedRegions = regions
    }

    var callCount: Int {
        lock.withLock {
            calls
        }
    }

    func regions(for frame: NativeBGRAFrame) throws -> [NativeRestorationRegion] {
        lock.withLock {
            calls += 1
        }
        return suppliedRegions.compactMap { $0.clamped(to: frame) }
    }
}

private final class CountingGeometryNativeRestorationRegionProvider: NativeGeometryRestorationRegionProvider, @unchecked Sendable {
    private let lock = NSLock()
    private var frameCalls = 0
    private var geometryCalls = 0
    private let suppliedRegions: [NativeRestorationRegion]

    init(regions: [NativeRestorationRegion]) {
        self.suppliedRegions = regions
    }

    var frameCallCount: Int {
        lock.withLock {
            frameCalls
        }
    }

    var geometryCallCount: Int {
        lock.withLock {
            geometryCalls
        }
    }

    func regions(for frame: NativeBGRAFrame) throws -> [NativeRestorationRegion] {
        lock.withLock {
            frameCalls += 1
        }
        return suppliedRegions.compactMap { $0.clamped(to: frame) }
    }

    func regions(frameWidth: Int, frameHeight: Int) throws -> [NativeRestorationRegion] {
        lock.withLock {
            geometryCalls += 1
        }
        return suppliedRegions.filter { region in
            region.x >= 0 &&
            region.y >= 0 &&
            region.width > 0 &&
            region.height > 0 &&
            region.x + region.width <= frameWidth &&
            region.y + region.height <= frameHeight
        }
    }
}

private final class FixedNativeRegionRestorer: NativeRegionRestorer, @unchecked Sendable {
    private let lock = NSLock()
    private var calls = 0
    private let restoredPixel: [UInt8]

    init(restoredPixel: [UInt8]) {
        self.restoredPixel = restoredPixel
    }

    var callCount: Int {
        lock.withLock {
            calls
        }
    }

    func restore(
        modelInput: [UInt8],
        width: Int,
        height: Int
    ) throws -> [UInt8] {
        lock.withLock {
            calls += 1
        }
        return Array(repeating: restoredPixel, count: width * height).flatMap { $0 }
    }
}

private final class FixedNativeTemporalRestorer: NativeTemporalRegionRestorer, @unchecked Sendable {
    private let lock = NSLock()
    private var calls = 0
    private let replacementPixel: [UInt8]

    init(replacementPixel: [UInt8]) {
        self.replacementPixel = replacementPixel
    }

    var callCount: Int {
        lock.withLock {
            calls
        }
    }

    func restore(clip: NativeRestorationClip) throws -> NativeRestorationClip {
        lock.withLock {
            calls += 1
        }
        return NativeRestorationClip(
            frames: clip.frames.map { frame in
                NativeBGRAFrame(
                    width: frame.width,
                    height: frame.height,
                    bytes: Array(
                        repeating: replacementPixel,
                        count: frame.width * frame.height
                    ).flatMap { $0 }
                )
            },
            frameRate: clip.frameRate
        )
    }
}

private func makeTestVideo(at url: URL) throws {
    let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let input = AVAssetWriterInput(
        mediaType: .video,
        outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: 64,
            AVVideoHeightKey: 64
        ]
    )
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: input,
        sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: 64,
            kCVPixelBufferHeightKey as String: 64
        ]
    )
    writer.add(input)
    writer.startWriting()
    writer.startSession(atSourceTime: .zero)

    for frame in 0..<6 {
        while !input.isReadyForMoreMediaData {
            Thread.sleep(forTimeInterval: 0.01)
        }
        let buffer = try makePixelBuffer(frame: frame)
        let time = CMTime(value: CMTimeValue(frame), timescale: 30)
        #expect(adaptor.append(buffer, withPresentationTime: time))
    }

    input.markAsFinished()
    let semaphore = DispatchSemaphore(value: 0)
    writer.finishWriting {
        semaphore.signal()
    }
    semaphore.wait()
    if writer.status == .failed {
        throw writer.error ?? LadaEngineError.failed("Test video writer failed")
    }
}

private func writeTestVideoFrame(
    _ pixelBuffer: CVPixelBuffer,
    width: Int,
    height: Int,
    to url: URL
) throws {
    let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let input = AVAssetWriterInput(
        mediaType: .video,
        outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height
        ]
    )
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: input,
        sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height
        ]
    )
    writer.add(input)
    writer.startWriting()
    writer.startSession(atSourceTime: .zero)

    for frame in 0..<2 {
        while !input.isReadyForMoreMediaData {
            Thread.sleep(forTimeInterval: 0.01)
        }
        let time = CMTime(value: CMTimeValue(frame), timescale: 30)
        #expect(adaptor.append(pixelBuffer, withPresentationTime: time))
    }

    input.markAsFinished()
    let semaphore = DispatchSemaphore(value: 0)
    writer.finishWriting {
        semaphore.signal()
    }
    semaphore.wait()
    if writer.status == .failed {
        throw writer.error ?? LadaEngineError.failed("Processed test video writer failed")
    }
}

private func makePixelBuffer(frame: Int) throws -> CVPixelBuffer {
    var maybeBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        64,
        64,
        kCVPixelFormatType_32BGRA,
        [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary,
        &maybeBuffer
    )
    guard status == kCVReturnSuccess, let buffer = maybeBuffer else {
        throw LadaEngineError.failed("Could not create test pixel buffer")
    }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else {
        throw LadaEngineError.failed("Could not access test pixel buffer")
    }
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    let pointer = baseAddress.assumingMemoryBound(to: UInt8.self)
    for y in 0..<64 {
        for x in 0..<64 {
            let offset = y * bytesPerRow + x * 4
            pointer[offset] = UInt8((x * 4 + frame * 12) % 255)
            pointer[offset + 1] = UInt8((y * 4 + frame * 20) % 255)
            pointer[offset + 2] = UInt8((frame * 36) % 255)
            pointer[offset + 3] = 255
        }
    }
    return buffer
}

private func projectRootURL(
    from start: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
) throws -> URL {
    var candidate = start.standardizedFileURL
    while true {
        if FileManager.default.fileExists(
            atPath: candidate.appendingPathComponent("Package.swift").path
        ) {
            return candidate
        }
        let parent = candidate.deletingLastPathComponent()
        if parent.path == candidate.path {
            throw LadaEngineError.failed(
                "Could not locate Package.swift from \(start.path)"
            )
        }
        candidate = parent
    }
}
