import AVFoundation
import CoreVideo
import Foundation

final class NativeMetalEngine: RestorationEngine, @unchecked Sendable {
    private let lock = NSLock()
    private let experimentalProcessedFrames: Bool
    private let regionProvider: (any NativeRestorationRegionProvider)?
    private var isCancelled = false

    init(
        experimentalProcessedFrames: Bool = ProcessInfo.processInfo.environment["LADA_NATIVE_PROCESSED_FRAMES"] == "1",
        regionProvider: (any NativeRestorationRegionProvider)? = nil
    ) {
        self.experimentalProcessedFrames = experimentalProcessedFrames
        self.regionProvider = regionProvider
    }

    func restore(
        request: RestorationRequest,
        progress: @escaping @Sendable (Double, TimeInterval?) -> Void,
        diagnostic: @escaping @Sendable (RestorationEngineDiagnostic) -> Void
    ) async throws {
        lock.withLock {
            isCancelled = false
        }
        diagnostic(RestorationEngineDiagnostic(event: "native-started"))
        try await NativeVideoPassthroughTranscoder.transcode(
            input: request.input,
            output: request.output,
            encodingPreset: request.encodingPreset,
            frameProcessor: makeFrameProcessor(),
            progress: progress,
            isCancelled: { [weak self] in
                self?.lock.withLock { self?.isCancelled ?? false } ?? false
            }
        )
        diagnostic(RestorationEngineDiagnostic(event: "native-completed"))
    }

    func probe() async -> EngineStatus {
        let metal = MetalCapabilities.current
        guard metal.isMetal4Ready else {
            return .unavailable(metal.statusDetail)
        }
        do {
            _ = try NativeMetalImageProcessor()
        } catch {
            return .unavailable(error.localizedDescription)
        }
        return .ready("\(metal.statusDetail) · Native AVFoundation + Metal image pipeline ready")
    }

    func cancel() {
        lock.withLock {
            isCancelled = true
        }
    }

    func pause() {}
    func resume() {}

    private func makeFrameProcessor() throws -> (@Sendable (CVPixelBuffer) throws -> CVPixelBuffer)? {
        guard experimentalProcessedFrames else {
            return nil
        }
        let pipeline = try NativeFrameRestorationPipeline(
            imageProcessor: NativeMetalImageProcessor(),
            regionProvider: regionProvider ?? makeDefaultRegionProvider()
        )
        return { pixelBuffer in
            let sourceFrame = try NativePixelBufferBridge.copyBGRAFrame(from: pixelBuffer)
            let processedFrame = try pipeline.process(frame: sourceFrame)
            return try NativePixelBufferBridge.makePixelBuffer(
                from: processedFrame
            )
        }
    }

    private func makeDefaultRegionProvider() -> any NativeRestorationRegionProvider {
        let detector = NativeCoreMLMosaicDetector()
        if detector.availability.isAvailable {
            return NativeDetectorRegionProvider(
                detector: detector,
                minimumConfidence: 0.15
            )
        }
        return CenterNativeRestorationRegionProvider()
    }
}

enum NativeVideoTranscoderError: LocalizedError {
    case videoTrackMissing
    case readerCannotAddOutput
    case writerCannotAddInput
    case cancelled
    case readerFailed(String)
    case writerFailed(String)

    var errorDescription: String? {
        switch self {
        case .videoTrackMissing:
            "The native engine could not find a video track."
        case .readerCannotAddOutput:
            "The native engine could not read this video format."
        case .writerCannotAddInput:
            "The native engine could not create the output video."
        case .cancelled:
            "Native video processing was cancelled."
        case .readerFailed(let message), .writerFailed(let message):
            message
        }
    }
}

enum NativeVideoPassthroughTranscoder {
    static func transcode(
        input: URL,
        output: URL,
        encodingPreset: String,
        frameProcessor: (@Sendable (CVPixelBuffer) throws -> CVPixelBuffer)? = nil,
        progress: @escaping @Sendable (Double, TimeInterval?) -> Void,
        isCancelled: @escaping @Sendable () -> Bool
    ) async throws {
        let asset = AVURLAsset(url: input)
        guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else {
            throw NativeVideoTranscoderError.videoTrackMissing
        }

        try? FileManager.default.removeItem(at: output)
        try FileManager.default.createDirectory(
            at: output.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let reader = try AVAssetReader(asset: asset)
        let writer = try AVAssetWriter(outputURL: output, fileType: .mp4)

        let videoOutput = AVAssetReaderTrackOutput(
            track: videoTrack,
            outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
        )
        videoOutput.alwaysCopiesSampleData = false
        guard reader.canAdd(videoOutput) else {
            throw NativeVideoTranscoderError.readerCannotAddOutput
        }
        reader.add(videoOutput)

        let preferredTransform = try await videoTrack.load(.preferredTransform)
        let naturalSize = try await videoTrack.load(.naturalSize).applying(preferredTransform)
        let width = max(2, Int(abs(naturalSize.width).rounded()))
        let height = max(2, Int(abs(naturalSize.height).rounded()))
        let codec: AVVideoCodecType = encodingPreset.contains("h264") ? .h264 : .hevc
        let videoInput = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: codec,
                AVVideoWidthKey: width,
                AVVideoHeightKey: height
            ]
        )
        videoInput.transform = preferredTransform
        videoInput.expectsMediaDataInRealTime = false
        guard writer.canAdd(videoInput) else {
            throw NativeVideoTranscoderError.writerCannotAddInput
        }
        writer.add(videoInput)

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: videoInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height
            ]
        )

        let audioPair = await makeAudioPair(asset: asset, reader: reader, writer: writer)

        guard writer.startWriting() else {
            throw NativeVideoTranscoderError.writerFailed(writer.error?.localizedDescription ?? "Native writer failed to start.")
        }
        guard reader.startReading() else {
            throw NativeVideoTranscoderError.readerFailed(reader.error?.localizedDescription ?? "Native reader failed to start.")
        }
        writer.startSession(atSourceTime: .zero)

        let durationSeconds = max(CMTimeGetSeconds(try await asset.load(.duration)), 0.001)
        let started = Date()
        let group = DispatchGroup()
        let state = TranscodeState()
        let videoIO = NativeVideoIO(
            output: videoOutput,
            input: videoInput,
            adaptor: adaptor,
            writer: writer
        )

        group.enter()
        videoInput.requestMediaDataWhenReady(on: DispatchQueue(label: "app.lada.native-video")) {
            while videoIO.input.isReadyForMoreMediaData {
                if isCancelled() {
                    state.cancel()
                    videoIO.input.markAsFinished()
                    group.leave()
                    return
                }
                guard let sample = videoIO.output.copyNextSampleBuffer() else {
                    videoIO.input.markAsFinished()
                    group.leave()
                    return
                }
                guard let imageBuffer = CMSampleBufferGetImageBuffer(sample) else {
                    continue
                }
                let timestamp = CMSampleBufferGetPresentationTimeStamp(sample)
                let outputBuffer: CVPixelBuffer
                do {
                    outputBuffer = try frameProcessor?(imageBuffer) ?? imageBuffer
                } catch {
                    state.fail(error)
                    videoIO.input.markAsFinished()
                    group.leave()
                    return
                }
                if videoIO.adaptor.append(outputBuffer, withPresentationTime: timestamp) == false {
                    state.fail(videoIO.writer.error)
                    videoIO.input.markAsFinished()
                    group.leave()
                    return
                }
                let seconds = CMTimeGetSeconds(timestamp)
                let fraction = min(max(seconds / durationSeconds, 0), 0.98)
                let elapsed = Date().timeIntervalSince(started)
                let remaining = fraction > 0 ? max((elapsed / fraction) - elapsed, 0) : nil
                progress(fraction, remaining)
            }
        }

        if let audioPair {
            group.enter()
            audioPair.input.requestMediaDataWhenReady(on: DispatchQueue(label: "app.lada.native-audio")) {
                while audioPair.input.isReadyForMoreMediaData {
                    if isCancelled() {
                        state.cancel()
                        audioPair.input.markAsFinished()
                        group.leave()
                        return
                    }
                    guard let sample = audioPair.output.copyNextSampleBuffer() else {
                        audioPair.input.markAsFinished()
                        group.leave()
                        return
                    }
                    if audioPair.input.append(sample) == false {
                        state.fail(videoIO.writer.error)
                        audioPair.input.markAsFinished()
                        group.leave()
                        return
                    }
                }
            }
        }

        waitForMediaCallbacks(group)
        if state.cancelled {
            reader.cancelReading()
            writer.cancelWriting()
            throw NativeVideoTranscoderError.cancelled
        }
        if let error = state.error {
            reader.cancelReading()
            writer.cancelWriting()
            throw NativeVideoTranscoderError.writerFailed(error.localizedDescription)
        }
        if reader.status == .failed {
            writer.cancelWriting()
            throw NativeVideoTranscoderError.readerFailed(reader.error?.localizedDescription ?? "Native reader failed.")
        }

        finishWriting(writer)

        if writer.status == .failed {
            throw NativeVideoTranscoderError.writerFailed(writer.error?.localizedDescription ?? "Native writer failed.")
        }
        progress(1, 0)
    }

    private static func makeAudioPair(
        asset: AVAsset,
        reader: AVAssetReader,
        writer: AVAssetWriter
    ) async -> NativeAudioIO? {
        guard let audioTrack = try? await asset.loadTracks(withMediaType: .audio).first else {
            return nil
        }
        let output = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: nil)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            return nil
        }

        let sourceFormatHint = try? await audioTrack.load(.formatDescriptions).first
        let input = AVAssetWriterInput(
            mediaType: .audio,
            outputSettings: nil,
            sourceFormatHint: sourceFormatHint
        )
        input.expectsMediaDataInRealTime = false
        guard writer.canAdd(input) else {
            return nil
        }
        reader.add(output)
        writer.add(input)
        return NativeAudioIO(output: output, input: input)
    }

    private static func waitForMediaCallbacks(_ group: DispatchGroup) {
        group.wait()
    }

    private static func finishWriting(_ writer: AVAssetWriter) {
        let semaphore = DispatchSemaphore(value: 0)
        writer.finishWriting {
            semaphore.signal()
        }
        semaphore.wait()
    }
}

private final class NativeVideoIO: @unchecked Sendable {
    let output: AVAssetReaderTrackOutput
    let input: AVAssetWriterInput
    let adaptor: AVAssetWriterInputPixelBufferAdaptor
    let writer: AVAssetWriter

    init(
        output: AVAssetReaderTrackOutput,
        input: AVAssetWriterInput,
        adaptor: AVAssetWriterInputPixelBufferAdaptor,
        writer: AVAssetWriter
    ) {
        self.output = output
        self.input = input
        self.adaptor = adaptor
        self.writer = writer
    }
}

private final class NativeAudioIO: @unchecked Sendable {
    let output: AVAssetReaderTrackOutput
    let input: AVAssetWriterInput

    init(output: AVAssetReaderTrackOutput, input: AVAssetWriterInput) {
        self.output = output
        self.input = input
    }
}

private final class TranscodeState: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var cancelled = false
    private(set) var error: Error?

    func cancel() {
        lock.withLock {
            cancelled = true
        }
    }

    func fail(_ error: Error?) {
        lock.withLock {
            self.error = error ?? NativeVideoTranscoderError.writerFailed("Native writer failed.")
        }
    }
}
