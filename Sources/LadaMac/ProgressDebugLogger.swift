import Foundation

final class ProgressDebugLogger: @unchecked Sendable {
    private let lock = NSLock()
    private let url: URL
    private var fileHandle: FileHandle?

    init?(url: URL) {
        self.url = url
        let directory = url.deletingLastPathComponent()
        do {
            try FileManager.default.createDirectory(
                at: directory,
                withIntermediateDirectories: true
            )
            FileManager.default.createFile(atPath: url.path, contents: nil)
            fileHandle = try FileHandle(forWritingTo: url)
        } catch {
            return nil
        }
    }

    deinit {
        close()
    }

    func log(_ event: ProgressDebugEvent) {
        lock.withLock {
            guard let fileHandle,
                  let data = try? JSONEncoder.progressDebug.encode(event)
            else {
                return
            }
            fileHandle.write(data)
            fileHandle.write(Data([0x0A]))
        }
    }

    func close() {
        lock.withLock {
            try? fileHandle?.close()
            fileHandle = nil
        }
    }
}

struct ProgressDebugEvent: Encodable {
    let timestamp: Date
    let event: String
    let jobID: String
    let source: String
    let output: String
    let rawProgress: Double?
    let stableProgress: Double?
    let rawRemainingSeconds: TimeInterval?
    let stableRemainingSeconds: TimeInterval?
    let elapsedSeconds: TimeInterval?
    let stage: String?
    let phase: String?
    let framesProcessed: Int?
    let totalFrames: Int?
    let windowFrames: Int?
    let windowSeconds: TimeInterval?
    let framesPerSecond: Double?
    let progressPercentPerHour: Double?
    let waitForRestoredFrameSeconds: TimeInterval?
    let waitForRestoredFramePercent: Double?
    let videoWriteSeconds: TimeInterval?
    let videoWritePercent: Double?
    let inferenceSeconds: TimeInterval?
    let inferencePercent: Double?
    let clipID: Int?
    let frameStart: Int?
    let frameEnd: Int?
    let clipLength: Int?
    let durationSeconds: TimeInterval?
    let detectionsInLastBatch: Int?
    let detectionsInWindow: Int?
    let originalDetections: Int?
    let droppedDetections: Int?
    let maxDetectionsPerFrame: Int?
    let minDetectionConfidence: Double?
    let clipsRestored: Int?
    let clipFramesRestored: Int?
    let clipFramesPerSecond: Double?
    let clipsPerMinute: Double?
    let restoreSeconds: TimeInterval?
    let restorePercent: Double?
    let averageClipSeconds: TimeInterval?
    let note: String?

    init(
        timestamp: Date = Date(),
        event: String,
        jobID: UUID,
        source: URL,
        output: URL,
        rawProgress: Double? = nil,
        stableProgress: Double? = nil,
        rawRemainingSeconds: TimeInterval? = nil,
        stableRemainingSeconds: TimeInterval? = nil,
        elapsedSeconds: TimeInterval? = nil,
        stage: String? = nil,
        phase: String? = nil,
        framesProcessed: Int? = nil,
        totalFrames: Int? = nil,
        windowFrames: Int? = nil,
        windowSeconds: TimeInterval? = nil,
        framesPerSecond: Double? = nil,
        progressPercentPerHour: Double? = nil,
        waitForRestoredFrameSeconds: TimeInterval? = nil,
        waitForRestoredFramePercent: Double? = nil,
        videoWriteSeconds: TimeInterval? = nil,
        videoWritePercent: Double? = nil,
        inferenceSeconds: TimeInterval? = nil,
        inferencePercent: Double? = nil,
        clipID: Int? = nil,
        frameStart: Int? = nil,
        frameEnd: Int? = nil,
        clipLength: Int? = nil,
        durationSeconds: TimeInterval? = nil,
        detectionsInLastBatch: Int? = nil,
        detectionsInWindow: Int? = nil,
        originalDetections: Int? = nil,
        droppedDetections: Int? = nil,
        maxDetectionsPerFrame: Int? = nil,
        minDetectionConfidence: Double? = nil,
        clipsRestored: Int? = nil,
        clipFramesRestored: Int? = nil,
        clipFramesPerSecond: Double? = nil,
        clipsPerMinute: Double? = nil,
        restoreSeconds: TimeInterval? = nil,
        restorePercent: Double? = nil,
        averageClipSeconds: TimeInterval? = nil,
        note: String? = nil
    ) {
        self.timestamp = timestamp
        self.event = event
        self.jobID = jobID.uuidString
        self.source = source.path
        self.output = output.path
        self.rawProgress = rawProgress
        self.stableProgress = stableProgress
        self.rawRemainingSeconds = rawRemainingSeconds
        self.stableRemainingSeconds = stableRemainingSeconds
        self.elapsedSeconds = elapsedSeconds
        self.stage = stage
        self.phase = phase
        self.framesProcessed = framesProcessed
        self.totalFrames = totalFrames
        self.windowFrames = windowFrames
        self.windowSeconds = windowSeconds
        self.framesPerSecond = framesPerSecond
        self.progressPercentPerHour = progressPercentPerHour
        self.waitForRestoredFrameSeconds = waitForRestoredFrameSeconds
        self.waitForRestoredFramePercent = waitForRestoredFramePercent
        self.videoWriteSeconds = videoWriteSeconds
        self.videoWritePercent = videoWritePercent
        self.inferenceSeconds = inferenceSeconds
        self.inferencePercent = inferencePercent
        self.clipID = clipID
        self.frameStart = frameStart
        self.frameEnd = frameEnd
        self.clipLength = clipLength
        self.durationSeconds = durationSeconds
        self.detectionsInLastBatch = detectionsInLastBatch
        self.detectionsInWindow = detectionsInWindow
        self.originalDetections = originalDetections
        self.droppedDetections = droppedDetections
        self.maxDetectionsPerFrame = maxDetectionsPerFrame
        self.minDetectionConfidence = minDetectionConfidence
        self.clipsRestored = clipsRestored
        self.clipFramesRestored = clipFramesRestored
        self.clipFramesPerSecond = clipFramesPerSecond
        self.clipsPerMinute = clipsPerMinute
        self.restoreSeconds = restoreSeconds
        self.restorePercent = restorePercent
        self.averageClipSeconds = averageClipSeconds
        self.note = note
    }
}

private extension JSONEncoder {
    static var progressDebug: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }
}
