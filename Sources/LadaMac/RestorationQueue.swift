import AppKit
import AVFoundation
import Foundation
import Observation
import UniformTypeIdentifiers

@MainActor
@Observable
final class RestorationQueue {
    var selection: SidebarSection = .queue
    var jobs: [RestorationJob] = []
    var selectedJobID: UUID?
    var isPaused = false
    var isImporterPresented = false
    var previewJobID: UUID?
    var lastError: String?
    var engineStatus: EngineStatus = .checking
    var isProgressDebugLoggingEnabled = true

    private let engine: any RestorationEngine
    private var activeTask: Task<Void, Never>?

    init(engine: any RestorationEngine = RestorationEngineSelection.makeEngine()) {
        self.engine = engine
        Task {
            engineStatus = await engine.probe()
        }

        if AppLaunchMode.engineSmoke {
            let root = Bundle.main.bundleURL
                .deletingLastPathComponent()
                .deletingLastPathComponent()
            let input = root.appendingPathComponent("work/smoke-input.mp4")
            let destination = root.appendingPathComponent("work/app-smoke-output")
            let job = RestorationJob(sourceURL: input, destinationURL: destination)
            job.profile = .fast
            job.outputFormat = .h264
            jobs = [job]
            selectedJobID = job.id
            Task {
                job.metadata = await VideoInspector.metadata(for: input)
            }
            return
        }

        guard ProcessInfo.processInfo.environment["LADA_DEMO_MODE"] == "1" else {
            return
        }
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let assets = root.appendingPathComponent("vendor/lada/assets")
        let destination = root.appendingPathComponent("work/demo-output")
        let demoInputs = [
            assets.appendingPathComponent("screenshot_gui_1_light.png"),
            assets.appendingPathComponent("screenshot_gui_2_light.png"),
            assets.appendingPathComponent("screenshot_view_yolo.png"),
        ]
        jobs = demoInputs.enumerated().map { index, url in
            let job = RestorationJob(sourceURL: url, destinationURL: destination)
            job.metadata = VideoMetadata(
                duration: [204, 171, 250][index],
                dimensions: index == 1
                    ? CGSize(width: 1280, height: 720)
                    : CGSize(width: 1920, height: 1080),
                frameRate: index == 2 ? 24 : 29.97,
                fileSize: Int64([1_200_000_000, 872_000_000, 1_800_000_000][index]),
                codec: "H.264"
            )
            job.profile = [.standard, .fast, .accurate][index]
            if index == 0 {
                job.state = .processing
                job.progress = 0.42
                job.estimatedSecondsRemaining = 131
            } else if index == 2 {
                job.state = .completed
                job.progress = 1
            }
            return job
        }
        selectedJobID = jobs.first?.id
    }

    var selectedJob: RestorationJob? {
        get { jobs.first { $0.id == selectedJobID } }
        set { selectedJobID = newValue?.id }
    }

    var previewJob: RestorationJob? {
        get { jobs.first { $0.id == previewJobID } }
        set { previewJobID = newValue?.id }
    }

    var isProcessing: Bool {
        jobs.contains { $0.state == .processing }
    }

    var waitingCount: Int {
        jobs.count { $0.state == .waiting }
    }

    var completedCount: Int {
        jobs.count { $0.state == .completed }
    }

    func presentImporter() {
        let panel = NSOpenPanel()
        panel.title = "Add Videos"
        panel.prompt = "Add to Queue"
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.movie, .mpeg4Movie, .quickTimeMovie]
        guard panel.runModal() == .OK else { return }
        add(urls: panel.urls)
    }

    func add(urls: [URL]) {
        let movies = FileManager.default.urls(
            for: .moviesDirectory,
            in: .userDomainMask
        ).first ?? FileManager.default.homeDirectoryForCurrentUser
        let destination = movies.appendingPathComponent("Lada", isDirectory: true)

        for url in urls where !jobs.contains(where: { $0.sourceURL == url }) {
            let job = RestorationJob(sourceURL: url, destinationURL: destination)
            jobs.append(job)
            selectedJobID = job.id
            Task {
                job.metadata = await VideoInspector.metadata(for: url)
            }
        }
    }

    func startQueue() {
        guard case .ready = engineStatus else {
            if case .unavailable(let reason) = engineStatus {
                lastError = "The restoration engine is unavailable: \(reason)"
            }
            return
        }
        guard activeTask == nil else {
            if isPaused {
                isPaused = false
                engine.resume()
                jobs.first(where: { $0.state == .paused })?.state = .processing
            }
            return
        }
        isPaused = false
        activeTask = Task {
            while let job = jobs.first(where: { $0.state == .waiting }) {
                if Task.isCancelled { break }
                while isPaused {
                    try? await Task.sleep(for: .milliseconds(200))
                }
                await process(job)
            }
            activeTask = nil
        }
    }

    func togglePause() {
        isPaused.toggle()
        if let current = jobs.first(where: { $0.state == .processing }) {
            engine.pause()
            current.state = .paused
        } else if let current = jobs.first(where: { $0.state == .paused }) {
            engine.resume()
            current.state = .processing
        }
    }

    func cancel(_ job: RestorationJob) {
        if job.state == .processing || job.state == .paused {
            engine.cancel()
            activeTask?.cancel()
            activeTask = nil
        }
        job.state = .waiting
        job.progress = 0
        job.estimatedSecondsRemaining = nil
        try? FileManager.default.removeItem(at: job.outputURL)
    }

    func shutdown() {
        engine.cancel()
        activeTask?.cancel()
        activeTask = nil
    }

    func remove(_ job: RestorationJob) {
        if job.state == .processing || job.state == .paused {
            cancel(job)
        }
        jobs.removeAll { $0.id == job.id }
        if selectedJobID == job.id {
            selectedJobID = jobs.first?.id
        }
    }

    func changeDestination(for job: RestorationJob) {
        let panel = NSOpenPanel()
        panel.title = "Choose Output Folder"
        panel.prompt = "Choose"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        job.destinationURL = url
    }

    func preview(_ job: RestorationJob) {
        guard case .completed = job.state else {
            lastError = "Preview is available after restoration completes."
            return
        }
        guard FileManager.default.fileExists(atPath: job.outputURL.path) else {
            lastError = "The restored video could not be found. It may have been moved or deleted."
            return
        }
        previewJobID = job.id
    }

    private func process(_ job: RestorationJob) async {
        job.state = .processing
        selectedJobID = job.id
        let jobID = job.id
        let sourceURL = job.sourceURL
        let outputURL = job.outputURL
        let debugLogger = makeProgressDebugLogger(for: job)
        let jobStartedAt = Date()
        let heartbeatTask = makeProgressDebugHeartbeat(
            logger: debugLogger,
            jobID: jobID,
            source: sourceURL,
            output: outputURL,
            startedAt: jobStartedAt
        )
        defer {
            heartbeatTask?.cancel()
            debugLogger?.close()
        }
        do {
            try FileManager.default.createDirectory(
                at: job.destinationURL,
                withIntermediateDirectories: true
            )
            let request = RestorationRequest(
                input: job.sourceURL,
                output: job.outputURL,
                device: "mps",
                detectionModel: job.profile.detectionModel,
                maxClipLength: job.profile.maxClipLength(memoryMode: job.memoryMode),
                memoryMode: job.memoryMode.rawValue,
                encodingPreset: job.outputFormat.encodingPreset(quality: job.quality)
            )
            let progressStabilizer = RestorationProgressStabilizerBox()
            let progressUpdateThrottler = ProgressUpdateThrottler()
            debugLogger?.log(ProgressDebugEvent(
                timestamp: jobStartedAt,
                event: "started",
                jobID: jobID,
                source: sourceURL,
                output: outputURL,
                note: "Progress debug logging enabled"
            ))
            try await engine.restore(request: request, progress: { progress, remaining in
                let now = Date()
                let stableUpdate = progressStabilizer.update(
                    rawProgress: progress,
                    rawRemaining: remaining,
                    now: now
                )
                guard progressUpdateThrottler.shouldEmit(
                    progress: stableUpdate.progress,
                    now: now
                ) else {
                    return
                }
                debugLogger?.log(ProgressDebugEvent(
                    timestamp: now,
                    event: "progress",
                    jobID: jobID,
                    source: sourceURL,
                    output: outputURL,
                    rawProgress: progress,
                    stableProgress: stableUpdate.progress,
                    rawRemainingSeconds: remaining,
                    stableRemainingSeconds: stableUpdate.estimatedSecondsRemaining,
                    elapsedSeconds: now.timeIntervalSince(jobStartedAt),
                    note: progress < stableUpdate.progress
                        ? "Raw worker progress moved backwards; UI kept stable progress."
                        : nil
                ))
                Task { @MainActor [weak self] in
                    guard let currentJob = self?.jobs.first(where: { $0.id == jobID }) else {
                        return
                    }
                    currentJob.progress = stableUpdate.progress
                    currentJob.estimatedSecondsRemaining = stableUpdate.estimatedSecondsRemaining
                }
            }, diagnostic: { diagnostic in
                debugLogger?.log(ProgressDebugEvent(
                    event: diagnostic.event,
                    jobID: jobID,
                    source: sourceURL,
                    output: outputURL,
                    elapsedSeconds: Date().timeIntervalSince(jobStartedAt),
                    stage: diagnostic.stage,
                    phase: diagnostic.phase,
                    framesProcessed: diagnostic.framesProcessed,
                    totalFrames: diagnostic.totalFrames,
                    windowFrames: diagnostic.windowFrames,
                    windowSeconds: diagnostic.windowSeconds,
                    framesPerSecond: diagnostic.framesPerSecond,
                    progressPercentPerHour: diagnostic.progressPercentPerHour,
                    waitForRestoredFrameSeconds: diagnostic.waitForRestoredFrameSeconds,
                    waitForRestoredFramePercent: diagnostic.waitForRestoredFramePercent,
                    videoWriteSeconds: diagnostic.videoWriteSeconds,
                    videoWritePercent: diagnostic.videoWritePercent,
                    inferenceSeconds: diagnostic.inferenceSeconds,
                    inferencePercent: diagnostic.inferencePercent,
                    clipID: diagnostic.clipID,
                    frameStart: diagnostic.frameStart,
                    frameEnd: diagnostic.frameEnd,
                    clipLength: diagnostic.clipLength,
                    durationSeconds: diagnostic.durationSeconds,
                    detectionsInLastBatch: diagnostic.detectionsInLastBatch,
                    detectionsInWindow: diagnostic.detectionsInWindow,
                    originalDetections: diagnostic.originalDetections,
                    droppedDetections: diagnostic.droppedDetections,
                    maxDetectionsPerFrame: diagnostic.maxDetectionsPerFrame,
                    minDetectionConfidence: diagnostic.minDetectionConfidence,
                    clipsRestored: diagnostic.clipsRestored,
                    clipFramesRestored: diagnostic.clipFramesRestored,
                    clipFramesPerSecond: diagnostic.clipFramesPerSecond,
                    clipsPerMinute: diagnostic.clipsPerMinute,
                    restoreSeconds: diagnostic.restoreSeconds,
                    restorePercent: diagnostic.restorePercent,
                    averageClipSeconds: diagnostic.averageClipSeconds,
                    note: diagnostic.message
                ))
            }
            )
            if !Task.isCancelled {
                job.progress = 1
                job.estimatedSecondsRemaining = nil
                job.state = .completed
                debugLogger?.log(ProgressDebugEvent(
                    event: "completed",
                    jobID: jobID,
                    source: sourceURL,
                    output: outputURL
                ))
            }
        } catch is CancellationError {
            debugLogger?.log(ProgressDebugEvent(
                event: "cancelled",
                jobID: jobID,
                source: sourceURL,
                output: outputURL
            ))
            job.state = .waiting
        } catch {
            if Task.isCancelled {
                debugLogger?.log(ProgressDebugEvent(
                    event: "cancelled",
                    jobID: jobID,
                    source: sourceURL,
                    output: outputURL
                ))
                job.state = .waiting
            } else {
                debugLogger?.log(ProgressDebugEvent(
                    event: "failed",
                    jobID: jobID,
                    source: sourceURL,
                    output: outputURL,
                    note: error.localizedDescription
                ))
                job.state = .failed(error.localizedDescription)
                lastError = error.localizedDescription
            }
        }
    }

    private func makeProgressDebugHeartbeat(
        logger: ProgressDebugLogger?,
        jobID: UUID,
        source: URL,
        output: URL,
        startedAt: Date
    ) -> Task<Void, Never>? {
        guard let logger else {
            return nil
        }

        return Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(60))
                if Task.isCancelled {
                    break
                }
                guard let currentJob = self?.jobs.first(where: { $0.id == jobID }),
                      currentJob.state == .processing || currentJob.state == .paused
                else {
                    break
                }
                logger.log(ProgressDebugEvent(
                    event: "heartbeat",
                    jobID: jobID,
                    source: source,
                    output: output,
                    stableProgress: currentJob.progress,
                    stableRemainingSeconds: currentJob.estimatedSecondsRemaining,
                    elapsedSeconds: Date().timeIntervalSince(startedAt),
                    note: "Queue is still alive; use this to detect backend stalls between progress events."
                ))
            }
        }
    }

    private func makeProgressDebugLogger(for job: RestorationJob) -> ProgressDebugLogger? {
        guard isProgressDebugLoggingEnabled else {
            job.progressDebugLogURL = nil
            return nil
        }
        let url = job.defaultProgressDebugLogURL
        job.progressDebugLogURL = url
        return ProgressDebugLogger(url: url)
    }
}

enum VideoInspector {
    static func metadata(for url: URL) async -> VideoMetadata {
        let asset = AVURLAsset(url: url)
        do {
            let duration = try await asset.load(.duration).seconds
            let tracks = try await asset.loadTracks(withMediaType: .video)
            let track = tracks.first
            let size = try await track?.load(.naturalSize) ?? .zero
            let transform = try await track?.load(.preferredTransform) ?? .identity
            let transformedSize = size.applying(transform)
            let rate = try await track?.load(.nominalFrameRate) ?? 0
            let values = try url.resourceValues(forKeys: [.fileSizeKey])
            let descriptions = try await track?.load(.formatDescriptions) ?? []
            let codec = descriptions.first.map { description in
                let subtype = CMFormatDescriptionGetMediaSubType(description)
                return String(format: "%c%c%c%c",
                    (subtype >> 24) & 0xff,
                    (subtype >> 16) & 0xff,
                    (subtype >> 8) & 0xff,
                    subtype & 0xff
                ).trimmingCharacters(in: .whitespaces)
            } ?? "—"
            return VideoMetadata(
                duration: duration,
                dimensions: CGSize(
                    width: abs(transformedSize.width),
                    height: abs(transformedSize.height)
                ),
                frameRate: Double(rate),
                fileSize: Int64(values.fileSize ?? 0),
                codec: codec
            )
        } catch {
            return VideoMetadata()
        }
    }
}
