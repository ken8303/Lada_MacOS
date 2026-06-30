import Foundation

protocol RestorationEngine: Sendable {
    func restore(
        request: RestorationRequest,
        progress: @escaping @Sendable (Double, TimeInterval?) -> Void,
        diagnostic: @escaping @Sendable (RestorationEngineDiagnostic) -> Void
    ) async throws
    func probe() async -> EngineStatus
    func cancel()
    func pause()
    func resume()
}

struct RestorationEngineDiagnostic: Sendable, Equatable {
    let event: String
    let message: String?
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

    init(
        event: String,
        message: String? = nil,
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
        averageClipSeconds: TimeInterval? = nil
    ) {
        self.event = event
        self.message = message
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
    }
}

enum LadaEngineError: LocalizedError {
    case workerMissing
    case invalidWorkerOutput
    case failed(String)

    var errorDescription: String? {
        switch self {
        case .workerMissing:
            "The bundled Lada worker could not be found."
        case .invalidWorkerOutput:
            "The Lada worker returned an unreadable response."
        case .failed(let message):
            message
        }
    }
}

final class PythonLadaEngine: RestorationEngine, @unchecked Sendable {
    private let lock = NSLock()
    private var process: Process?
    private var workerInput: FileHandle?

    func restore(
        request: RestorationRequest,
        progress: @escaping @Sendable (Double, TimeInterval?) -> Void,
        diagnostic: @escaping @Sendable (RestorationEngineDiagnostic) -> Void
    ) async throws {
        guard let workerURL = workerURL() else {
            throw LadaEngineError.workerMissing
        }

        let workerRequest = WorkerRequest(
            input: request.input.path,
            output: request.output.path,
            device: request.device,
            detectionModel: request.detectionModel,
            maxClipLength: request.maxClipLength,
            memoryMode: request.memoryMode,
            encodingPreset: request.encodingPreset,
            simulateWhenUnavailable: true
        )
        let inputData = try JSONEncoder().encode(workerRequest)

        let task = Process()
        let stdout = Pipe()
        let stdin = Pipe()
        let stderr = Pipe()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        task.arguments = [workerURL.path]
        task.standardInput = stdin
        task.standardOutput = stdout
        task.standardError = stderr
        task.environment = workerEnvironment()

        lock.withLock {
            process = task
            workerInput = stdin.fileHandleForWriting
        }
        defer {
            lock.withLock {
                process = nil
                workerInput = nil
            }
        }

        try task.run()
        stdin.fileHandleForWriting.write(inputData)
        stdin.fileHandleForWriting.write(Data([0x0A]))

        var recentLogs: [String] = []
        for try await line in stdout.fileHandleForReading.bytes.lines {
            guard let data = line.data(using: .utf8),
                  let event = try? JSONDecoder().decode(WorkerEvent.self, from: data)
            else { continue }

            switch event.type {
            case "progress":
                progress(event.progress ?? 0, event.remainingSeconds)
            case "started":
                diagnostic(RestorationEngineDiagnostic(
                    event: "worker-started",
                    message: event.diagnosticMessage
                ))
            case "log":
                if let message = event.message, !message.isEmpty {
                    recentLogs.append(message)
                    if recentLogs.count > 20 {
                        recentLogs.removeFirst()
                    }
                    diagnostic(RestorationEngineDiagnostic(
                        event: "worker-log",
                        message: message
                    ))
                }
            case "diagnostic":
                diagnostic(RestorationEngineDiagnostic(
                    event: "worker-diagnostic",
                    message: event.diagnosticMessage,
                    stage: event.stage,
                    phase: event.phase,
                    framesProcessed: event.framesProcessed,
                    totalFrames: event.totalFrames,
                    windowFrames: event.windowFrames,
                    windowSeconds: event.windowSeconds,
                    framesPerSecond: event.framesPerSecond,
                    progressPercentPerHour: event.progressPercentPerHour,
                    waitForRestoredFrameSeconds: event.waitForRestoredFrameSeconds,
                    waitForRestoredFramePercent: event.waitForRestoredFramePercent,
                    videoWriteSeconds: event.videoWriteSeconds,
                    videoWritePercent: event.videoWritePercent,
                    inferenceSeconds: event.inferenceSeconds,
                    inferencePercent: event.inferencePercent,
                    clipID: event.clipID,
                    frameStart: event.frameStart,
                    frameEnd: event.frameEnd,
                    clipLength: event.clipLength,
                    durationSeconds: event.durationSeconds,
                    detectionsInLastBatch: event.detectionsInLastBatch,
                    detectionsInWindow: event.detectionsInWindow,
                    originalDetections: event.originalDetections,
                    droppedDetections: event.droppedDetections,
                    maxDetectionsPerFrame: event.maxDetectionsPerFrame,
                    minDetectionConfidence: event.minDetectionConfidence,
                    clipsRestored: event.clipsRestored,
                    clipFramesRestored: event.clipFramesRestored,
                    clipFramesPerSecond: event.clipFramesPerSecond,
                    clipsPerMinute: event.clipsPerMinute,
                    restoreSeconds: event.restoreSeconds,
                    restorePercent: event.restorePercent,
                    averageClipSeconds: event.averageClipSeconds
                ))
            case "error":
                let details = recentLogs.suffix(6).joined(separator: "\n")
                let message = event.message ?? "Restoration failed."
                diagnostic(RestorationEngineDiagnostic(
                    event: "worker-error",
                    message: details.isEmpty ? message : "\(message)\n\(details)"
                ))
                throw LadaEngineError.failed(
                    details.isEmpty ? message : "\(message)\n\n\(details)"
                )
            case "cancelled":
                diagnostic(RestorationEngineDiagnostic(event: "worker-cancelled"))
                throw CancellationError()
            case "completed":
                diagnostic(RestorationEngineDiagnostic(
                    event: "worker-completed",
                    message: event.diagnosticMessage
                ))
                break
            default:
                continue
            }
        }

        try? stdin.fileHandleForWriting.close()
        task.waitUntilExit()
        if Task.isCancelled {
            throw CancellationError()
        }
        if task.terminationStatus != 0 {
            let data = stderr.fileHandleForReading.readDataToEndOfFile()
            let message = String(data: data, encoding: .utf8) ?? "Lada worker failed."
            throw LadaEngineError.failed(message)
        }
    }

    func probe() async -> EngineStatus {
        let metal = MetalCapabilities.current
        guard metal.isMetal4Ready else {
            return .unavailable(metal.statusDetail)
        }

        guard let workerURL = workerURL() else {
            return .unavailable("Worker missing")
        }

        let task = Process()
        let stdout = Pipe()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        task.arguments = [workerURL.path, "--probe"]
        task.standardOutput = stdout
        task.standardError = stdout
        task.environment = workerEnvironment()

        do {
            try task.run()
            let data = stdout.fileHandleForReading.readDataToEndOfFile()
            task.waitUntilExit()
            let lines = String(data: data, encoding: .utf8)?
                .split(separator: "\n")
                .reversed() ?? []
            for line in lines {
                guard let eventData = line.data(using: .utf8),
                      let event = try? JSONDecoder().decode(WorkerEvent.self, from: eventData),
                      event.type == "readiness"
                else { continue }
                if event.ready == true {
                    let workerMessage = event.message ?? "MPS and models available"
                    return .ready("\(metal.statusDetail) · \(workerMessage)")
                }
                return .unavailable(event.message ?? "Runtime check failed")
            }
            return .unavailable("Runtime returned no health result")
        } catch {
            return .unavailable(error.localizedDescription)
        }
    }

    func cancel() {
        sendControl("cancel")
    }

    func pause() {
        sendControl("pause")
    }

    func resume() {
        sendControl("resume")
    }

    private func sendControl(_ command: String) {
        lock.withLock {
            guard let workerInput else { return }
            let payload = Data("{\"command\":\"\(command)\"}\n".utf8)
            do {
                try workerInput.write(contentsOf: payload)
            } catch {
                process?.interrupt()
            }
        }
    }

    private func workerEnvironment() -> [String: String] {
        var environment = ProcessInfo.processInfo.environment
        if let bundledRoot = Bundle.main.resourceURL?.appendingPathComponent("lada"),
           FileManager.default.fileExists(
               atPath: bundledRoot.appendingPathComponent("lada/__init__.py").path
           ) {
            environment["LADA_SOURCE_ROOT"] = bundledRoot.path
        } else if let sourceRoot = developmentSourceRoot() {
            environment["LADA_SOURCE_ROOT"] = sourceRoot.path
        }
        if let resources = Bundle.main.resourceURL {
            let python = resources.appendingPathComponent("runtime/python/bin/python3.12")
            let sitePackages = resources.appendingPathComponent("runtime/site-packages")
            if FileManager.default.isExecutableFile(atPath: python.path) {
                environment["LADA_PYTHON"] = python.path
                environment["LADA_SITE_PACKAGES"] = sitePackages.path
            }
        }
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        if environment["LADA_EXPERIMENTAL_RAW_METAL"] == "1" {
            environment["PYTORCH_MPS_PREFER_METAL"] = "1"
        } else {
            environment.removeValue(forKey: "PYTORCH_MPS_PREFER_METAL")
        }
        return environment
    }

    private func developmentSourceRoot() -> URL? {
        let current = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("vendor/lada")
        let besideOutputs = Bundle.main.bundleURL
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("vendor/lada")
        return [current, besideOutputs].first {
            FileManager.default.fileExists(
                atPath: $0.appendingPathComponent("pyproject.toml").path
            )
        }
    }

    private func workerURL() -> URL? {
        if let bundled = Bundle.main.url(
            forResource: "lada_worker",
            withExtension: "py"
        ) {
            return bundled
        }
        let developmentPath = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath
        )
        .appendingPathComponent("Sources/LadaMac/Resources/lada_worker.py")
        return FileManager.default.fileExists(atPath: developmentPath.path)
            ? developmentPath
            : nil
    }
}

struct RestorationRequest: Sendable {
    let input: URL
    let output: URL
    let device: String
    let detectionModel: String
    let maxClipLength: Int
    let memoryMode: String
    let encodingPreset: String
}

private struct WorkerRequest: Encodable {
    let input: String
    let output: String
    let device: String
    let detectionModel: String
    let maxClipLength: Int
    let memoryMode: String
    let encodingPreset: String
    let simulateWhenUnavailable: Bool
}

private struct WorkerEvent: Decodable {
    let type: String
    let progress: Double?
    let remainingSeconds: Double?
    let message: String?
    let note: String?
    let ready: Bool?
    let command: [String]?
    let device: String?
    let output: String?
    let inProgressOutput: String?
    let simulated: Bool?
    let fallbackDevice: String?
    let maxClipLength: Int?
    let stage: String?
    let phase: String?
    let framesProcessed: Int?
    let totalFrames: Int?
    let windowFrames: Int?
    let windowSeconds: Double?
    let framesPerSecond: Double?
    let progressPercentPerHour: Double?
    let waitForRestoredFrameSeconds: Double?
    let waitForRestoredFramePercent: Double?
    let videoWriteSeconds: Double?
    let videoWritePercent: Double?
    let inferenceSeconds: Double?
    let inferencePercent: Double?
    let clipID: Int?
    let frameStart: Int?
    let frameEnd: Int?
    let clipLength: Int?
    let durationSeconds: Double?
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
    let restoreSeconds: Double?
    let restorePercent: Double?
    let averageClipSeconds: Double?

    var diagnosticMessage: String? {
        var fields: [String] = []
        if let message, !message.isEmpty {
            fields.append(message)
        }
        if let note, !note.isEmpty {
            fields.append(note)
        }
        if let stage {
            fields.append("stage=\(stage)")
        }
        if let phase {
            fields.append("phase=\(phase)")
        }
        if let device {
            fields.append("device=\(device)")
        }
        if let fallbackDevice {
            fields.append("fallbackDevice=\(fallbackDevice)")
        }
        if let maxClipLength {
            fields.append("maxClipLength=\(maxClipLength)")
        }
        if let simulated {
            fields.append("simulated=\(simulated)")
        }
        if let output {
            fields.append("output=\(output)")
        }
        if let inProgressOutput {
            fields.append("inProgressOutput=\(inProgressOutput)")
        }
        if let command {
            fields.append("command=\(command.joined(separator: " "))")
        }
        if let framesPerSecond {
            fields.append(String(format: "fps=%.2f", framesPerSecond))
        }
        if let clipFramesPerSecond {
            fields.append(String(format: "clipFPS=%.2f", clipFramesPerSecond))
        }
        if let clipsPerMinute {
            fields.append(String(format: "clipsPerMinute=%.1f", clipsPerMinute))
        }
        if let progressPercentPerHour {
            fields.append(String(format: "progressPerHour=%.2f%%", progressPercentPerHour))
        }
        if let clipID {
            fields.append("clipID=\(clipID)")
        }
        if let originalDetections {
            fields.append("originalDetections=\(originalDetections)")
        }
        if let detectionsInLastBatch {
            fields.append("keptDetections=\(detectionsInLastBatch)")
        }
        if let droppedDetections {
            fields.append("droppedDetections=\(droppedDetections)")
        }
        if let maxDetectionsPerFrame {
            fields.append("maxDetectionsPerFrame=\(maxDetectionsPerFrame)")
        }
        if let minDetectionConfidence {
            fields.append(String(format: "minDetectionConfidence=%.2f", minDetectionConfidence))
        }
        return fields.isEmpty ? nil : fields.joined(separator: " · ")
    }
}

typealias LadaEngine = PythonLadaEngine
