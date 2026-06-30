import AVFoundation
import Foundation
import Observation
import UniformTypeIdentifiers

enum SidebarSection: String, CaseIterable, Identifiable {
    case queue = "Queue"
    case history = "History"
    case models = "Models"
    case settings = "Settings"

    var id: Self { self }

    var systemImage: String {
        switch self {
        case .queue: "list.bullet"
        case .history: "clock"
        case .models: "cpu"
        case .settings: "gearshape"
        }
    }
}

enum RestorationProfile: String, CaseIterable, Codable, Identifiable {
    case fast = "Fast"
    case standard = "Standard"
    case accurate = "Accurate"

    var id: Self { self }

    var detectionModel: String {
        switch self {
        case .fast, .standard: "v4-fast"
        case .accurate: "v4-accurate"
        }
    }

    var maxClipLength: Int {
        maxClipLength(memoryMode: .automatic)
    }

    func maxClipLength(memoryMode: MemoryMode) -> Int {
        switch self {
        case .fast:
            switch memoryMode {
            case .longVideo: 30
            case .conservative: 30
            case .automatic: 45
            case .performance: 60
            }
        case .standard:
            switch memoryMode {
            case .longVideo: 45
            case .conservative: 45
            case .automatic: 60
            case .performance: 75
            }
        case .accurate:
            switch memoryMode {
            case .longVideo: 60
            case .conservative: 60
            case .automatic: 90
            case .performance: 90
            }
        }
    }
}

enum QualityPreset: String, CaseIterable, Codable, Identifiable {
    case balanced = "Balanced"
    case high = "High (Recommended)"
    case maximum = "Maximum"

    var id: Self { self }
}

enum MemoryMode: String, CaseIterable, Codable, Identifiable {
    case longVideo = "Long Video"
    case automatic = "Auto (Unified Memory)"
    case conservative = "Conservative"
    case performance = "Performance"

    var id: Self { self }
}

enum OutputFormat: String, CaseIterable, Codable, Identifiable {
    case hevc = "MP4 (H.265)"
    case h264 = "MP4 (H.264)"

    var id: Self { self }

    var encodingPreset: String {
        encodingPreset(quality: .high)
    }

    func encodingPreset(quality: QualityPreset) -> String {
        let suffix = switch quality {
        case .balanced: "fast"
        case .high: "balanced"
        case .maximum: "hq"
        }
        return switch self {
        case .hevc: "hevc-apple-gpu-\(suffix)"
        case .h264: "h264-apple-gpu-\(suffix)"
        }
    }
}

enum JobState: Equatable {
    case waiting
    case processing
    case paused
    case completed
    case failed(String)

    var title: String {
        switch self {
        case .waiting: "Waiting"
        case .processing: "Processing"
        case .paused: "Paused"
        case .completed: "Completed"
        case .failed: "Failed"
        }
    }
}

enum EngineStatus: Equatable {
    case checking
    case ready(String)
    case unavailable(String)

    var title: String {
        switch self {
        case .checking: "Checking engine…"
        case .ready: "Metal 4 ready"
        case .unavailable: "Engine unavailable"
        }
    }

    var detail: String {
        switch self {
        case .checking: "Verifying bundled runtime"
        case .ready(let detail), .unavailable(let detail): detail
        }
    }
}

struct VideoMetadata: Equatable {
    var duration: TimeInterval = 0
    var dimensions: CGSize = .zero
    var frameRate: Double = 0
    var fileSize: Int64 = 0
    var codec: String = "—"
}

@Observable
final class RestorationJob: Identifiable {
    let id = UUID()
    let sourceURL: URL
    let addedAt = Date()
    var metadata = VideoMetadata()
    var profile: RestorationProfile = .standard
    var quality: QualityPreset = .balanced
    var memoryMode: MemoryMode = .longVideo
    var outputFormat: OutputFormat = .hevc
    var destinationURL: URL
    var state: JobState = .waiting
    var progress: Double = 0
    var estimatedSecondsRemaining: TimeInterval?
    var progressDebugLogURL: URL?

    init(sourceURL: URL, destinationURL: URL) {
        self.sourceURL = sourceURL
        self.destinationURL = destinationURL
    }

    var displayName: String { sourceURL.lastPathComponent }

    var outputURL: URL {
        destinationURL
            .appendingPathComponent(sourceURL.deletingPathExtension().lastPathComponent + ".restored")
            .appendingPathExtension("mp4")
    }

    var defaultProgressDebugLogURL: URL {
        destinationURL
            .appendingPathComponent(sourceURL.deletingPathExtension().lastPathComponent + ".progress-debug")
            .appendingPathExtension("jsonl")
    }
}
