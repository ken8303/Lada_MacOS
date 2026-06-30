import Foundation

struct StableProgressUpdate: Equatable {
    let progress: Double
    let estimatedSecondsRemaining: TimeInterval?
}

struct RestorationProgressStabilizer {
    private static let staleProgressInterval: TimeInterval = 5 * 60

    private let startedAt: Date
    private var bestProgress: Double = 0
    private var lastProgressAdvancedAt: Date

    init(startedAt: Date = Date()) {
        self.startedAt = startedAt
        self.lastProgressAdvancedAt = startedAt
    }

    mutating func update(
        rawProgress: Double,
        rawRemaining: TimeInterval?,
        now: Date = Date()
    ) -> StableProgressUpdate {
        let clampedProgress = rawProgress.isFinite
            ? min(max(rawProgress, 0), 1)
            : bestProgress
        let previousBestProgress = bestProgress
        bestProgress = max(bestProgress, clampedProgress)
        if bestProgress > previousBestProgress + 0.000_001 {
            lastProgressAdvancedAt = now
        }

        let elapsed = max(now.timeIntervalSince(startedAt), 0)
        let computedRemaining = Self.remainingTime(
            elapsed: elapsed,
            progress: bestProgress
        )
        let fallbackRemaining = Self.cleanRemaining(rawRemaining)
        let candidateRemaining = computedRemaining ?? fallbackRemaining
        let realisticRemaining = Self.realisticRemaining(
            elapsedAverage: computedRemaining,
            currentWorkerEstimate: fallbackRemaining,
            candidate: candidateRemaining
        )
        let progressIsStale = now.timeIntervalSince(lastProgressAdvancedAt) >= Self.staleProgressInterval

        return StableProgressUpdate(
            progress: bestProgress,
            estimatedSecondsRemaining: progressIsStale ? nil : realisticRemaining
        )
    }

    private static func remainingTime(
        elapsed: TimeInterval,
        progress: Double
    ) -> TimeInterval? {
        guard elapsed >= 5,
              progress > 0.001,
              progress < 1
        else {
            return nil
        }
        return elapsed * (1 - progress) / progress
    }

    private static func cleanRemaining(_ remaining: TimeInterval?) -> TimeInterval? {
        guard let remaining,
              remaining.isFinite,
              remaining >= 0,
              remaining <= 12 * 3600
        else {
            return nil
        }
        return remaining
    }

    private static func realisticRemaining(
        elapsedAverage: TimeInterval?,
        currentWorkerEstimate: TimeInterval?,
        candidate: TimeInterval?
    ) -> TimeInterval? {
        guard let elapsedAverage else {
            return currentWorkerEstimate ?? candidate
        }
        guard let currentWorkerEstimate else {
            return elapsedAverage
        }
        return max(elapsedAverage, currentWorkerEstimate)
    }
}

final class RestorationProgressStabilizerBox: @unchecked Sendable {
    private let lock = NSLock()
    private var stabilizer = RestorationProgressStabilizer()

    func update(
        rawProgress: Double,
        rawRemaining: TimeInterval?,
        now: Date = Date()
    ) -> StableProgressUpdate {
        lock.withLock {
            stabilizer.update(
                rawProgress: rawProgress,
                rawRemaining: rawRemaining,
                now: now
            )
        }
    }
}

final class ProgressUpdateThrottler: @unchecked Sendable {
    private let lock = NSLock()
    private let minimumInterval: TimeInterval
    private let minimumProgressDelta: Double
    private var lastEmittedAt: Date?
    private var lastEmittedProgress: Double = 0

    init(
        minimumInterval: TimeInterval = 1,
        minimumProgressDelta: Double = 0.001
    ) {
        self.minimumInterval = minimumInterval
        self.minimumProgressDelta = minimumProgressDelta
    }

    func shouldEmit(progress: Double, now: Date = Date()) -> Bool {
        lock.withLock {
            if progress >= 1 {
                lastEmittedAt = now
                lastEmittedProgress = progress
                return true
            }
            guard let lastEmittedAt else {
                self.lastEmittedAt = now
                lastEmittedProgress = progress
                return true
            }
            let progressAdvancedEnough = progress - lastEmittedProgress >= minimumProgressDelta
            let waitedLongEnough = now.timeIntervalSince(lastEmittedAt) >= minimumInterval
            guard progressAdvancedEnough || waitedLongEnough else {
                return false
            }
            self.lastEmittedAt = now
            lastEmittedProgress = progress
            return true
        }
    }
}
