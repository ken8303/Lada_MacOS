import Foundation

struct NativeDetectorReferenceComparison: Sendable {
    let referenceCount: Int
    let candidateCount: Int
    let matches: [Match]
    let missingReferenceCount: Int
    let extraCandidateCount: Int

    var passed: Bool {
        missingReferenceCount == 0 && extraCandidateCount == 0
    }

    var minimumIoU: Float {
        matches.map(\.iou).min() ?? 0
    }

    var maximumConfidenceDelta: Float {
        matches.map(\.confidenceDelta).max() ?? 0
    }

    struct Match: Sendable {
        let referenceIndex: Int
        let candidateIndex: Int
        let iou: Float
        let confidenceDelta: Float
    }
}

struct NativeDetectorReferenceComparator: Sendable {
    let minimumIoU: Float
    let maximumConfidenceDelta: Float

    init(
        minimumIoU: Float = 0.5,
        maximumConfidenceDelta: Float = 0.2
    ) {
        self.minimumIoU = minimumIoU
        self.maximumConfidenceDelta = maximumConfidenceDelta
    }

    func compare(
        reference: [NativeMosaicDetection],
        candidate: [NativeMosaicDetection]
    ) -> NativeDetectorReferenceComparison {
        var remainingCandidateIndices = Set(candidate.indices)
        var matches: [NativeDetectorReferenceComparison.Match] = []

        for referenceIndex in reference.indices {
            let referenceDetection = reference[referenceIndex]
            let best = remainingCandidateIndices
                .map { candidateIndex in
                    (
                        candidateIndex: candidateIndex,
                        iou: referenceDetection.boundingBox.iou(
                            with: candidate[candidateIndex].boundingBox
                        ),
                        confidenceDelta: abs(
                            referenceDetection.confidence - candidate[candidateIndex].confidence
                        )
                    )
                }
                .filter {
                    $0.iou >= minimumIoU &&
                        $0.confidenceDelta <= maximumConfidenceDelta
                }
                .sorted {
                    if $0.iou == $1.iou {
                        return $0.confidenceDelta < $1.confidenceDelta
                    }
                    return $0.iou > $1.iou
                }
                .first

            guard let best else {
                continue
            }

            remainingCandidateIndices.remove(best.candidateIndex)
            matches.append(
                NativeDetectorReferenceComparison.Match(
                    referenceIndex: referenceIndex,
                    candidateIndex: best.candidateIndex,
                    iou: best.iou,
                    confidenceDelta: best.confidenceDelta
                )
            )
        }

        return NativeDetectorReferenceComparison(
            referenceCount: reference.count,
            candidateCount: candidate.count,
            matches: matches,
            missingReferenceCount: reference.count - matches.count,
            extraCandidateCount: candidate.count - matches.count
        )
    }
}

extension NativeRestorationRegion {
    func iou(with other: NativeRestorationRegion) -> Float {
        let left = max(x, other.x)
        let top = max(y, other.y)
        let right = min(x + width, other.x + other.width)
        let bottom = min(y + height, other.y + other.height)
        let intersection = max(0, right - left) * max(0, bottom - top)
        let union = area + other.area - intersection
        guard union > 0 else {
            return 0
        }
        return Float(intersection) / Float(union)
    }

    private var area: Int {
        max(0, width) * max(0, height)
    }
}
