import Foundation

enum NativeDetectionMaskCoordinateSpace: Sendable {
    case modelInput
    case sourceFrame
}

struct NativeDetectionMaskMetadata: Sendable {
    let width: Int
    let height: Int
    let coordinateSpace: NativeDetectionMaskCoordinateSpace
}

struct NativeMosaicDetection: Sendable {
    let confidence: Float
    let boundingBox: NativeRestorationRegion
    let mask: NativeDetectionMaskMetadata?
}

protocol NativeMosaicDetector: Sendable {
    func detections(for frame: NativeBGRAFrame) throws -> [NativeMosaicDetection]
}

struct NativeDetectorRegionProvider: NativeRestorationRegionProvider {
    let detector: any NativeMosaicDetector
    let minimumConfidence: Float

    init(
        detector: any NativeMosaicDetector,
        minimumConfidence: Float = 0.25
    ) {
        self.detector = detector
        self.minimumConfidence = minimumConfidence
    }

    func regions(for frame: NativeBGRAFrame) throws -> [NativeRestorationRegion] {
        try detector.detections(for: frame)
            .filter { $0.confidence >= minimumConfidence }
            .compactMap { $0.boundingBox.clamped(to: frame) }
    }
}

struct FixedNativeMosaicDetector: NativeMosaicDetector {
    let detections: [NativeMosaicDetection]

    func detections(for frame: NativeBGRAFrame) throws -> [NativeMosaicDetection] {
        detections
    }
}

extension NativeRestorationRegion {
    func clamped(to frame: NativeBGRAFrame) -> NativeRestorationRegion? {
        let minX = max(x, 0)
        let minY = max(y, 0)
        let maxX = min(x + width, frame.width)
        let maxY = min(y + height, frame.height)
        guard maxX > minX, maxY > minY else {
            return nil
        }
        return NativeRestorationRegion(
            x: minX,
            y: minY,
            width: maxX - minX,
            height: maxY - minY
        )
    }
}
