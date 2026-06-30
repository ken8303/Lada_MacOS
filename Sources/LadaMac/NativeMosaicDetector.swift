import CoreVideo
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

protocol NativePixelBufferMosaicDetector: NativeMosaicDetector {
    func detections(for pixelBuffer: CVPixelBuffer) throws -> [NativeMosaicDetection]
}

struct NativeDetectorRegionProvider: NativePixelBufferRestorationRegionProvider {
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

    func regions(for pixelBuffer: CVPixelBuffer) throws -> [NativeRestorationRegion] {
        guard let pixelBufferDetector = detector as? any NativePixelBufferMosaicDetector else {
            let frame = try NativePixelBufferBridge.copyBGRAFrame(from: pixelBuffer)
            return try regions(for: frame)
        }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        return try pixelBufferDetector.detections(for: pixelBuffer)
            .filter { $0.confidence >= minimumConfidence }
            .compactMap { $0.boundingBox.clamped(frameWidth: width, frameHeight: height) }
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
        clamped(frameWidth: frame.width, frameHeight: frame.height)
    }

    func clamped(frameWidth: Int, frameHeight: Int) -> NativeRestorationRegion? {
        let minX = max(x, 0)
        let minY = max(y, 0)
        let maxX = min(x + width, frameWidth)
        let maxY = min(y + height, frameHeight)
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
