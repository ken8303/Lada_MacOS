import Foundation

struct NativeYOLOOutputShape: Sendable {
    let batch: Int
    let channels: Int
    let anchors: Int

    static let ladaMosaicDetector = NativeYOLOOutputShape(
        batch: 1,
        channels: 38,
        anchors: 8400
    )

    var maskCoefficientCount: Int { 32 }
    var classCount: Int { channels - 4 - maskCoefficientCount }
}

struct NativeYOLOPrototypeShape: Sendable {
    let batch: Int
    let channels: Int
    let height: Int
    let width: Int

    static let ladaMosaicDetector = NativeYOLOPrototypeShape(
        batch: 1,
        channels: 32,
        height: 160,
        width: 160
    )
}

struct NativeYOLOInputLayout: Sendable {
    let modelInputSize: Int
    let frameWidth: Int
    let frameHeight: Int
    let scale: Float
    let scaledWidth: Int
    let scaledHeight: Int
    let padX: Float
    let padY: Float

    static func aspectFit(
        modelInputSize: Int,
        frameWidth: Int,
        frameHeight: Int
    ) -> NativeYOLOInputLayout? {
        guard modelInputSize > 0,
              frameWidth > 0,
              frameHeight > 0
        else {
            return nil
        }

        let widthScale = Float(modelInputSize) / Float(frameWidth)
        let heightScale = Float(modelInputSize) / Float(frameHeight)
        let scale = min(widthScale, heightScale)
        let scaledWidth = max(1, Int((Float(frameWidth) * scale).rounded()))
        let scaledHeight = max(1, Int((Float(frameHeight) * scale).rounded()))
        return NativeYOLOInputLayout(
            modelInputSize: modelInputSize,
            frameWidth: frameWidth,
            frameHeight: frameHeight,
            scale: scale,
            scaledWidth: scaledWidth,
            scaledHeight: scaledHeight,
            padX: Float(modelInputSize - scaledWidth) / 2,
            padY: Float(modelInputSize - scaledHeight) / 2
        )
    }

    func sourceRegion(
        centerX: Float,
        centerY: Float,
        width: Float,
        height: Float
    ) -> NativeRestorationRegion {
        let left = Int(((centerX - width / 2 - padX) / scale).rounded(.down))
        let top = Int(((centerY - height / 2 - padY) / scale).rounded(.down))
        let right = Int(((centerX + width / 2 - padX) / scale).rounded(.up))
        let bottom = Int(((centerY + height / 2 - padY) / scale).rounded(.up))
        return NativeRestorationRegion(
            x: left,
            y: top,
            width: max(0, right - left),
            height: max(0, bottom - top)
        )
    }
}

struct NativeYOLOPostprocessor: Sendable {
    let outputShape: NativeYOLOOutputShape
    let prototypeShape: NativeYOLOPrototypeShape
    let modelInputSize: Int
    let confidenceThreshold: Float
    let iouThreshold: Float

    init(
        outputShape: NativeYOLOOutputShape = .ladaMosaicDetector,
        prototypeShape: NativeYOLOPrototypeShape = .ladaMosaicDetector,
        modelInputSize: Int = 640,
        confidenceThreshold: Float = 0.25,
        iouThreshold: Float = 0.7
    ) {
        self.outputShape = outputShape
        self.prototypeShape = prototypeShape
        self.modelInputSize = modelInputSize
        self.confidenceThreshold = confidenceThreshold
        self.iouThreshold = iouThreshold
    }

    func detections(
        output0: [Float],
        output1: [Float],
        frameWidth: Int,
        frameHeight: Int
    ) -> [NativeMosaicDetection] {
        guard outputShape.batch == 1,
              prototypeShape.batch == 1,
              outputShape.classCount > 0,
              output0.count == outputShape.batch * outputShape.channels * outputShape.anchors,
              output1.count == prototypeShape.batch * prototypeShape.channels * prototypeShape.height * prototypeShape.width,
              frameWidth > 0,
              frameHeight > 0
        else {
            return []
        }

        guard let layout = NativeYOLOInputLayout.aspectFit(
            modelInputSize: modelInputSize,
            frameWidth: frameWidth,
            frameHeight: frameHeight
        ) else {
            return []
        }
        var candidates: [NativeMosaicDetection] = []

        for anchor in 0..<outputShape.anchors {
            var bestConfidence: Float = 0
            for classIndex in 0..<outputShape.classCount {
                let score = value(output0, channel: 4 + classIndex, anchor: anchor)
                bestConfidence = max(bestConfidence, score)
            }
            guard bestConfidence >= confidenceThreshold else {
                continue
            }

            let centerX = value(output0, channel: 0, anchor: anchor)
            let centerY = value(output0, channel: 1, anchor: anchor)
            let width = value(output0, channel: 2, anchor: anchor)
            let height = value(output0, channel: 3, anchor: anchor)
            let unclamped = layout.sourceRegion(
                centerX: centerX,
                centerY: centerY,
                width: width,
                height: height
            )
            let frame = NativeBGRAFrame(
                width: frameWidth,
                height: frameHeight,
                bytes: []
            )
            guard let boundingBox = unclamped.clamped(to: frame) else {
                continue
            }
            candidates.append(
                NativeMosaicDetection(
                    confidence: bestConfidence,
                    boundingBox: boundingBox,
                    mask: NativeDetectionMaskMetadata(
                        width: prototypeShape.width,
                        height: prototypeShape.height,
                        coordinateSpace: .modelInput
                    )
                )
            )
        }

        return suppressOverlaps(candidates)
    }

    private func value(_ output: [Float], channel: Int, anchor: Int) -> Float {
        output[channel * outputShape.anchors + anchor]
    }

    private func suppressOverlaps(_ detections: [NativeMosaicDetection]) -> [NativeMosaicDetection] {
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        var kept: [NativeMosaicDetection] = []
        for detection in sorted {
            let overlapsExisting = kept.contains {
                iou(detection.boundingBox, $0.boundingBox) > iouThreshold
            }
            if !overlapsExisting {
                kept.append(detection)
            }
        }
        return kept
    }

    private func iou(
        _ lhs: NativeRestorationRegion,
        _ rhs: NativeRestorationRegion
    ) -> Float {
        let left = max(lhs.x, rhs.x)
        let top = max(lhs.y, rhs.y)
        let right = min(lhs.x + lhs.width, rhs.x + rhs.width)
        let bottom = min(lhs.y + lhs.height, rhs.y + rhs.height)
        let intersection = max(0, right - left) * max(0, bottom - top)
        let lhsArea = lhs.width * lhs.height
        let rhsArea = rhs.width * rhs.height
        let union = lhsArea + rhsArea - intersection
        guard union > 0 else {
            return 0
        }
        return Float(intersection) / Float(union)
    }
}
