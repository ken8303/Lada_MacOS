import Foundation

struct NativeDetectorReferenceFixture: Decodable, Sendable {
    let schemaVersion: Int
    let source: String
    let weights: String
    let device: String
    let imgsz: Int
    let stride: Int
    let conf: Float
    let iou: Float
    let classes: [Int]?
    let letterbox: Letterbox
    let frames: [Frame]

    struct Letterbox: Decodable, Sendable {
        let kind: String
        let auto: Bool
        let stride: Int
    }

    struct Frame: Decodable, Sendable {
        let frameIndex: Int
        let width: Int
        let height: Int
        let detections: [Detection]

        var nativeFrame: NativeBGRAFrame {
            NativeBGRAFrame(
                width: width,
                height: height,
                bytes: []
            )
        }

        var nativeDetections: [NativeMosaicDetection] {
            detections.compactMap { $0.nativeDetection(clampedTo: nativeFrame) }
        }
    }

    struct Detection: Decodable, Sendable {
        let cls: Int
        let confidence: Float
        let xyxy: [Float]
        let maskShape: [Int]?
        let maskArea: Int?

        func nativeDetection(clampedTo frame: NativeBGRAFrame) -> NativeMosaicDetection? {
            guard xyxy.count == 4 else {
                return nil
            }
            let region = NativeRestorationRegion(
                x: Int(xyxy[0].rounded(.down)),
                y: Int(xyxy[1].rounded(.down)),
                width: max(0, Int(xyxy[2].rounded(.up)) - Int(xyxy[0].rounded(.down))),
                height: max(0, Int(xyxy[3].rounded(.up)) - Int(xyxy[1].rounded(.down)))
            )
            guard let boundingBox = region.clamped(to: frame) else {
                return nil
            }
            return NativeMosaicDetection(
                confidence: confidence,
                boundingBox: boundingBox,
                mask: maskMetadata
            )
        }

        private var maskMetadata: NativeDetectionMaskMetadata? {
            guard let maskShape, maskShape.count >= 3 else {
                return nil
            }
            return NativeDetectionMaskMetadata(
                width: maskShape[2],
                height: maskShape[1],
                coordinateSpace: .modelInput
            )
        }
    }
}

extension NativeDetectorReferenceFixture {
    static func load(from url: URL) throws -> NativeDetectorReferenceFixture {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(
            NativeDetectorReferenceFixture.self,
            from: Data(contentsOf: url)
        )
    }
}
