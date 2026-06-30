import Foundation

enum NativeRestorerModelContract {
    static let modelName = NativeModelBundleCatalog.restorerModelName
    static let inputFeatureName = "frames"
    static let outputFeatureName = "restored_frames"
    static let layout = "BTCHW"
    static let colorOrder = "BGR"
    static let batchSize = 1
    static let channelCount = 3
    static let preferredClipLength = 16
    static let preferredSpatialSize = 256
    static let normalizedValueRange = "0...1"

    static func inputShape(
        clipLength: Int = preferredClipLength,
        spatialSize: Int = preferredSpatialSize
    ) -> [Int] {
        [
            batchSize,
            clipLength,
            channelCount,
            spatialSize,
            spatialSize
        ]
    }

    static func outputShape(
        clipLength: Int = preferredClipLength,
        spatialSize: Int = preferredSpatialSize
    ) -> [Int] {
        inputShape(
            clipLength: clipLength,
            spatialSize: spatialSize
        )
    }

    static func expectedElementCount(
        clipLength: Int = preferredClipLength,
        spatialSize: Int = preferredSpatialSize
    ) -> Int {
        inputShape(
            clipLength: clipLength,
            spatialSize: spatialSize
        ).reduce(1, *)
    }
}
