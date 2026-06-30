import Foundation
import Metal

struct MetalCapabilities: Equatable, Sendable {
    let deviceName: String?
    let isAppleSilicon: Bool
    let isMetalAvailable: Bool

    var isMetal4Ready: Bool {
        isAppleSilicon && isMetalAvailable
    }

    var statusDetail: String {
        guard isAppleSilicon else {
            return "Apple Silicon M1 or later is required for this Metal 4 build"
        }
        guard isMetalAvailable else {
            return "Metal device unavailable"
        }
        if let deviceName, !deviceName.isEmpty {
            return "Metal 4 / MPS on \(deviceName) · \(NativeCoreAICapabilities.current().statusDetail)"
        }
        return "Metal 4 / MPS on Apple Silicon · \(NativeCoreAICapabilities.current().statusDetail)"
    }

    static var current: MetalCapabilities {
        let device = MTLCreateSystemDefaultDevice()
        #if arch(arm64)
        let appleSilicon = true
        #else
        let appleSilicon = false
        #endif
        return MetalCapabilities(
            deviceName: device?.name,
            isAppleSilicon: appleSilicon,
            isMetalAvailable: device != nil
        )
    }
}
