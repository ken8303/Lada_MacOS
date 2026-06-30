import CoreVideo
import Foundation

struct NativeBGRAFrame: Sendable {
    let width: Int
    let height: Int
    let bytes: [UInt8]
}

enum NativePixelBufferBridgeError: LocalizedError {
    case unsupportedPixelFormat
    case baseAddressUnavailable
    case invalidFrameData
    case pixelBufferCreationFailed(Int32)

    var errorDescription: String? {
        switch self {
        case .unsupportedPixelFormat:
            "The native frame bridge expected a BGRA pixel buffer."
        case .baseAddressUnavailable:
            "The native frame bridge could not read the pixel buffer memory."
        case .invalidFrameData:
            "The native frame bridge received BGRA bytes that do not match the frame size."
        case .pixelBufferCreationFailed(let status):
            "The native frame bridge could not create a pixel buffer. CoreVideo status: \(status)."
        }
    }
}

enum NativePixelBufferBridge {
    static func copyBGRAFrame(from pixelBuffer: CVPixelBuffer) throws -> NativeBGRAFrame {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA else {
            throw NativePixelBufferBridgeError.unsupportedPixelFormat
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw NativePixelBufferBridgeError.baseAddressUnavailable
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let compactRowBytes = width * 4
        let source = baseAddress.assumingMemoryBound(to: UInt8.self)
        var bytes = [UInt8](repeating: 0, count: compactRowBytes * height)

        for y in 0..<height {
            let sourceRow = source.advanced(by: y * bytesPerRow)
            bytes.withUnsafeMutableBufferPointer { destination in
                destination.baseAddress?
                    .advanced(by: y * compactRowBytes)
                    .update(from: sourceRow, count: compactRowBytes)
            }
        }

        return NativeBGRAFrame(width: width, height: height, bytes: bytes)
    }

    static func makePixelBuffer(from frame: NativeBGRAFrame) throws -> CVPixelBuffer {
        guard frame.width > 0,
              frame.height > 0,
              frame.bytes.count == frame.width * frame.height * 4
        else {
            throw NativePixelBufferBridgeError.invalidFrameData
        }

        var maybePixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            frame.width,
            frame.height,
            kCVPixelFormatType_32BGRA,
            [
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true
            ] as CFDictionary,
            &maybePixelBuffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer = maybePixelBuffer else {
            throw NativePixelBufferBridgeError.pixelBufferCreationFailed(status)
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw NativePixelBufferBridgeError.baseAddressUnavailable
        }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let compactRowBytes = frame.width * 4
        let destination = baseAddress.assumingMemoryBound(to: UInt8.self)

        frame.bytes.withUnsafeBufferPointer { source in
            guard let sourceBaseAddress = source.baseAddress else {
                return
            }
            for y in 0..<frame.height {
                destination
                    .advanced(by: y * bytesPerRow)
                    .update(
                        from: sourceBaseAddress.advanced(by: y * compactRowBytes),
                        count: compactRowBytes
                    )
            }
        }

        return pixelBuffer
    }
}
