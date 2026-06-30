import Foundation
import Metal

enum NativeMetalImageProcessorError: LocalizedError {
    case metalUnavailable
    case libraryBuildFailed(String)
    case functionMissing
    case pipelineBuildFailed(String)
    case commandQueueMissing
    case bufferCreationFailed
    case invalidInput
    case commandFailed(String)

    var errorDescription: String? {
        switch self {
        case .metalUnavailable:
            "Metal is unavailable on this Mac."
        case .libraryBuildFailed(let message):
            "The native Metal image library could not be built: \(message)"
        case .functionMissing:
            "The native Metal blend kernel could not be found."
        case .pipelineBuildFailed(let message):
            "The native Metal blend pipeline could not be created: \(message)"
        case .commandQueueMissing:
            "The native Metal command queue could not be created."
        case .bufferCreationFailed:
            "The native Metal image buffers could not be created."
        case .invalidInput:
            "The native Metal image processor received mismatched frame data."
        case .commandFailed(let message):
            "The native Metal image command failed: \(message)"
        }
    }
}

final class NativeMetalImageProcessor: @unchecked Sendable {
    private struct BlendParams {
        var width: UInt32
        var height: UInt32
    }

    private struct ResizeParams {
        var sourceWidth: UInt32
        var sourceHeight: UInt32
        var outputWidth: UInt32
        var outputHeight: UInt32
    }

    private struct CropParams {
        var sourceWidth: UInt32
        var sourceHeight: UInt32
        var originX: UInt32
        var originY: UInt32
        var cropWidth: UInt32
        var cropHeight: UInt32
    }

    private struct CompositeRegionParams {
        var width: UInt32
        var height: UInt32
        var originX: UInt32
        var originY: UInt32
        var regionWidth: UInt32
        var regionHeight: UInt32
    }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let blendPipeline: MTLComputePipelineState
    private let resizePipeline: MTLComputePipelineState
    private let cropPipeline: MTLComputePipelineState
    private let compositeRegionPipeline: MTLComputePipelineState

    init(device: MTLDevice? = MTLCreateSystemDefaultDevice()) throws {
        guard let device else {
            throw NativeMetalImageProcessorError.metalUnavailable
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw NativeMetalImageProcessorError.commandQueueMissing
        }
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: Self.kernelSource, options: nil)
        } catch {
            throw NativeMetalImageProcessorError.libraryBuildFailed(error.localizedDescription)
        }
        guard let blendFunction = library.makeFunction(name: "blend_bgra"),
              let resizeFunction = library.makeFunction(name: "resize_bgra_nearest"),
              let cropFunction = library.makeFunction(name: "crop_bgra"),
              let compositeRegionFunction = library.makeFunction(name: "composite_bgra_region")
        else {
            throw NativeMetalImageProcessorError.functionMissing
        }
        do {
            blendPipeline = try device.makeComputePipelineState(function: blendFunction)
            resizePipeline = try device.makeComputePipelineState(function: resizeFunction)
            cropPipeline = try device.makeComputePipelineState(function: cropFunction)
            compositeRegionPipeline = try device.makeComputePipelineState(function: compositeRegionFunction)
        } catch {
            throw NativeMetalImageProcessorError.pipelineBuildFailed(error.localizedDescription)
        }
        self.device = device
        self.commandQueue = commandQueue
    }

    func blendBGRA(
        source: [UInt8],
        restored: [UInt8],
        mask: [Float],
        width: Int,
        height: Int
    ) throws -> [UInt8] {
        guard width > 0, height > 0 else {
            throw NativeMetalImageProcessorError.invalidInput
        }
        let pixelCount = width * height
        guard source.count == pixelCount * 4,
              restored.count == pixelCount * 4,
              mask.count == pixelCount
        else {
            throw NativeMetalImageProcessorError.invalidInput
        }

        guard let sourceBuffer = device.makeBuffer(
                bytes: source,
                length: source.count,
                options: .storageModeShared
              ),
              let restoredBuffer = device.makeBuffer(
                bytes: restored,
                length: restored.count,
                options: .storageModeShared
              ),
              let maskBuffer = device.makeBuffer(
                bytes: mask,
                length: mask.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
              ),
              let outputBuffer = device.makeBuffer(
                length: source.count,
                options: .storageModeShared
              )
        else {
            throw NativeMetalImageProcessorError.bufferCreationFailed
        }

        var params = BlendParams(width: UInt32(width), height: UInt32(height))
        guard let paramsBuffer = device.makeBuffer(
            bytes: &params,
            length: MemoryLayout<BlendParams>.stride,
            options: .storageModeShared
        ) else {
            throw NativeMetalImageProcessorError.bufferCreationFailed
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw NativeMetalImageProcessorError.commandQueueMissing
        }

        encoder.setComputePipelineState(blendPipeline)
        encoder.setBuffer(sourceBuffer, offset: 0, index: 0)
        encoder.setBuffer(restoredBuffer, offset: 0, index: 1)
        encoder.setBuffer(maskBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 4)

        let threadsPerThreadgroup = MTLSize(
            width: max(1, min(blendPipeline.threadExecutionWidth, pixelCount)),
            height: 1,
            depth: 1
        )
        let grid = MTLSize(width: pixelCount, height: 1, depth: 1)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw NativeMetalImageProcessorError.commandFailed(error.localizedDescription)
        }

        let pointer = outputBuffer.contents().bindMemory(
            to: UInt8.self,
            capacity: source.count
        )
        return Array(UnsafeBufferPointer(start: pointer, count: source.count))
    }

    func resizeBGRANearest(
        source: [UInt8],
        width: Int,
        height: Int,
        outputWidth: Int,
        outputHeight: Int
    ) throws -> [UInt8] {
        guard width > 0, height > 0, outputWidth > 0, outputHeight > 0 else {
            throw NativeMetalImageProcessorError.invalidInput
        }
        guard source.count == width * height * 4 else {
            throw NativeMetalImageProcessorError.invalidInput
        }

        let outputByteCount = outputWidth * outputHeight * 4
        guard let sourceBuffer = device.makeBuffer(
                bytes: source,
                length: source.count,
                options: .storageModeShared
              ),
              let outputBuffer = device.makeBuffer(
                length: outputByteCount,
                options: .storageModeShared
              )
        else {
            throw NativeMetalImageProcessorError.bufferCreationFailed
        }

        var params = ResizeParams(
            sourceWidth: UInt32(width),
            sourceHeight: UInt32(height),
            outputWidth: UInt32(outputWidth),
            outputHeight: UInt32(outputHeight)
        )
        guard let paramsBuffer = device.makeBuffer(
            bytes: &params,
            length: MemoryLayout<ResizeParams>.stride,
            options: .storageModeShared
        ) else {
            throw NativeMetalImageProcessorError.bufferCreationFailed
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw NativeMetalImageProcessorError.commandQueueMissing
        }

        let pixelCount = outputWidth * outputHeight
        encoder.setComputePipelineState(resizePipeline)
        encoder.setBuffer(sourceBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 2)

        let threadsPerThreadgroup = MTLSize(
            width: max(1, min(resizePipeline.threadExecutionWidth, pixelCount)),
            height: 1,
            depth: 1
        )
        let grid = MTLSize(width: pixelCount, height: 1, depth: 1)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw NativeMetalImageProcessorError.commandFailed(error.localizedDescription)
        }

        let pointer = outputBuffer.contents().bindMemory(
            to: UInt8.self,
            capacity: outputByteCount
        )
        return Array(UnsafeBufferPointer(start: pointer, count: outputByteCount))
    }

    func cropBGRA(
        source: [UInt8],
        width: Int,
        height: Int,
        x: Int,
        y: Int,
        cropWidth: Int,
        cropHeight: Int
    ) throws -> [UInt8] {
        guard width > 0, height > 0,
              x >= 0, y >= 0,
              cropWidth > 0, cropHeight > 0,
              x + cropWidth <= width,
              y + cropHeight <= height
        else {
            throw NativeMetalImageProcessorError.invalidInput
        }
        guard source.count == width * height * 4 else {
            throw NativeMetalImageProcessorError.invalidInput
        }

        let outputByteCount = cropWidth * cropHeight * 4
        guard let sourceBuffer = device.makeBuffer(
                bytes: source,
                length: source.count,
                options: .storageModeShared
              ),
              let outputBuffer = device.makeBuffer(
                length: outputByteCount,
                options: .storageModeShared
              )
        else {
            throw NativeMetalImageProcessorError.bufferCreationFailed
        }

        var params = CropParams(
            sourceWidth: UInt32(width),
            sourceHeight: UInt32(height),
            originX: UInt32(x),
            originY: UInt32(y),
            cropWidth: UInt32(cropWidth),
            cropHeight: UInt32(cropHeight)
        )
        guard let paramsBuffer = device.makeBuffer(
            bytes: &params,
            length: MemoryLayout<CropParams>.stride,
            options: .storageModeShared
        ) else {
            throw NativeMetalImageProcessorError.bufferCreationFailed
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw NativeMetalImageProcessorError.commandQueueMissing
        }

        let pixelCount = cropWidth * cropHeight
        encoder.setComputePipelineState(cropPipeline)
        encoder.setBuffer(sourceBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 2)

        let threadsPerThreadgroup = MTLSize(
            width: max(1, min(cropPipeline.threadExecutionWidth, pixelCount)),
            height: 1,
            depth: 1
        )
        let grid = MTLSize(width: pixelCount, height: 1, depth: 1)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw NativeMetalImageProcessorError.commandFailed(error.localizedDescription)
        }

        let pointer = outputBuffer.contents().bindMemory(
            to: UInt8.self,
            capacity: outputByteCount
        )
        return Array(UnsafeBufferPointer(start: pointer, count: outputByteCount))
    }

    func compositeBGRARegion(
        source: [UInt8],
        restoredRegion: [UInt8],
        mask: [Float],
        width: Int,
        height: Int,
        x: Int,
        y: Int,
        regionWidth: Int,
        regionHeight: Int
    ) throws -> [UInt8] {
        guard width > 0, height > 0,
              x >= 0, y >= 0,
              regionWidth > 0, regionHeight > 0,
              x + regionWidth <= width,
              y + regionHeight <= height
        else {
            throw NativeMetalImageProcessorError.invalidInput
        }
        let pixelCount = width * height
        let regionPixelCount = regionWidth * regionHeight
        guard source.count == pixelCount * 4,
              restoredRegion.count == regionPixelCount * 4,
              mask.count == regionPixelCount
        else {
            throw NativeMetalImageProcessorError.invalidInput
        }

        guard let sourceBuffer = device.makeBuffer(
                bytes: source,
                length: source.count,
                options: .storageModeShared
              ),
              let restoredBuffer = device.makeBuffer(
                bytes: restoredRegion,
                length: restoredRegion.count,
                options: .storageModeShared
              ),
              let maskBuffer = device.makeBuffer(
                bytes: mask,
                length: mask.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
              ),
              let outputBuffer = device.makeBuffer(
                length: source.count,
                options: .storageModeShared
              )
        else {
            throw NativeMetalImageProcessorError.bufferCreationFailed
        }

        var params = CompositeRegionParams(
            width: UInt32(width),
            height: UInt32(height),
            originX: UInt32(x),
            originY: UInt32(y),
            regionWidth: UInt32(regionWidth),
            regionHeight: UInt32(regionHeight)
        )
        guard let paramsBuffer = device.makeBuffer(
            bytes: &params,
            length: MemoryLayout<CompositeRegionParams>.stride,
            options: .storageModeShared
        ) else {
            throw NativeMetalImageProcessorError.bufferCreationFailed
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw NativeMetalImageProcessorError.commandQueueMissing
        }

        encoder.setComputePipelineState(compositeRegionPipeline)
        encoder.setBuffer(sourceBuffer, offset: 0, index: 0)
        encoder.setBuffer(restoredBuffer, offset: 0, index: 1)
        encoder.setBuffer(maskBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 4)

        let threadsPerThreadgroup = MTLSize(
            width: max(1, min(compositeRegionPipeline.threadExecutionWidth, pixelCount)),
            height: 1,
            depth: 1
        )
        let grid = MTLSize(width: pixelCount, height: 1, depth: 1)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw NativeMetalImageProcessorError.commandFailed(error.localizedDescription)
        }

        let pointer = outputBuffer.contents().bindMemory(
            to: UInt8.self,
            capacity: source.count
        )
        return Array(UnsafeBufferPointer(start: pointer, count: source.count))
    }

    private static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct BlendParams {
        uint width;
        uint height;
    };

    struct ResizeParams {
        uint sourceWidth;
        uint sourceHeight;
        uint outputWidth;
        uint outputHeight;
    };

    struct CropParams {
        uint sourceWidth;
        uint sourceHeight;
        uint originX;
        uint originY;
        uint cropWidth;
        uint cropHeight;
    };

    struct CompositeRegionParams {
        uint width;
        uint height;
        uint originX;
        uint originY;
        uint regionWidth;
        uint regionHeight;
    };

    kernel void blend_bgra(
        device const uchar *source [[buffer(0)]],
        device const uchar *restored [[buffer(1)]],
        device const float *mask [[buffer(2)]],
        device uchar *output [[buffer(3)]],
        constant BlendParams &params [[buffer(4)]],
        uint id [[thread_position_in_grid]]
    ) {
        const uint count = params.width * params.height;
        if (id >= count) {
            return;
        }

        const float alpha = clamp(mask[id], 0.0f, 1.0f);
        const uint base = id * 4;

        for (uint channel = 0; channel < 3; channel++) {
            const float sourceValue = float(source[base + channel]);
            const float restoredValue = float(restored[base + channel]);
            const float mixed = sourceValue + (restoredValue - sourceValue) * alpha;
            output[base + channel] = uchar(clamp(round(mixed), 0.0f, 255.0f));
        }
        output[base + 3] = source[base + 3];
    }

    kernel void resize_bgra_nearest(
        device const uchar *source [[buffer(0)]],
        device uchar *output [[buffer(1)]],
        constant ResizeParams &params [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        const uint count = params.outputWidth * params.outputHeight;
        if (id >= count) {
            return;
        }

        const uint outputX = id % params.outputWidth;
        const uint outputY = id / params.outputWidth;
        const uint sourceX = min((outputX * params.sourceWidth) / params.outputWidth, params.sourceWidth - 1);
        const uint sourceY = min((outputY * params.sourceHeight) / params.outputHeight, params.sourceHeight - 1);
        const uint sourceBase = (sourceY * params.sourceWidth + sourceX) * 4;
        const uint outputBase = id * 4;

        output[outputBase] = source[sourceBase];
        output[outputBase + 1] = source[sourceBase + 1];
        output[outputBase + 2] = source[sourceBase + 2];
        output[outputBase + 3] = source[sourceBase + 3];
    }

    kernel void crop_bgra(
        device const uchar *source [[buffer(0)]],
        device uchar *output [[buffer(1)]],
        constant CropParams &params [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        const uint count = params.cropWidth * params.cropHeight;
        if (id >= count) {
            return;
        }

        const uint cropX = id % params.cropWidth;
        const uint cropY = id / params.cropWidth;
        const uint sourceX = params.originX + cropX;
        const uint sourceY = params.originY + cropY;
        if (sourceX >= params.sourceWidth || sourceY >= params.sourceHeight) {
            return;
        }

        const uint sourceBase = (sourceY * params.sourceWidth + sourceX) * 4;
        const uint outputBase = id * 4;

        output[outputBase] = source[sourceBase];
        output[outputBase + 1] = source[sourceBase + 1];
        output[outputBase + 2] = source[sourceBase + 2];
        output[outputBase + 3] = source[sourceBase + 3];
    }

    kernel void composite_bgra_region(
        device const uchar *source [[buffer(0)]],
        device const uchar *restoredRegion [[buffer(1)]],
        device const float *mask [[buffer(2)]],
        device uchar *output [[buffer(3)]],
        constant CompositeRegionParams &params [[buffer(4)]],
        uint id [[thread_position_in_grid]]
    ) {
        const uint count = params.width * params.height;
        if (id >= count) {
            return;
        }

        const uint x = id % params.width;
        const uint y = id / params.width;
        const uint base = id * 4;

        output[base] = source[base];
        output[base + 1] = source[base + 1];
        output[base + 2] = source[base + 2];
        output[base + 3] = source[base + 3];

        if (x < params.originX || y < params.originY) {
            return;
        }
        const uint regionX = x - params.originX;
        const uint regionY = y - params.originY;
        if (regionX >= params.regionWidth || regionY >= params.regionHeight) {
            return;
        }

        const uint regionID = regionY * params.regionWidth + regionX;
        const uint regionBase = regionID * 4;
        const float alpha = clamp(mask[regionID], 0.0f, 1.0f);

        for (uint channel = 0; channel < 3; channel++) {
            const float sourceValue = float(source[base + channel]);
            const float restoredValue = float(restoredRegion[regionBase + channel]);
            const float mixed = sourceValue + (restoredValue - sourceValue) * alpha;
            output[base + channel] = uchar(clamp(round(mixed), 0.0f, 255.0f));
        }
        output[base + 3] = source[base + 3];
    }
    """
}
