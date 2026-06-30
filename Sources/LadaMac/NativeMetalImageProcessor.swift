import CoreVideo
import Foundation
import Metal
import MetalPerformanceShaders

enum NativeMetalImageProcessorError: LocalizedError {
    case metalUnavailable
    case libraryBuildFailed(String)
    case functionMissing
    case pipelineBuildFailed(String)
    case commandQueueMissing
    case bufferCreationFailed
    case textureCreationFailed
    case textureCacheCreationFailed(CVReturn)
    case cvTextureCreationFailed(CVReturn)
    case pixelBufferPoolCreationFailed(CVReturn)
    case pixelBufferCreationFailed(CVReturn)
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
        case .textureCreationFailed:
            "The native Metal image processor could not allocate a texture."
        case .textureCacheCreationFailed(let status):
            "The native Metal CVMetalTextureCache could not be created (\(status))."
        case .cvTextureCreationFailed(let status):
            "The native Metal processor could not wrap a CVPixelBuffer as a texture (\(status))."
        case .pixelBufferPoolCreationFailed(let status):
            "The native Metal output pixel-buffer pool could not be created (\(status))."
        case .pixelBufferCreationFailed(let status):
            "The native Metal output pixel buffer could not be created (\(status))."
        case .invalidInput:
            "The native Metal image processor received mismatched frame data."
        case .commandFailed(let message):
            "The native Metal image command failed: \(message)"
        }
    }
}

final class NativeMetalTexturePool: @unchecked Sendable {
    private struct Key: Hashable {
        let width: Int
        let height: Int
        let pixelFormat: MTLPixelFormat
        let usageRawValue: UInt
        let storageMode: MTLStorageMode
    }

    private let device: MTLDevice
    private let lock = NSLock()
    private var freeList: [Key: [MTLTexture]] = [:]

    init(device: MTLDevice) {
        self.device = device
    }

    func texture(
        width: Int,
        height: Int,
        pixelFormat: MTLPixelFormat = .bgra8Unorm,
        usage: MTLTextureUsage = [.shaderRead, .shaderWrite],
        storageMode: MTLStorageMode = .private
    ) throws -> MTLTexture {
        guard width > 0, height > 0 else {
            throw NativeMetalImageProcessorError.invalidInput
        }
        let key = Key(
            width: width,
            height: height,
            pixelFormat: pixelFormat,
            usageRawValue: usage.rawValue,
            storageMode: storageMode
        )
        lock.lock()
        if var available = freeList[key], let texture = available.popLast() {
            freeList[key] = available
            lock.unlock()
            return texture
        }
        lock.unlock()

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = usage
        descriptor.storageMode = storageMode
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw NativeMetalImageProcessorError.textureCreationFailed
        }
        return texture
    }

    func recycle(_ texture: MTLTexture) {
        let key = Key(
            width: texture.width,
            height: texture.height,
            pixelFormat: texture.pixelFormat,
            usageRawValue: texture.usage.rawValue,
            storageMode: texture.storageMode
        )
        lock.lock()
        freeList[key, default: []].append(texture)
        lock.unlock()
    }
}

final class NativeMetalPixelBufferTextureBridge: @unchecked Sendable {
    private let textureCache: CVMetalTextureCache

    init(device: MTLDevice) throws {
        var cache: CVMetalTextureCache?
        let status = CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache)
        guard status == kCVReturnSuccess, let cache else {
            throw NativeMetalImageProcessorError.textureCacheCreationFailed(status)
        }
        textureCache = cache
    }

    func texture(from pixelBuffer: CVPixelBuffer) throws -> (cvTexture: CVMetalTexture, texture: MTLTexture) {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            textureCache,
            pixelBuffer,
            nil,
            .bgra8Unorm,
            width,
            height,
            0,
            &cvTexture
        )
        guard status == kCVReturnSuccess,
              let cvTexture,
              let texture = CVMetalTextureGetTexture(cvTexture)
        else {
            throw NativeMetalImageProcessorError.cvTextureCreationFailed(status)
        }
        return (cvTexture, texture)
    }

    func flush() {
        CVMetalTextureCacheFlush(textureCache, 0)
    }
}

final class NativeMetalPixelBufferPool: @unchecked Sendable {
    private struct Key: Hashable {
        let width: Int
        let height: Int
    }

    private let lock = NSLock()
    private var pools: [Key: CVPixelBufferPool] = [:]

    func pixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        guard width > 0, height > 0 else {
            throw NativeMetalImageProcessorError.invalidInput
        }
        let key = Key(width: width, height: height)
        let pool: CVPixelBufferPool = try lock.withLock {
            if let pool = pools[key] {
                return pool
            }
            var maybePool: CVPixelBufferPool?
            let status = CVPixelBufferPoolCreate(
                kCFAllocatorDefault,
                [
                    kCVPixelBufferPoolMinimumBufferCountKey: 3
                ] as CFDictionary,
                [
                    kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
                    kCVPixelBufferWidthKey: width,
                    kCVPixelBufferHeightKey: height,
                    kCVPixelBufferCGImageCompatibilityKey: true,
                    kCVPixelBufferCGBitmapContextCompatibilityKey: true,
                    kCVPixelBufferMetalCompatibilityKey: true,
                    kCVPixelBufferIOSurfacePropertiesKey: [:]
                ] as CFDictionary,
                &maybePool
            )
            guard status == kCVReturnSuccess, let maybePool else {
                throw NativeMetalImageProcessorError.pixelBufferPoolCreationFailed(status)
            }
            pools[key] = maybePool
            return maybePool
        }

        var maybePixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(
            kCFAllocatorDefault,
            pool,
            &maybePixelBuffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer = maybePixelBuffer else {
            throw NativeMetalImageProcessorError.pixelBufferCreationFailed(status)
        }
        return pixelBuffer
    }
}

final class NativeMetalImageProcessor: @unchecked Sendable {
    private struct BlendParams {
        var alpha: Float
    }

    private struct CompositeRegionParams {
        var originX: UInt32
        var originY: UInt32
        var alpha: Float
    }

    let device: MTLDevice
    let texturePool: NativeMetalTexturePool
    let pixelBufferBridge: NativeMetalPixelBufferTextureBridge
    let outputPixelBufferPool: NativeMetalPixelBufferPool

    private let commandQueue: MTLCommandQueue
    private let blendPipeline: MTLComputePipelineState
    private let compositeRegionPipeline: MTLComputePipelineState
    private let bilinearScale: MPSImageBilinearScale

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
        guard let blendFunction = library.makeFunction(name: "blend_bgra_tex"),
              let compositeRegionFunction = library.makeFunction(name: "composite_bgra_region_tex")
        else {
            throw NativeMetalImageProcessorError.functionMissing
        }
        do {
            blendPipeline = try device.makeComputePipelineState(function: blendFunction)
            compositeRegionPipeline = try device.makeComputePipelineState(function: compositeRegionFunction)
        } catch {
            throw NativeMetalImageProcessorError.pipelineBuildFailed(error.localizedDescription)
        }

        self.device = device
        self.commandQueue = commandQueue
        self.texturePool = NativeMetalTexturePool(device: device)
        self.pixelBufferBridge = try NativeMetalPixelBufferTextureBridge(device: device)
        self.outputPixelBufferPool = NativeMetalPixelBufferPool()
        self.bilinearScale = MPSImageBilinearScale(device: device)
    }

    func makeCommandBuffer() throws -> MTLCommandBuffer {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw NativeMetalImageProcessorError.commandQueueMissing
        }
        return commandBuffer
    }

    func crop(
        _ source: MTLTexture,
        region: NativeRestorationRegion,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLTexture {
        guard region.width > 0, region.height > 0,
              region.x >= 0, region.y >= 0,
              region.x + region.width <= source.width,
              region.y + region.height <= source.height
        else {
            throw NativeMetalImageProcessorError.invalidInput
        }
        let destination = try texturePool.texture(
            width: region.width,
            height: region.height,
            pixelFormat: source.pixelFormat,
            usage: [.shaderRead, .shaderWrite]
        )
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw NativeMetalImageProcessorError.commandQueueMissing
        }
        blit.copy(
            from: source,
            sourceSlice: 0,
            sourceLevel: 0,
            sourceOrigin: MTLOrigin(x: region.x, y: region.y, z: 0),
            sourceSize: MTLSize(width: region.width, height: region.height, depth: 1),
            to: destination,
            destinationSlice: 0,
            destinationLevel: 0,
            destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
        )
        blit.endEncoding()
        return destination
    }

    func resize(
        _ source: MTLTexture,
        toWidth width: Int,
        height: Int,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLTexture {
        guard width > 0, height > 0 else {
            throw NativeMetalImageProcessorError.invalidInput
        }
        let destination = try texturePool.texture(
            width: width,
            height: height,
            pixelFormat: source.pixelFormat,
            usage: [.shaderRead, .shaderWrite]
        )
        bilinearScale.encode(
            commandBuffer: commandBuffer,
            sourceTexture: source,
            destinationTexture: destination
        )
        return destination
    }

    func compositeRegion(
        source: MTLTexture,
        restoredRegion: MTLTexture,
        origin: (x: Int, y: Int),
        alpha: Float,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLTexture {
        guard origin.x >= 0, origin.y >= 0,
              origin.x + restoredRegion.width <= source.width,
              origin.y + restoredRegion.height <= source.height
        else {
            throw NativeMetalImageProcessorError.invalidInput
        }
        let destination = try texturePool.texture(
            width: source.width,
            height: source.height,
            pixelFormat: source.pixelFormat,
            usage: [.shaderRead, .shaderWrite]
        )
        var params = CompositeRegionParams(
            originX: UInt32(origin.x),
            originY: UInt32(origin.y),
            alpha: min(max(alpha, 0), 1)
        )
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw NativeMetalImageProcessorError.commandQueueMissing
        }
        encoder.setComputePipelineState(compositeRegionPipeline)
        encoder.setTexture(source, index: 0)
        encoder.setTexture(restoredRegion, index: 1)
        encoder.setTexture(destination, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<CompositeRegionParams>.stride, index: 0)
        dispatch2D(encoder, pipeline: compositeRegionPipeline, width: source.width, height: source.height)
        encoder.endEncoding()
        return destination
    }

    func blend(
        source: MTLTexture,
        restored: MTLTexture,
        alpha: Float,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLTexture {
        guard source.width == restored.width,
              source.height == restored.height,
              source.pixelFormat == restored.pixelFormat
        else {
            throw NativeMetalImageProcessorError.invalidInput
        }
        let destination = try texturePool.texture(
            width: source.width,
            height: source.height,
            pixelFormat: source.pixelFormat,
            usage: [.shaderRead, .shaderWrite]
        )
        var params = BlendParams(alpha: min(max(alpha, 0), 1))
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw NativeMetalImageProcessorError.commandQueueMissing
        }
        encoder.setComputePipelineState(blendPipeline)
        encoder.setTexture(source, index: 0)
        encoder.setTexture(restored, index: 1)
        encoder.setTexture(destination, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<BlendParams>.stride, index: 0)
        dispatch2D(encoder, pipeline: blendPipeline, width: source.width, height: source.height)
        encoder.endEncoding()
        return destination
    }

    func makeTexture(fromBGRABytes bytes: [UInt8], width: Int, height: Int) throws -> MTLTexture {
        guard width > 0, height > 0, bytes.count == width * height * 4 else {
            throw NativeMetalImageProcessorError.invalidInput
        }
        let staging = try texturePool.texture(
            width: width,
            height: height,
            usage: [.shaderRead],
            storageMode: .shared
        )
        bytes.withUnsafeBytes { pointer in
            staging.replace(
                region: MTLRegion(
                    origin: MTLOrigin(x: 0, y: 0, z: 0),
                    size: MTLSize(width: width, height: height, depth: 1)
                ),
                mipmapLevel: 0,
                withBytes: pointer.baseAddress!,
                bytesPerRow: width * 4
            )
        }
        return staging
    }

    func readBGRABytes(from texture: MTLTexture) throws -> [UInt8] {
        let staging: MTLTexture
        if texture.storageMode == .shared {
            staging = texture
        } else {
            staging = try texturePool.texture(
                width: texture.width,
                height: texture.height,
                pixelFormat: texture.pixelFormat,
                usage: [.shaderRead],
                storageMode: .shared
            )
            let commandBuffer = try makeCommandBuffer()
            guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                throw NativeMetalImageProcessorError.commandQueueMissing
            }
            blit.copy(
                from: texture,
                sourceSlice: 0,
                sourceLevel: 0,
                sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                sourceSize: MTLSize(width: texture.width, height: texture.height, depth: 1),
                to: staging,
                destinationSlice: 0,
                destinationLevel: 0,
                destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
            )
            blit.endEncoding()
            try submitAndWait(commandBuffer)
        }

        let bytesPerRow = staging.width * 4
        var bytes = [UInt8](repeating: 0, count: bytesPerRow * staging.height)
        bytes.withUnsafeMutableBytes { pointer in
            staging.getBytes(
                pointer.baseAddress!,
                bytesPerRow: bytesPerRow,
                from: MTLRegion(
                    origin: MTLOrigin(x: 0, y: 0, z: 0),
                    size: MTLSize(width: staging.width, height: staging.height, depth: 1)
                ),
                mipmapLevel: 0
            )
        }
        if staging !== texture {
            texturePool.recycle(staging)
        }
        return bytes
    }

    func makePixelBuffer(from texture: MTLTexture) throws -> CVPixelBuffer {
        guard texture.pixelFormat == .bgra8Unorm else {
            throw NativeMetalImageProcessorError.invalidInput
        }
        let pixelBuffer = try outputPixelBufferPool.pixelBuffer(
            width: texture.width,
            height: texture.height
        )
        let wrappedDestination = try pixelBufferBridge.texture(from: pixelBuffer)
        let commandBuffer = try makeCommandBuffer()
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw NativeMetalImageProcessorError.commandQueueMissing
        }
        blit.copy(
            from: texture,
            sourceSlice: 0,
            sourceLevel: 0,
            sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: texture.width, height: texture.height, depth: 1),
            to: wrappedDestination.texture,
            destinationSlice: 0,
            destinationLevel: 0,
            destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
        )
        blit.endEncoding()
        try submitAndWait(commandBuffer)
        _ = wrappedDestination.cvTexture
        return pixelBuffer
    }

    func submit(_ commandBuffer: MTLCommandBuffer) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            commandBuffer.addCompletedHandler { buffer in
                if let error = buffer.error {
                    continuation.resume(throwing: NativeMetalImageProcessorError.commandFailed(error.localizedDescription))
                } else {
                    continuation.resume()
                }
            }
            commandBuffer.commit()
        }
    }

    func submitAndWait(_ commandBuffer: MTLCommandBuffer) throws {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw NativeMetalImageProcessorError.commandFailed(error.localizedDescription)
        }
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
        var output = source
        for id in 0..<pixelCount {
            let alpha = min(max(mask[id], 0), 1)
            let base = id * 4
            for channel in 0..<3 {
                let sourceValue = Float(source[base + channel])
                let restoredValue = Float(restored[base + channel])
                let mixed = sourceValue + (restoredValue - sourceValue) * alpha
                output[base + channel] = UInt8(min(max(round(mixed), 0), 255))
            }
            output[base + 3] = source[base + 3]
        }
        return output
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
        var output = [UInt8](repeating: 0, count: outputWidth * outputHeight * 4)
        for outputY in 0..<outputHeight {
            let sourceY = min((outputY * height) / outputHeight, height - 1)
            for outputX in 0..<outputWidth {
                let sourceX = min((outputX * width) / outputWidth, width - 1)
                let sourceBase = (sourceY * width + sourceX) * 4
                let outputBase = (outputY * outputWidth + outputX) * 4
                output[outputBase] = source[sourceBase]
                output[outputBase + 1] = source[sourceBase + 1]
                output[outputBase + 2] = source[sourceBase + 2]
                output[outputBase + 3] = source[sourceBase + 3]
            }
        }
        return output
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
        var output = [UInt8](repeating: 0, count: cropWidth * cropHeight * 4)
        for cropY in 0..<cropHeight {
            for cropX in 0..<cropWidth {
                let sourceBase = ((y + cropY) * width + (x + cropX)) * 4
                let outputBase = (cropY * cropWidth + cropX) * 4
                output[outputBase] = source[sourceBase]
                output[outputBase + 1] = source[sourceBase + 1]
                output[outputBase + 2] = source[sourceBase + 2]
                output[outputBase + 3] = source[sourceBase + 3]
            }
        }
        return output
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

        var output = source
        for regionY in 0..<regionHeight {
            for regionX in 0..<regionWidth {
                let regionID = regionY * regionWidth + regionX
                let alpha = min(max(mask[regionID], 0), 1)
                let sourceBase = ((y + regionY) * width + (x + regionX)) * 4
                let regionBase = regionID * 4
                for channel in 0..<3 {
                    let sourceValue = Float(source[sourceBase + channel])
                    let restoredValue = Float(restoredRegion[regionBase + channel])
                    let mixed = sourceValue + (restoredValue - sourceValue) * alpha
                    output[sourceBase + channel] = UInt8(min(max(round(mixed), 0), 255))
                }
                output[sourceBase + 3] = source[sourceBase + 3]
            }
        }
        return output
    }

    private func dispatch2D(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        width: Int,
        height: Int
    ) {
        let threadWidth = pipeline.threadExecutionWidth
        let threadHeight = max(1, pipeline.maxTotalThreadsPerThreadgroup / threadWidth)
        let threadsPerThreadgroup = MTLSize(width: threadWidth, height: threadHeight, depth: 1)
        let threadgroups = MTLSize(
            width: (width + threadWidth - 1) / threadWidth,
            height: (height + threadHeight - 1) / threadHeight,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    private static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct BlendParams {
        float alpha;
    };

    struct CompositeRegionParams {
        uint originX;
        uint originY;
        float alpha;
    };

    kernel void blend_bgra_tex(
        texture2d<float, access::read> source [[texture(0)]],
        texture2d<float, access::read> restored [[texture(1)]],
        texture2d<float, access::write> output [[texture(2)]],
        constant BlendParams &params [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
            return;
        }
        const float4 sourceColor = source.read(gid);
        const float4 restoredColor = restored.read(gid);
        float4 mixed = mix(sourceColor, restoredColor, params.alpha);
        mixed.a = sourceColor.a;
        output.write(mixed, gid);
    }

    kernel void composite_bgra_region_tex(
        texture2d<float, access::read> source [[texture(0)]],
        texture2d<float, access::read> restoredRegion [[texture(1)]],
        texture2d<float, access::write> output [[texture(2)]],
        constant CompositeRegionParams &params [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        if (gid.x >= output.get_width() || gid.y >= output.get_height()) {
            return;
        }
        const float4 sourceColor = source.read(gid);
        if (gid.x < params.originX || gid.y < params.originY) {
            output.write(sourceColor, gid);
            return;
        }

        const uint2 regionCoord = uint2(gid.x - params.originX, gid.y - params.originY);
        if (regionCoord.x >= restoredRegion.get_width() || regionCoord.y >= restoredRegion.get_height()) {
            output.write(sourceColor, gid);
            return;
        }

        const float4 restoredColor = restoredRegion.read(regionCoord);
        float4 mixed = mix(sourceColor, restoredColor, params.alpha);
        mixed.a = sourceColor.a;
        output.write(mixed, gid);
    }
    """
}
