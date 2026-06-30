import Foundation

final class NativeCoreAIEngine: RestorationEngine, @unchecked Sendable {
    private let fallbackEngine: any RestorationEngine
    private let capabilitiesProvider: @Sendable () -> NativeCoreAICapabilities

    init(
        fallbackEngine: any RestorationEngine = NativeMetalEngine(),
        capabilitiesProvider: @escaping @Sendable () -> NativeCoreAICapabilities = {
            NativeCoreAICapabilities.current()
        }
    ) {
        self.fallbackEngine = fallbackEngine
        self.capabilitiesProvider = capabilitiesProvider
    }

    func restore(
        request: RestorationRequest,
        progress: @escaping @Sendable (Double, TimeInterval?) -> Void,
        diagnostic: @escaping @Sendable (RestorationEngineDiagnostic) -> Void
    ) async throws {
        let capabilities = capabilitiesProvider()
        diagnostic(RestorationEngineDiagnostic(
            event: "coreai-capability",
            message: capabilities.statusDetail
        ))
        guard capabilities.isReadyForAssets else {
            diagnostic(RestorationEngineDiagnostic(
                event: "coreai-fallback",
                message: "Core AI assets are not bundled yet; using native Metal fallback path."
            ))
            try await fallbackEngine.restore(
                request: request,
                progress: progress,
                diagnostic: diagnostic
            )
            return
        }

        diagnostic(RestorationEngineDiagnostic(
            event: "coreai-fallback",
            message: "Core AI assets detected, but detector/restorer inference wiring is not enabled yet; using native Metal fallback path."
        ))
        try await fallbackEngine.restore(
            request: request,
            progress: progress,
            diagnostic: diagnostic
        )
    }

    func probe() async -> EngineStatus {
        let capabilities = capabilitiesProvider()
        guard capabilities.isFrameworkPresent else {
            return .unavailable(capabilities.statusDetail)
        }
        guard capabilities.isSwiftModuleAvailable else {
            return .unavailable(capabilities.statusDetail)
        }
        if capabilities.hasAnyAsset {
            return .ready(capabilities.statusDetail)
        }
        return .unavailable(capabilities.statusDetail)
    }

    func cancel() {
        fallbackEngine.cancel()
    }

    func pause() {
        fallbackEngine.pause()
    }

    func resume() {
        fallbackEngine.resume()
    }
}
