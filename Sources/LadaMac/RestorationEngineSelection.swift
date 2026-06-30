import Foundation

enum RestorationEngineMode: String, CaseIterable, Sendable {
    case python
    case nativeMetal = "native-metal"
    case nativeCoreAI = "native-coreai"

    var title: String {
        switch self {
        case .python:
            "Python Lada"
        case .nativeMetal:
            "Native Metal"
        case .nativeCoreAI:
            "Native Core AI"
        }
    }

    var detail: String {
        switch self {
        case .python:
            "Production path using bundled Python, PyTorch/MPS, and Lada models"
        case .nativeMetal:
            "Experimental native AVFoundation + Metal path; restoration model is still scaffolded"
        case .nativeCoreAI:
            "Experimental macOS 27 Core AI path; falls back until .aimodel inference is wired"
        }
    }
}

enum RestorationEngineSelection {
    static let environmentKey = "LADA_ENGINE"

    static var currentMode: RestorationEngineMode {
        mode(
            environment: ProcessInfo.processInfo.environment,
            arguments: CommandLine.arguments
        )
    }

    static func makeEngine() -> any RestorationEngine {
        makeEngine(mode: currentMode)
    }

    static func makeEngine(mode: RestorationEngineMode) -> any RestorationEngine {
        switch mode {
        case .python:
            PythonLadaEngine()
        case .nativeMetal:
            NativeMetalEngine()
        case .nativeCoreAI:
            NativeCoreAIEngine()
        }
    }

    static func mode(
        environment: [String: String],
        arguments: [String]
    ) -> RestorationEngineMode {
        if let argumentValue = argumentEngineValue(arguments: arguments),
           let mode = RestorationEngineMode(rawValue: argumentValue)
        {
            return mode
        }
        if let value = environment[environmentKey],
           let mode = RestorationEngineMode(rawValue: value)
        {
            return mode
        }
        return .python
    }

    private static func argumentEngineValue(arguments: [String]) -> String? {
        for argument in arguments {
            if argument.hasPrefix("--engine=") {
                return String(argument.dropFirst("--engine=".count))
            }
        }
        return nil
    }
}
