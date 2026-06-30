import SwiftUI

@main
struct LadaMacApp: App {
    @State private var queue = RestorationQueue()

    var body: some Scene {
        WindowGroup("Lada") {
            ContentView()
                .environment(queue)
                .frame(minWidth: 1_080, minHeight: 680)
                .task {
                    if AppLaunchMode.engineSmoke {
                        queue.startQueue()
                    }
                }
        }
        .defaultSize(width: 1_360, height: 860)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Add Videos…") {
                    queue.presentImporter()
                }
                .keyboardShortcut("o")
            }
        }

        Settings {
            SettingsView()
                .environment(queue)
                .frame(width: 560, height: 420)
        }
    }
}

enum AppLaunchMode {
    static let engineSmoke =
        ProcessInfo.processInfo.environment["LADA_ENGINE_SMOKE"] == "1"
        || CommandLine.arguments.contains("--engine-smoke")
}
