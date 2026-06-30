import AVFoundation
import AVKit
import SwiftUI

struct ContentView: View {
    @Environment(RestorationQueue.self) private var queue

    var body: some View {
        @Bindable var queue = queue

        NavigationSplitView {
            Sidebar()
                .navigationSplitViewColumnWidth(min: 176, ideal: 194, max: 220)
        } content: {
            QueueList()
                .navigationSplitViewColumnWidth(min: 560, ideal: 720)
        } detail: {
            Inspector()
                .navigationSplitViewColumnWidth(min: 310, ideal: 350, max: 410)
        }
        .alert(
            "Restoration Error",
            isPresented: Binding(
                get: { queue.lastError != nil },
                set: { if !$0 { queue.lastError = nil } }
            )
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(queue.lastError ?? "")
        }
        .sheet(
            isPresented: Binding(
                get: { queue.previewJob != nil },
                set: { if !$0 { queue.previewJobID = nil } }
            )
        ) {
            if let job = queue.previewJob {
                ResultPreviewView(job: job) {
                    queue.previewJobID = nil
                }
            }
        }
        .onReceive(NotificationCenter.default.publisher(
            for: NSApplication.willTerminateNotification
        )) { _ in
            queue.shutdown()
        }
    }
}

private struct Sidebar: View {
    @Environment(RestorationQueue.self) private var queue

    var body: some View {
        @Bindable var queue = queue

        VStack(spacing: 0) {
            List(SidebarSection.allCases, selection: $queue.selection) { section in
                Label(section.rawValue, systemImage: section.systemImage)
                    .badge(section == .queue && queue.jobs.isEmpty == false
                        ? queue.jobs.count
                        : 0)
                    .tag(section)
            }

            Spacer()

            HStack(spacing: 11) {
                Image(systemName: "apple.logo")
                    .font(.title2)
                VStack(alignment: .leading, spacing: 2) {
                    Text(queue.engineStatus.title)
                        .font(.callout.weight(.semibold))
                    Text(queue.engineStatus.detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer()
                Circle()
                    .fill(engineStatusColor)
                    .frame(width: 9, height: 9)
            }
            .padding(12)
            .background(.background.secondary, in: .rect(cornerRadius: 10))
            .padding(12)
        }
        .navigationTitle("Lada")
    }

    private var engineStatusColor: Color {
        switch queue.engineStatus {
        case .checking: .orange
        case .ready: .green
        case .unavailable: .red
        }
    }
}

private struct QueueList: View {
    @Environment(RestorationQueue.self) private var queue

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Text("Focus Queue")
                    .font(.headline)
                Spacer()
                Button {
                    queue.presentImporter()
                } label: {
                    Label("Add Videos…", systemImage: "plus")
                }
                .buttonStyle(.borderedProminent)

                Button {
                    queue.startQueue()
                } label: {
                    Label("Start Queue", systemImage: "play.fill")
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)
                .disabled(
                    (queue.waitingCount == 0 && !queue.isPaused)
                    || {
                        if case .ready = queue.engineStatus { return false }
                        return true
                    }()
                )

                Button {
                    queue.togglePause()
                } label: {
                    Label(queue.isPaused ? "Resume" : "Pause",
                          systemImage: queue.isPaused ? "play.fill" : "pause.fill")
                }
                .disabled(!queue.isProcessing && !queue.isPaused)

                Divider().frame(height: 20)
                Label("On-device", systemImage: "lock.fill")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            Divider()
            QueueHeader()
            Divider()

            if queue.jobs.isEmpty {
                ContentUnavailableView {
                    Label("No Videos Yet", systemImage: "film.stack")
                } description: {
                    Text("Add videos to prepare a private, on-device restoration queue.")
                } actions: {
                    Button("Add Videos…") {
                        queue.presentImporter()
                    }
                    .buttonStyle(.borderedProminent)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(queue.jobs) { job in
                            QueueRow(job: job, isSelected: queue.selectedJobID == job.id)
                                .contentShape(Rectangle())
                                .onTapGesture { queue.selectedJobID = job.id }
                            Divider()
                        }
                    }
                }
            }

            Divider()
            Text("\(queue.isProcessing ? 1 : 0) processing   •   \(queue.waitingCount) waiting   •   \(queue.completedCount) completed")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
        }
        .navigationTitle("Focus Queue")
    }
}

private struct QueueHeader: View {
    var body: some View {
        HStack(spacing: 16) {
            Text("Video").frame(maxWidth: .infinity, alignment: .leading)
            Text("Duration").frame(width: 72, alignment: .leading)
            Text("Profile").frame(width: 82, alignment: .leading)
            Text("State").frame(width: 102, alignment: .leading)
            Text("Progress").frame(width: 130, alignment: .leading)
            Text("ETA").frame(width: 70, alignment: .leading)
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .padding(.horizontal, 18)
        .padding(.vertical, 11)
    }
}

private struct QueueRow: View {
    @Environment(RestorationQueue.self) private var queue
    @Bindable var job: RestorationJob
    let isSelected: Bool

    var body: some View {
        HStack(spacing: 16) {
            HStack(spacing: 12) {
                VideoThumbnail(url: job.sourceURL)
                    .frame(width: 126, height: 72)
                    .clipShape(.rect(cornerRadius: 6))

                VStack(alignment: .leading, spacing: 4) {
                    Text(job.displayName)
                        .font(.callout.weight(.semibold))
                        .lineLimit(1)
                    Text(metadataLine)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(ByteCountFormatter.string(
                        fromByteCount: job.metadata.fileSize,
                        countStyle: .file
                    ))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Text(job.metadata.duration.formattedDuration)
                .frame(width: 72, alignment: .leading)

            Text("Lada\n\(job.profile.rawValue)")
                .frame(width: 82, alignment: .leading)

            StateLabel(state: job.state)
                .frame(width: 102, alignment: .leading)

            HStack(spacing: 8) {
                ProgressView(value: job.progress)
                    .progressViewStyle(.linear)
                Text(job.progress, format: .percent.precision(.fractionLength(0)))
                    .monospacedDigit()
                    .frame(width: 38, alignment: .trailing)
            }
            .frame(width: 130)

            Text(job.estimatedSecondsRemaining?.formattedDuration ?? "—")
                .monospacedDigit()
                .frame(width: 70, alignment: .leading)
        }
        .font(.callout)
        .padding(.horizontal, 18)
        .padding(.vertical, 18)
        .background(isSelected ? Color.accentColor.opacity(0.08) : Color.clear)
        .overlay(alignment: .leading) {
            if isSelected {
                Rectangle()
                    .fill(Color.accentColor)
                    .frame(width: 3)
            }
        }
        .contextMenu {
            if case .completed = job.state {
                Button("Preview Result") { queue.preview(job) }
            }
            Button("Cancel and Reset") { queue.cancel(job) }
            Button("Remove", role: .destructive) { queue.remove(job) }
        }
    }

    private var metadataLine: String {
        let width = Int(job.metadata.dimensions.width)
        let height = Int(job.metadata.dimensions.height)
        return "\(width) × \(height) · \(job.metadata.frameRate.formatted(.number.precision(.fractionLength(0...2)))) fps"
    }
}

private struct StateLabel: View {
    let state: JobState

    var body: some View {
        HStack(alignment: .top, spacing: 6) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
                .padding(.top, 5)
            VStack(alignment: .leading, spacing: 2) {
                Text(state.title)
                if case .processing = state {
                    Text("Enhancing…")
                        .font(.caption)
                        .foregroundStyle(Color.accentColor)
                } else if case .waiting = state {
                    Text("In queue")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var color: Color {
        switch state {
        case .waiting, .paused: .gray
        case .processing: .blue
        case .completed: .green
        case .failed: .red
        }
    }
}

private struct Inspector: View {
    @Environment(RestorationQueue.self) private var queue

    var body: some View {
        if let job = queue.selectedJob {
            @Bindable var job = job
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    HStack {
                        Text(job.displayName)
                            .font(.headline)
                            .lineLimit(1)
                        Spacer()
                        Button {
                            queue.selectedJobID = nil
                        } label: {
                            Image(systemName: "xmark")
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                    }

                    ZStack {
                        VideoThumbnail(url: job.sourceURL)
                        Circle()
                            .fill(.black.opacity(0.55))
                            .frame(width: 42, height: 42)
                        Image(systemName: "play.fill")
                            .foregroundStyle(.white)
                    }
                    .frame(height: 190)
                    .clipShape(.rect(cornerRadius: 7))

                    GroupBox("Video Info") {
                        LabeledContent("Resolution", value: "\(Int(job.metadata.dimensions.width)) × \(Int(job.metadata.dimensions.height))")
                        LabeledContent("Frame Rate", value: "\(job.metadata.frameRate.formatted(.number.precision(.fractionLength(0...2)))) fps")
                        LabeledContent("Duration", value: job.metadata.duration.formattedDuration)
                        LabeledContent("Codec", value: job.metadata.codec)
                        LabeledContent("File Size", value: ByteCountFormatter.string(fromByteCount: job.metadata.fileSize, countStyle: .file))
                    }

                    Divider()
                    Text("Restoration Mode").font(.headline)

                    Picker("Profile", selection: $job.profile) {
                        ForEach(RestorationProfile.allCases) { profile in
                            Text(profile.rawValue).tag(profile)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()

                    Picker("Quality", selection: $job.quality) {
                        ForEach(QualityPreset.allCases) { preset in
                            Text(preset.rawValue).tag(preset)
                        }
                    }
                    Text("Balanced uses faster Apple encoding; Maximum uses higher bitrate and may take more space.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Picker("Memory Mode", selection: $job.memoryMode) {
                        ForEach(MemoryMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    Text("Long Video uses smaller steady Apple GPU clips for dense or multi-hour videos; Performance uses larger clips when the video is lighter.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Picker("Output Format", selection: $job.outputFormat) {
                        ForEach(OutputFormat.allCases) { format in
                            Text(format.rawValue).tag(format)
                        }
                    }

                    Divider()
                    Text("Destination").font(.headline)
                    HStack {
                        Image(systemName: "folder.fill")
                            .foregroundStyle(.blue)
                        Text(job.destinationURL.path(percentEncoded: false))
                            .font(.caption)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Button("Change…") {
                            queue.changeDestination(for: job)
                        }
                    }

                    if let debugLogURL = job.progressDebugLogURL {
                        Divider()
                        Text("Troubleshooting").font(.headline)
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Progress debug log")
                                .font(.caption.weight(.semibold))
                            Text(debugLogURL.path(percentEncoded: false))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                                .truncationMode(.middle)
                            Button {
                                NSWorkspace.shared.activateFileViewerSelecting([debugLogURL])
                            } label: {
                                Label("Reveal Debug Log", systemImage: "doc.text.magnifyingglass")
                            }
                        }
                    }

                    if case .completed = job.state {
                        HStack {
                            Button {
                                queue.preview(job)
                            } label: {
                                Label("Preview Result", systemImage: "play.rectangle.fill")
                            }
                            .buttonStyle(.borderedProminent)

                            Button("Show in Finder") {
                                NSWorkspace.shared.activateFileViewerSelecting([job.outputURL])
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .trailing)
                    }
                }
                .padding(16)
            }
        } else {
            ContentUnavailableView(
                "Select a Video",
                systemImage: "sidebar.right",
                description: Text("Choose a queued video to review its settings.")
            )
        }
    }
}

private struct ResultPreviewView: View {
    @Bindable var job: RestorationJob
    let close: () -> Void
    @State private var selection: PreviewSelection = .restored
    @State private var player = AVPlayer()

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(job.displayName)
                        .font(.headline)
                        .lineLimit(1)
                    Text(selection == .restored ? "Restored preview" : "Original source")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Picker("Preview", selection: $selection) {
                    ForEach(PreviewSelection.allCases) { option in
                        Text(option.rawValue).tag(option)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 210)

                Button("Done", action: close)
                    .keyboardShortcut(.cancelAction)
            }
            .padding(16)

            Divider()

            VideoPlayer(player: player)
                .background(.black)
                .frame(minWidth: 760, idealWidth: 960, maxWidth: .infinity,
                       minHeight: 430, idealHeight: 540, maxHeight: .infinity)
                .onAppear {
                    loadSelectedVideo(autoplay: true)
                }
                .onChange(of: selection) {
                    loadSelectedVideo(autoplay: true)
                }
                .onDisappear {
                    player.pause()
                    player.replaceCurrentItem(with: nil)
                }

            Divider()

            HStack {
                Image(systemName: "checkmark.seal.fill")
                    .foregroundStyle(.green)
                Text("Playing from this Mac")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Reveal File") {
                    NSWorkspace.shared.activateFileViewerSelecting([selectedURL])
                }
            }
            .padding(12)
        }
        .frame(minWidth: 820, minHeight: 560)
    }

    private var selectedURL: URL {
        switch selection {
        case .source: job.sourceURL
        case .restored: job.outputURL
        }
    }

    private func loadSelectedVideo(autoplay: Bool) {
        player.pause()
        player.replaceCurrentItem(with: AVPlayerItem(url: selectedURL))
        if autoplay {
            player.play()
        }
    }
}

private enum PreviewSelection: String, CaseIterable, Identifiable {
    case restored = "Restored"
    case source = "Source"

    var id: Self { self }
}

struct SettingsView: View {
    @Environment(RestorationQueue.self) private var queue

    var body: some View {
        @Bindable var queue = queue

        Form {
            Section("Processing") {
                LabeledContent("Compute device", value: "Apple Metal 4 / MPS")
                LabeledContent("Architecture", value: "Apple Silicon arm64 only")
                LabeledContent("Privacy", value: "All processing stays on this Mac")
                Toggle("Progress debug logging", isOn: $queue.isProgressDebugLoggingEnabled)
                Text("Writes a per-job JSONL log beside the output video with raw progress, stabilized progress, and ETA values.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Section("Engine") {
                let engineMode = RestorationEngineSelection.currentMode
                LabeledContent("Selected engine", value: engineMode.title)
                Text(engineMode.detail)
                    .foregroundStyle(.secondary)
                LabeledContent("Status", value: queue.engineStatus.title)
                Text(queue.engineStatus.detail)
                    .foregroundStyle(.secondary)
                let coreML = NativeCoreMLCapabilities.current()
                LabeledContent("Core ML", value: coreML.statusTitle)
                Text(coreML.statusDetail)
                    .foregroundStyle(.secondary)
                let coreAI = NativeCoreAICapabilities.current()
                LabeledContent("Core AI", value: coreAI.isReadyForAssets ? "Ready" : "Waiting")
                Text(coreAI.statusDetail)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

private struct VideoThumbnail: View {
    let url: URL
    @State private var image: NSImage?

    var body: some View {
        ZStack {
            Rectangle().fill(.quaternary)
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFill()
            } else {
                Image(systemName: "film")
                    .font(.title)
                    .foregroundStyle(.secondary)
            }
        }
        .task(id: url) {
            image = await makeThumbnail()
        }
    }

    private func makeThumbnail() async -> NSImage? {
        let asset = AVURLAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 960, height: 540)
        do {
            let cgImage = try await generator.image(at: CMTime(seconds: 0.1, preferredTimescale: 600)).image
            return NSImage(cgImage: cgImage, size: .zero)
        } catch {
            return nil
        }
    }
}

private extension TimeInterval {
    var formattedDuration: String {
        guard isFinite, self > 0 else { return "00:00" }
        let total = Int(self.rounded())
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        let seconds = total % 60
        return hours > 0
            ? String(format: "%02d:%02d:%02d", hours, minutes, seconds)
            : String(format: "%02d:%02d", minutes, seconds)
    }
}

private extension NSLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}
