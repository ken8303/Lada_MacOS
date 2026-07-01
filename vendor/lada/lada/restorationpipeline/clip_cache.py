# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
"""Two-pass mosaic restoration cache support."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from queue import Empty
from typing import Callable

import torch

from lada import LOG_LEVEL
from lada.restorationpipeline.mosaic_detector import Clip
from lada.utils import threading_utils
from lada.utils.threading_utils import EOF_MARKER, STOP_MARKER, ErrorMarker, PipelineQueue, PipelineThread

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

MANIFEST_FILENAME = "manifest.json"
CACHE_FORMAT_VERSION = 1


@dataclass
class ClipManifestEntry:
    id: int
    frame_start: int
    frame_end: int
    size: int
    pad_mode: str
    file_path: str
    boxes: list[tuple[int, int, int, int]]
    crop_shapes: list[tuple[int, int]]
    pad_after_resizes: list[tuple[int, int, int, int]]


@dataclass
class CacheManifest:
    format_version: int
    video_file: str
    max_clip_length: int
    clip_size: int
    pad_mode: str
    frame_detection_counts: list[tuple[int, int]]
    clips: list[ClipManifestEntry]


class ClipCacheWriter:
    def __init__(self, cache_dir: str, video_file: str, max_clip_length: int, clip_size: int, pad_mode: str):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.video_file = video_file
        self.max_clip_length = max_clip_length
        self.clip_size = clip_size
        self.pad_mode = pad_mode
        self._frame_detection_counts: list[tuple[int, int]] = []
        self._clip_entries: list[ClipManifestEntry] = []
        self._lock = threading.Lock()

    def write_clip(self, clip: Clip) -> None:
        payload_path = os.path.join(self.cache_dir, f"clip_{clip.id:08d}.pt")
        torch.save(
            {
                "frames": torch.stack(clip.frames, dim=0),
                "masks": torch.stack(clip.masks, dim=0),
            },
            payload_path,
        )
        entry = ClipManifestEntry(
            id=clip.id,
            frame_start=clip.frame_start,
            frame_end=clip.frame_end,
            size=clip.size,
            pad_mode=clip.pad_mode,
            file_path=clip.file_path,
            boxes=[tuple(box) for box in clip.boxes],
            crop_shapes=[tuple(shape) for shape in clip.crop_shapes],
            pad_after_resizes=[tuple(pad) for pad in clip.pad_after_resizes],
        )
        with self._lock:
            self._clip_entries.append(entry)

    def record_frame_detection_count(self, frame_num: int, num_mosaics_detected: int) -> None:
        with self._lock:
            self._frame_detection_counts.append((frame_num, num_mosaics_detected))

    def finalize(self) -> None:
        with self._lock:
            clips = sorted(self._clip_entries, key=lambda entry: entry.id)
            counts = list(self._frame_detection_counts)
        manifest = CacheManifest(
            format_version=CACHE_FORMAT_VERSION,
            video_file=self.video_file,
            max_clip_length=self.max_clip_length,
            clip_size=self.clip_size,
            pad_mode=self.pad_mode,
            frame_detection_counts=counts,
            clips=clips,
        )
        manifest_path = os.path.join(self.cache_dir, MANIFEST_FILENAME)
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(asdict(manifest), file)
        logger.info(
            "ClipCacheWriter: wrote %d clips and %d frame records to %s",
            len(clips),
            len(counts),
            self.cache_dir,
        )


def _as_error_marker(error: BaseException) -> ErrorMarker:
    return ErrorMarker(message=str(error), stack_trace=traceback.format_exc())


def run_detect_only_pass(
    mosaic_detector,
    cache_dir: str,
    video_file: str,
    max_clip_length: int,
    clip_size: int,
    pad_mode: str,
    start_ns: int = 0,
    stop_requested: Callable[[], bool] = lambda: False,
) -> None:
    """Run live mosaic detection and persist its queue outputs to disk."""
    writer = ClipCacheWriter(cache_dir, video_file, max_clip_length, clip_size, pad_mode)
    errors: list[BaseException] = []

    def drain_detection_queue() -> None:
        while not stop_requested():
            elem = mosaic_detector.frame_detection_queue.get()
            if elem is EOF_MARKER or elem is STOP_MARKER:
                return
            frame_num, num_mosaics_detected = elem
            writer.record_frame_detection_count(frame_num, num_mosaics_detected)

    def drain_clip_queue() -> None:
        while not stop_requested():
            elem = mosaic_detector.mosaic_clip_queue.get()
            if elem is EOF_MARKER or elem is STOP_MARKER:
                return
            writer.write_clip(elem)

    def guarded(target: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            try:
                target()
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
                try:
                    mosaic_detector.stop()
                except Exception:
                    logger.debug("failed to stop detector after cache writer error", exc_info=True)
        return run

    detection_thread = threading.Thread(target=guarded(drain_detection_queue), name="clip cache detection drain", daemon=True)
    clip_thread = threading.Thread(target=guarded(drain_clip_queue), name="clip cache clip drain", daemon=True)

    mosaic_detector.start(start_ns=start_ns)
    detection_thread.start()
    clip_thread.start()
    try:
        while detection_thread.is_alive() or clip_thread.is_alive():
            if stop_requested():
                mosaic_detector.stop()
                break
            detection_thread.join(timeout=0.1)
            clip_thread.join(timeout=0.1)
        if errors:
            raise errors[0]
    finally:
        mosaic_detector.stop()
        detection_thread.join(timeout=2)
        clip_thread.join(timeout=2)
        writer.finalize()


class CachedMosaicDetector:
    """Replay cached clips through FrameRestorer's existing queue contract."""

    def __init__(
        self,
        cache_dir: str,
        frame_detection_queue: PipelineQueue,
        mosaic_clip_queue: PipelineQueue,
        error_handler: Callable[[ErrorMarker], None],
        diagnostic_callback=None,
    ):
        self.cache_dir = cache_dir
        self.frame_detection_queue = frame_detection_queue
        self.mosaic_clip_queue = mosaic_clip_queue
        self.error_handler = error_handler
        self.diagnostic_callback = diagnostic_callback
        self.stop_requested = False
        self._replay_thread: PipelineThread | None = None
        self._manifest: CacheManifest | None = None

    def _emit_diagnostic(self, stage, phase=None, **payload):
        if self.diagnostic_callback is None:
            return
        try:
            self.diagnostic_callback(stage, phase, **payload)
        except Exception:
            logger.debug("CachedMosaicDetector: diagnostic callback failed", exc_info=True)

    def _load_manifest(self) -> CacheManifest:
        manifest_path = os.path.join(self.cache_dir, MANIFEST_FILENAME)
        with open(manifest_path, encoding="utf-8") as file:
            raw = json.load(file)
        if raw.get("format_version") != CACHE_FORMAT_VERSION:
            raise ValueError(f"Unsupported clip cache format: {raw.get('format_version')}")
        raw["clips"] = [ClipManifestEntry(**clip) for clip in raw["clips"]]
        raw["frame_detection_counts"] = [tuple(item) for item in raw["frame_detection_counts"]]
        return CacheManifest(**raw)

    def _reconstruct_clip(self, entry: ClipManifestEntry) -> Clip:
        payload_path = os.path.join(self.cache_dir, f"clip_{entry.id:08d}.pt")
        payload = torch.load(payload_path, map_location="cpu")
        clip = Clip.__new__(Clip)
        clip.id = entry.id
        clip.file_path = entry.file_path
        clip.frame_start = entry.frame_start
        clip.frame_end = entry.frame_end
        clip.size = entry.size
        clip.pad_mode = entry.pad_mode
        clip.frames = list(payload["frames"].unbind(0))
        clip.masks = list(payload["masks"].unbind(0))
        clip.boxes = [tuple(box) for box in entry.boxes]
        clip.crop_shapes = [tuple(shape) for shape in entry.crop_shapes]
        clip.pad_after_resizes = [tuple(pad) for pad in entry.pad_after_resizes]
        clip._index = 0
        return clip

    def _replay_worker(self):
        logger.debug("CachedMosaicDetector: replay worker started")
        manifest = self._manifest
        assert manifest is not None
        window_started_at = time.monotonic()
        window_frames = 0
        try:
            for frame_num, num_mosaics_detected in manifest.frame_detection_counts:
                if self.stop_requested:
                    break
                self.frame_detection_queue.put((frame_num, num_mosaics_detected))
                window_frames += 1
                window_seconds = time.monotonic() - window_started_at
                if window_seconds >= 10:
                    self._emit_diagnostic(
                        "cached-detect-window",
                        "progress",
                        framesProcessed=frame_num,
                        windowFrames=window_frames,
                        windowSeconds=window_seconds,
                        framesPerSecond=window_frames / window_seconds if window_seconds > 0 else None,
                    )
                    window_started_at = time.monotonic()
                    window_frames = 0
            self.frame_detection_queue.put(EOF_MARKER)

            for entry in sorted(manifest.clips, key=lambda clip: (clip.frame_start, clip.id)):
                if self.stop_requested:
                    break
                self.mosaic_clip_queue.put(self._reconstruct_clip(entry))
            self.mosaic_clip_queue.put(EOF_MARKER)
        except Exception as error:
            logger.exception("CachedMosaicDetector: replay worker failed")
            self.error_handler(_as_error_marker(error))
        logger.debug("CachedMosaicDetector: replay worker stopped")

    def start(self, start_ns: int):
        self._manifest = self._load_manifest()
        self.stop_requested = False
        self._replay_thread = PipelineThread(
            name="cached mosaic detector replay worker",
            target=self._replay_worker,
            error_handler=self.error_handler,
        )
        self._replay_thread.start()

    def stop(self):
        self.stop_requested = True
        threading_utils.empty_out_queue(self.frame_detection_queue)
        threading_utils.empty_out_queue(self.mosaic_clip_queue)
        if self._replay_thread:
            self._replay_thread.join()
        self._replay_thread = None
