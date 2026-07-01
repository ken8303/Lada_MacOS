#!/usr/bin/env python3
"""Cut a frame-range test clip with PyAV when command-line ffmpeg is absent."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import av


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--end-frame", required=True, type=int)
    return parser.parse_args()


def stream_rate(stream: av.video.stream.VideoStream) -> Fraction:
    rate = stream.average_rate or stream.base_rate
    if rate is None:
        raise SystemExit("Could not detect source frame rate. Install ffmpeg or pass through a source with frame-rate metadata.")
    return Fraction(rate)


def make_encoder(output: av.container.OutputContainer, fps: Fraction, width: int, height: int):
    last_error: Exception | None = None
    for codec in ("libx264", "h264_videotoolbox", "mpeg4"):
        try:
            stream = output.add_stream(codec, rate=fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            if codec == "libx264":
                stream.options = {"preset": "veryfast", "crf": "18"}
            return stream, codec
        except Exception as error:
            last_error = error
    raise SystemExit(f"Could not create an MP4 video encoder: {last_error}")


def main() -> int:
    args = parse_args()
    if args.end_frame <= args.start_frame:
        raise SystemExit("--end-frame must be greater than --start-frame")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(args.input)) as source:
        in_stream = next((stream for stream in source.streams if stream.type == "video"), None)
        if in_stream is None:
            raise SystemExit("No video stream found")

        fps = stream_rate(in_stream)
        width = in_stream.codec_context.width
        height = in_stream.codec_context.height
        if width <= 0 or height <= 0:
            raise SystemExit("Could not detect source dimensions")

        with av.open(str(args.output), mode="w", format="mp4") as target:
            out_stream, codec = make_encoder(target, fps, width, height)
            written = 0
            decoded_index = -1
            for frame in source.decode(in_stream):
                decoded_index += 1
                if decoded_index < args.start_frame:
                    continue
                if decoded_index >= args.end_frame:
                    break

                frame = frame.reformat(width=width, height=height, format="yuv420p")
                frame.pts = None
                for packet in out_stream.encode(frame):
                    target.mux(packet)
                written += 1

            for packet in out_stream.encode(None):
                target.mux(packet)

    if written <= 0:
        raise SystemExit("No frames were written. Check the input frame range.")

    print(f"PyAV wrote {written} frames using encoder {codec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
