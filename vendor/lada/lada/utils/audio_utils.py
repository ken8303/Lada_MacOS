# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging

import av
import io
import os
import subprocess
import shutil
from fractions import Fraction
from typing import Optional
from lada.utils import video_utils, os_utils

logger = logging.getLogger(__name__)

def combine_audio_video_files(av_video_metadata: video_utils.VideoMetadata, tmp_v_video_input_path, av_video_output_path):
    audio_codec = get_audio_codec(av_video_metadata.video_file)
    if audio_codec:
        needs_audio_reencoding = not is_output_container_compatible_with_input_audio_codec(audio_codec, av_video_output_path)
        needs_video_delay = av_video_metadata.start_pts > 0

        if shutil.which("ffmpeg") is None:
            if needs_audio_reencoding:
                raise RuntimeError(
                    f"Audio codec '{audio_codec}' is not compatible with the selected output container. "
                    "Choose MP4 with AAC-compatible audio or install FFmpeg for audio transcoding."
                )
            video_delay = (
                av_video_metadata.start_pts * av_video_metadata.time_base
                if needs_video_delay
                else Fraction(0)
            )
            _remux_audio_with_pyav(
                av_video_metadata.video_file,
                tmp_v_video_input_path,
                av_video_output_path,
                video_delay,
            )
        else:
            cmd = ["ffmpeg", "-y", "-loglevel", "quiet"]
            cmd += ["-i", av_video_metadata.video_file]
            if needs_video_delay > 0:
                delay_in_seconds = float(av_video_metadata.start_pts * av_video_metadata.time_base)
                cmd += ["-itsoffset", str(delay_in_seconds)]
            cmd += ["-i", tmp_v_video_input_path]
            if needs_audio_reencoding:
                cmd += ["-c:v", "copy"]
            else:
                cmd += ["-c", "copy"]
            cmd += ["-map", "1:v:0"]
            cmd += ["-map", "0:a:0"]
            cmd += [av_video_output_path]
            subprocess.run(cmd, stdout=subprocess.PIPE, startupinfo=os_utils.get_subprocess_startup_info())
    else:
        shutil.copy(tmp_v_video_input_path, av_video_output_path)
    os.remove(tmp_v_video_input_path)

def _packet_iterator(container, input_stream, output_stream, timestamp_offset=Fraction(0)):
    for packet in container.demux(input_stream):
        if packet.dts is None:
            continue
        source_time_base = Fraction(packet.time_base)
        sort_timestamp = Fraction(packet.dts) * source_time_base + timestamp_offset
        if timestamp_offset:
            timestamp_shift = int(timestamp_offset / source_time_base)
            packet.dts += timestamp_shift
            if packet.pts is not None:
                packet.pts += timestamp_shift
        packet.stream = output_stream
        yield sort_timestamp, packet

def _remux_audio_with_pyav(input_av_path, input_video_path, output_path, video_delay=Fraction(0)):
    """Combine restored video and compatible source audio without FFmpeg CLI."""
    with (
        av.open(input_av_path, metadata_errors='ignore') as source_container,
        av.open(input_video_path, metadata_errors='ignore') as video_container,
        av.open(output_path, "w") as output_container,
    ):
        input_video_stream = video_container.streams.video[0]
        input_audio_stream = source_container.streams.audio[0]
        output_video_stream = output_container.add_stream_from_template(input_video_stream)
        output_audio_stream = output_container.add_stream_from_template(input_audio_stream)

        video_packets = iter(_packet_iterator(
            video_container,
            input_video_stream,
            output_video_stream,
            video_delay,
        ))
        audio_packets = iter(_packet_iterator(
            source_container,
            input_audio_stream,
            output_audio_stream,
        ))
        next_video = next(video_packets, None)
        next_audio = next(audio_packets, None)

        while next_video is not None or next_audio is not None:
            if next_audio is None or (
                next_video is not None and next_video[0] <= next_audio[0]
            ):
                output_container.mux(next_video[1])
                next_video = next(video_packets, None)
            else:
                output_container.mux(next_audio[1])
                next_audio = next(audio_packets, None)

def get_audio_codec(file_path: str) -> Optional[str]:
    if shutil.which("ffprobe") is None:
        with av.open(file_path, metadata_errors='ignore') as container:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            codec = stream.codec_context.codec
            return codec.name.lower() if codec is not None else stream.codec_context.name.lower()

    cmd = f"ffprobe -loglevel error -select_streams a:0 -show_entries stream=codec_name -of default=nw=1:nk=1"
    cmd = cmd.split() + [file_path]
    cmd_result = subprocess.run(cmd, stdout=subprocess.PIPE, startupinfo=os_utils.get_subprocess_startup_info())
    audio_codec = cmd_result.stdout.decode('utf-8').strip().lower()
    return audio_codec if len(audio_codec) > 0 else None

def is_output_container_compatible_with_input_audio_codec(audio_codec: str, output_path: str) -> bool:
    file_extension = os.path.splitext(output_path)[1]
    file_extension = file_extension.lower()
    if file_extension in ('.mp4', '.m4v'):
        output_container_format = "mp4"
    elif file_extension == '.mkv':
        output_container_format = "matroska"
    else:
        logger.info(f"Couldn't determine video container format based on file extension: {file_extension}")
        return False

    buf = io.BytesIO()
    with av.open(buf, 'w', output_container_format) as container:
        return audio_codec in container.supported_codecs
