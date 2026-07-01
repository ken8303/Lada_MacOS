# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import argparse
import json
import os
import pathlib
import sys
import tempfile
import textwrap
import time

# When frozen (PyInstaller) on macOS, multiprocessing spawn re-executes this
# executable with -B -S -I -c "from multiprocessing.resource_tracker import main; main(...)".
# PyTorch triggers this. Stdlib freeze_support() only handles worker processes
# (argv[1] == '--multiprocessing-fork'); it does not handle the resource_tracker
# process (which uses -c). See: Lib/multiprocessing/spawn.py `is_forking()` and
# `freeze_support()` in CPython.
# (https://github.com/python/cpython/blob/629a363ddd2889f023d5925506e61f5b6647accd/Lib/multiprocessing/spawn.py#L57-L80).
# Run the -c code and exit so argparse never sees it.
if getattr(sys, "frozen", False) and sys.platform == "darwin":
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "-c" and i + 1 < len(sys.argv):
            exec(compile(sys.argv[i + 1], "<string>", "exec"))
            sys.exit(0)
        if sys.argv[i] == "-c":
            break

try:
    import torch
except ModuleNotFoundError:
    from lada import IS_FLATPAK
    if IS_FLATPAK:
        print(_("No GPU Add-On installed"))
        print(_("In order to use the application you need to install one of Lada's Flatpak Add-Ons that is compatible with your hardware"))
        sys.exit(1)
    else:
        raise

from lada import VERSION, ModelFiles
from lada.cli import utils
from lada.utils import audio_utils, video_utils
from lada.utils.os_utils import gpu_has_fp16_acceleration, get_default_torch_device
from lada.restorationpipeline.frame_restorer import FrameRestorer
from lada.restorationpipeline import load_detection_model, load_models, load_restoration_model, preferred_pad_mode_for_model
from lada.restorationpipeline.clip_cache import run_detect_only_pass
from lada.utils.threading_utils import STOP_MARKER, ErrorMarker
from lada.utils.video_utils import get_video_meta_data, VideoWriter, get_default_preset_name

def setup_argparser() -> argparse.ArgumentParser:
    examples_header_text = _("Examples:")

    example1_text = _("Restore video with default settings:")
    example1_command = _("%(prog)s --input input.mp4")

    example2_text = _("Restore all videos found in the specified directory and save them to a different folder:")
    example2_command = _("%(prog)s --input path/to/input/dir/ --output /path/to/output/dir/")

    example3_text = _("Use Nvidia hardware-accelerated encoder by selecting a preset:")
    example3_command = _("%(prog)s --input input.mp4 --encoding-preset hevc-nvidia-gpu-hq")

    example4_text = _("Set encoding parameters directly without using an encoding preset:")
    example4_command = _("%(prog)s --input input.mp4 --encoder libx265 --encoder-options '-crf 26 -preset fast -x265-params log_level=error'")

    parser = argparse.ArgumentParser(
        usage=_('%(prog)s [options]'),
        description=_("Restore pixelated adult videos (JAV)"),
        epilog=_(textwrap.dedent(f'''\
            {examples_header_text}
                * {example1_text}
                    {example1_command}
                * {example2_text}
                     {example2_command}
                * {example3_text}
                    {example3_command}
                * {example4_text}
                    {example4_command}
            ''')),
        formatter_class=utils.TranslatableHelpFormatter,
        add_help=False)

    group_general = parser.add_argument_group(_('General'))
    group_general.add_argument('--input', type=str, help=_('Path to pixelated video file or directory containing video files'))
    group_general.add_argument('--output', type=str, help=_('Path used to save output file(s). If path is a directory then file name will be chosen automatically (see --output-file-pattern). If no output path was given then the directory of the input file will be used'))
    group_general.add_argument('--temporary-directory', type=str, default=tempfile.gettempdir(), help=_('Directory for temporary video files during restoration process. Alternatively, you can use the environment variable TMPDIR. (default: %(default)s)'))
    group_general.add_argument('--output-file-pattern', type=str, default="{orig_file_name}.restored.mp4", help=_("Pattern used to determine output file name(s). Used when input is a directory, or a file but no output path was specified. Must include the placeholder '{orig_file_name}'. (default: %(default)s)"))
    group_general.add_argument('--device', type=str, default=get_default_torch_device(), help=_('Device used for running Restoration and Detection models. Use "--list-devices" to see what\'s available (default: %(default)s)'))
    group_general.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=gpu_has_fp16_acceleration(), help=_("Reduces VRAM usage and may increase speed on modern GPUs, with negligible quality difference. (default: %(default)s)"))
    group_general.add_argument('--list-devices', action='store_true', help=_("List available devices and exit"))
    group_general.add_argument('--version', action='store_true', help=_("Display version and exit"))
    group_general.add_argument('--help', action='store_true', help=_("Show this help message and exit"))
    group_general.add_argument('--progress-json', action='store_true', help=argparse.SUPPRESS)
    group_general.add_argument('--cache-detections-to', type=str, help=argparse.SUPPRESS)
    group_general.add_argument('--restore-from-cache', type=str, help=argparse.SUPPRESS)

    export = parser.add_argument_group(_('Export'))
    export.add_argument('--encoding-preset', type=str, default=get_default_preset_name(), help=_('Select encoding preset by name. Use "--list-encoding-presets" to see what\'s available. Ignored if "--encoder" and "--encoder-options" are used (default: %(default)s)'))
    export.add_argument('--list-encoding-presets', action='store_true', help=_("List available encoding presets and exit"))
    export.add_argument('--encoder', type=str, help=_('Select video encoder by name. Use "--list-encoders" to see what\'s available. (default: %(default)s)'))
    export.add_argument('--list-encoders', action='store_true', help=_("List available encoders and exit"))
    export.add_argument('--encoder-options', type=str, help=_("Space-separated list of options for the encoder set via \"--encoder\". Use \"--list-encoder-options\" to see what's available. (default: %(default)s)"))
    export.add_argument('--list-encoder-options', metavar='ENCODER', type=str, help=_("List available options of the given encoder and exit"))
    export.add_argument('--mp4-fast-start',  default=False, action=argparse.BooleanOptionalAction, help=_("Allows playing the file while it's being written. Sets .mp4 mov flags \"frag_keyframe+empty_moov+faststart\". (default: %(default)s)"))

    group_restoration = parser.add_argument_group(_('Mosaic Restoration'))
    group_restoration.add_argument('--list-mosaic-restoration-models', action='store_true', help=_("List available restoration models found in model weights directory and exit (default location is './model_weights' if not overwritten by environment variable LADA_MODEL_WEIGHTS_DIR)"))
    group_restoration.add_argument('--mosaic-restoration-model', type=str, default='basicvsrpp-v1.2', help=_('Name of detection model or path to model weights file. Use "--list-mosaic-restoration-models" to see what\'s available. (default: %(default)s)'))
    group_restoration.add_argument('--mosaic-restoration-config-path', type=str, default=None, help=_("Path to restoration model configuration file. You'll not have to set this unless you're training your own custom models"))
    group_restoration.add_argument('--max-clip-length', type=int, default=180, help=_('Maximum number of frames for restoration. Higher values improve temporal stability. Lower values reduce memory footprint. If set too low flickering could appear (default: %(default)s)'))

    group_detection = parser.add_argument_group(_('Mosaic Detection'))
    group_detection.add_argument('--mosaic-detection-model', type=str, default='v4-fast', help=_('Name of detection model or path to model weights file. Use "--list-mosaic-detection-models" to see what\'s available. (default: %(default)s)'))
    group_detection.add_argument('--list-mosaic-detection-models', action='store_true', help=_("List available detection models found in model weights directory and exit (default location is './model_weights' if not overwritten by environment variable LADA_MODEL_WEIGHTS_DIR)"))
    group_detection.add_argument('--detect-face-mosaics', action=argparse.BooleanOptionalAction, default=False, help=_("Detect and ignore areas of pixelated faces. Can prevent restoration artifacts but may worsen detection of NSFW mosaics. Available for models v3 and newer. (default: %(default)s)"))

    return parser

def process_video_file(input_path: str, output_path: str, temp_dir_path: str, device: torch.device, mosaic_restoration_model, mosaic_detection_model,
                       mosaic_restoration_model_name, preferred_pad_mode, max_clip_length, encoder: str, encoder_options: str, mp4_fast_start,
                       progress_json=False, cache_detections_to: str | None = None, restore_from_cache: str | None = None):
    video_metadata = get_video_meta_data(input_path)
    diagnostics_enabled = progress_json
    try:
        diagnostic_window_seconds = max(10.0, float(os.environ.get("LADA_DIAGNOSTIC_WINDOW_SECONDS", "60")))
    except ValueError:
        diagnostic_window_seconds = 60.0

    def emit_diagnostic(stage, phase=None, **payload):
        if not diagnostics_enabled:
            return
        print(json.dumps({
            "type": "diagnostic",
            "stage": stage,
            "phase": phase,
            **payload,
        }), flush=True)

    frame_restorer = FrameRestorer(device, input_path, max_clip_length, mosaic_restoration_model_name,
                 mosaic_detection_model, mosaic_restoration_model, preferred_pad_mode,
                 diagnostic_callback=emit_diagnostic if diagnostics_enabled else None,
                 restore_from_cache=restore_from_cache)
    if cache_detections_to:
        emit_diagnostic(
            "detect-only",
            "start",
            input=input_path,
            cacheDirectory=cache_detections_to,
            totalFrames=video_metadata.frames_count,
            fps=float(video_metadata.video_fps),
            maxClipLength=max_clip_length,
        )
        started_at = time.monotonic()
        run_detect_only_pass(
            frame_restorer.mosaic_detector,
            cache_detections_to,
            input_path,
            max_clip_length,
            clip_size=256,
            pad_mode=preferred_pad_mode,
        )
        emit_diagnostic(
            "detect-only",
            "complete",
            cacheDirectory=cache_detections_to,
            durationSeconds=time.monotonic() - started_at,
        )
        return
    success = True
    pathlib.Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    write_in_progress_output = os.environ.get("LADA_WRITE_IN_PROGRESS_OUTPUT", "1") == "1"
    if write_in_progress_output:
        output_stem, output_ext = os.path.splitext(output_path)
        video_tmp_file_output_path = f"{output_stem}.in-progress{output_ext}"
    else:
        video_tmp_file_output_path = os.path.join(temp_dir_path, f"{os.path.basename(os.path.splitext(output_path)[0])}.tmp{os.path.splitext(output_path)[1]}")
    if os.path.exists(video_tmp_file_output_path):
        os.remove(video_tmp_file_output_path)
    progress_callback = None
    if progress_json:
        def progress_callback(progress, remaining, **payload):
            print(json.dumps({
                "type": "progress",
                "progress": progress,
                "remainingSeconds": remaining,
                **payload,
            }), flush=True)
    frame_restorer_progressbar = utils.Progressbar(
        video_metadata,
        progress_callback=progress_callback,
        disable=progress_json,
    )
    try:
        emit_diagnostic(
            "restore",
            "start",
            input=input_path,
            output=output_path,
            inProgressOutput=video_tmp_file_output_path,
            totalFrames=video_metadata.frames_count,
            fps=float(video_metadata.video_fps),
            maxClipLength=max_clip_length,
            encoder=encoder,
            encoderOptions=encoder_options,
        )
        frame_restorer.start()
        frame_restorer_progressbar.init()
        with VideoWriter(video_tmp_file_output_path, video_metadata.video_width, video_metadata.video_height,
                         video_metadata.video_fps_exact, encoder=encoder, encoder_options=encoder_options,
                         time_base=video_metadata.time_base, mp4_fast_start=mp4_fast_start) as video_writer:
            restore_iterator = iter(frame_restorer)
            window_started_at = time.monotonic()
            window_frames = 0
            window_wait_seconds = 0.0
            window_write_seconds = 0.0
            while True:
                next_started_at = time.monotonic()
                try:
                    elem = next(restore_iterator)
                except StopIteration:
                    break
                window_wait_seconds += time.monotonic() - next_started_at
                if elem is STOP_MARKER or isinstance(elem, ErrorMarker):
                    success = False
                    frame_restorer_progressbar.error = True
                    print("Error on export: frame restorer stopped prematurely")
                    break
                (restored_frame, restored_frame_pts) = elem
                write_started_at = time.monotonic()
                video_writer.write(restored_frame, restored_frame_pts, bgr2rgb=True)
                window_write_seconds += time.monotonic() - write_started_at
                window_frames += 1
                frame_restorer_progressbar.update()
                now = time.monotonic()
                window_seconds = now - window_started_at
                if window_seconds >= diagnostic_window_seconds:
                    emit_diagnostic(
                        "export-window",
                        "progress",
                        framesProcessed=frame_restorer_progressbar.frames_processed,
                        totalFrames=video_metadata.frames_count,
                        windowFrames=window_frames,
                        windowSeconds=window_seconds,
                        framesPerSecond=window_frames / window_seconds if window_seconds > 0 else None,
                        progressPercentPerHour=(
                            (window_frames / video_metadata.frames_count) * 100 * 3600 / window_seconds
                            if window_seconds > 0 and video_metadata.frames_count > 0 else None
                        ),
                        waitForRestoredFrameSeconds=window_wait_seconds,
                        videoWriteSeconds=window_write_seconds,
                        waitForRestoredFramePercent=window_wait_seconds / window_seconds if window_seconds > 0 else None,
                        videoWritePercent=window_write_seconds / window_seconds if window_seconds > 0 else None,
                    )
                    window_started_at = now
                    window_frames = 0
                    window_wait_seconds = 0.0
                    window_write_seconds = 0.0
    except (Exception, KeyboardInterrupt) as e:
        success = False
        if isinstance(e, KeyboardInterrupt):
            raise e
        else:
            print("Error on export", e)
    finally:
        frame_restorer.stop()
        frame_restorer_progressbar.close(ensure_completed_bar=success)

    if success:
        emit_diagnostic("audio", "start")
        print(_("Processing audio"))
        audio_started_at = time.monotonic()
        audio_utils.combine_audio_video_files(video_metadata, video_tmp_file_output_path, output_path)
        emit_diagnostic("audio", "complete", durationSeconds=time.monotonic() - audio_started_at)
        if os.path.exists(video_tmp_file_output_path):
            os.remove(video_tmp_file_output_path)
    else:
        if os.path.exists(video_tmp_file_output_path) and not write_in_progress_output:
            os.remove(video_tmp_file_output_path)
        elif os.path.exists(video_tmp_file_output_path):
            emit_diagnostic("export-output", "partial-kept", inProgressOutput=video_tmp_file_output_path)

def main():
    argparser = setup_argparser()
    args = argparser.parse_args()
    if args.version:
        print("Lada: ", VERSION)
        sys.exit(0)
    if args.list_encoders:
        utils.dump_encoders()
        sys.exit(0)
    if args.list_mosaic_detection_models:
        utils.dump_available_detection_models()
        sys.exit(0)
    if args.list_mosaic_restoration_models:
        utils.dump_available_restoration_models()
        sys.exit(0)
    if args.list_devices:
        utils.dump_torch_devices()
        sys.exit(0)
    if args.list_encoding_presets:
        utils.dump_available_encoding_presets()
        sys.exit(0)
    if args.list_encoder_options:
        utils.dump_encoder_options(args.list_encoder_options)
        sys.exit(0)
    if args.help or not args.input:
        argparser.print_help()
        sys.exit(0)
    if args.cache_detections_to and args.restore_from_cache:
        print(_("Only one of --cache-detections-to or --restore-from-cache can be used"))
        sys.exit(1)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(_("GPU {device} selected but CUDA is not available").format(device=args.device))
        sys.exit(1)
    if args.device == "mps":
        from lada.utils.os_utils import has_mps
        if not has_mps():
            print(_("MPS selected but MPS (Metal) is not available"))
            sys.exit(1)
    if "{orig_file_name}" not in args.output_file_pattern or "." not in args.output_file_pattern:
        print(_("Invalid file name pattern. It must include the template string '{orig_file_name}' and a file extension"))
        sys.exit(1)
    if os.path.isdir(args.input) and args.output is not None and os.path.isfile(args.output):
        print(_("Invalid output directory. If input is a directory then --output must also be set to a directory"))
        sys.exit(1)
    if not (os.path.isfile(args.input) or os.path.isdir(args.input)):
        print(_("Invalid input. No file or directory at {input_path}").format(input_path=args.input))
        sys.exit(1)
    if args.temporary_directory and not os.path.isdir(args.temporary_directory):
        print(_("Temporary directory {temporary_path} doesn't exist. Creating…").format(temporary_path=args.temporary_directory))
        os.makedirs(args.temporary_directory)

    if detection_modelfile := ModelFiles.get_detection_model_by_name(args.mosaic_detection_model):
        mosaic_detection_model_path = detection_modelfile.path
    elif os.path.isfile(args.mosaic_detection_model):
        mosaic_detection_model_path = args.mosaic_detection_model
    else:
        print(_("Invalid mosaic detection model"))
        sys.exit(1)

    if restoration_modelfile := ModelFiles.get_restoration_model_by_name(args.mosaic_restoration_model):
        mosaic_restoration_model_name = args.mosaic_restoration_model
        mosaic_restoration_model_path = restoration_modelfile.path
    elif os.path.isfile(args.mosaic_restoration_model):
        mosaic_restoration_model_path = args.mosaic_restoration_model
        mosaic_restoration_model_name = 'basicvsrpp' # Assume custom model is basicvsrpp. DeepMosaics custom path is not supported
    else:
        print(_("Invalid mosaic restoration model"))
        sys.exit(1)

    encoder = None
    encoder_options = None
    if args.encoder:
        encoder = args.encoder
        encoder_options = args.encoder_options if args.encoder_options else ''
    elif args.encoding_preset:
        encoding_presets = video_utils.get_encoding_presets()
        found = False
        for preset in encoding_presets:
            if preset.name == args.encoding_preset:
                found = True
                encoder = preset.encoder_name
                encoder_options = preset.encoder_options
                break
        if not found:
            print(_("Invalid encoding preset"))
            sys.exit(1)
    else:
        print(_('Either "--encoding-preset" or "--encoder" together with "--encoder-options" must be used'))
        sys.exit(1)
    assert encoder is not None and encoder_options is not None

    device = torch.device(args.device)
    preferred_pad_mode = preferred_pad_mode_for_model(mosaic_restoration_model_name)
    if args.cache_detections_to:
        mosaic_detection_model = load_detection_model(
            mosaic_detection_model_path,
            device,
            args.fp16,
            args.detect_face_mosaics,
        )
        mosaic_restoration_model = None
    elif args.restore_from_cache:
        mosaic_detection_model = None
        mosaic_restoration_model = load_restoration_model(
            device,
            mosaic_restoration_model_name,
            mosaic_restoration_model_path,
            args.mosaic_restoration_config_path,
            args.fp16,
        )
    else:
        mosaic_detection_model, mosaic_restoration_model, preferred_pad_mode = load_models(
            device, mosaic_restoration_model_name, mosaic_restoration_model_path, args.mosaic_restoration_config_path,
            mosaic_detection_model_path, args.fp16, args.detect_face_mosaics
        )

    input_files, output_files = utils.setup_input_and_output_paths(args.input, args.output, args.output_file_pattern)

    single_file_input = len(input_files) == 1

    for input_path, output_path in zip(input_files, output_files):
        if not single_file_input:
            print(f"{os.path.basename(input_path)}:")
        try:
            process_video_file(input_path=input_path, output_path=output_path, temp_dir_path=args.temporary_directory, device=device, mosaic_restoration_model=mosaic_restoration_model, mosaic_detection_model=mosaic_detection_model,
                               mosaic_restoration_model_name=mosaic_restoration_model_name, preferred_pad_mode=preferred_pad_mode, max_clip_length=args.max_clip_length,
                               encoder=encoder, encoder_options=encoder_options, mp4_fast_start=args.mp4_fast_start,
                               progress_json=args.progress_json, cache_detections_to=args.cache_detections_to,
                               restore_from_cache=args.restore_from_cache)
        except KeyboardInterrupt:
            print(_("Received Ctrl-C, stopping restoration."))
            break

if __name__ == '__main__':
    main()
