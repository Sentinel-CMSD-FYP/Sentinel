from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.pipeline.camera import ALTITUDE_SOURCES, SENSOR_ASPECTS, camera_block
from apps.pipeline.ingest import ingest_clip
from apps.pipeline.runner import RunError


class Command(BaseCommand):
    help = (
        "Make a clip under CLIPS_ROOT from a video file or a folder of frames, e.g. "
        "python manage.py ingest_clip data/dronecrowd/test_data-001/test_data/images seq011 "
        "--seq 011 --fps 25"
    )

    def add_arguments(self, parser):
        parser.add_argument("source", type=Path, help="mp4 file, or a folder of numbered images")
        parser.add_argument("clip_id")
        parser.add_argument("--seq", help="keep only img<seq>* in a frame folder (DroneCrowd)")
        parser.add_argument("--fps", type=float,
                            help="frame rate; needed for a frame folder (default 30)")
        parser.add_argument("--name", help="label shown in the viewer (default: the clip id)")
        cam = parser.add_argument_group(
            "camera (optional; with --altitude and a FOV, flow is in m/s instead of px/frame)")
        cam.add_argument("--altitude", type=float, help="height above the ground in metres")
        fov = cam.add_mutually_exclusive_group()
        fov.add_argument("--hfov", type=float, help="horizontal field of view in degrees")
        fov.add_argument("--dfov", type=float,
                         help="diagonal field of view in degrees (as on spec sheets)")
        cam.add_argument("--sensor-aspect", default="4:3", choices=sorted(SENSOR_ASPECTS),
                         help="sensor shape the --dfov belongs to (default 4:3)")
        cam.add_argument("--camera-model", help="drone or camera name, e.g. 'DJI Mini 3'")
        cam.add_argument("--altitude-source", default="assumed", choices=ALTITUDE_SOURCES,
                         help="how the altitude is known (default assumed)")

    def handle(self, *args, **opts):
        source = opts["source"]
        if source.is_dir() and opts["fps"] is None:
            self.stdout.write("note: a frame folder has no frame rate; assuming 30 fps")
        camera = None
        if opts["altitude"] is not None or opts["hfov"] is not None or opts["dfov"] is not None:
            if opts["altitude"] is None:
                raise CommandError("--altitude is needed with --hfov or --dfov")
            try:
                camera = camera_block(opts["altitude"], hfov_deg=opts["hfov"],
                                      dfov_deg=opts["dfov"], sensor_aspect=opts["sensor_aspect"],
                                      model=opts["camera_model"],
                                      altitude_source=opts["altitude_source"])
            except ValueError as e:
                raise CommandError(str(e)) from e
        try:
            clip = ingest_clip(Path(settings.CLIPS_ROOT), source, opts["clip_id"],
                               seq=opts["seq"], fps=opts["fps"], name=opts["name"],
                               camera=camera)
        except RunError as e:
            raise CommandError(str(e)) from e
        w, h = clip["frame_size"]
        self.stdout.write(f"ingested clip {clip['clip_id']!r}: {clip['n_frames']} frames at "
                          f"{clip['fps']:g} fps, {w}x{h}")
        if camera:
            self.stdout.write(f"camera: {camera['altitude_m']:g} m "
                              f"({camera['altitude_source']}), horizontal FOV "
                              f"{camera['hfov_deg']:g} degrees")
