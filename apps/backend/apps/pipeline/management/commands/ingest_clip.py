from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

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

    def handle(self, *args, **opts):
        source = opts["source"]
        if source.is_dir() and opts["fps"] is None:
            self.stdout.write("note: a frame folder has no frame rate; assuming 30 fps")
        try:
            clip = ingest_clip(Path(settings.CLIPS_ROOT), source, opts["clip_id"],
                               seq=opts["seq"], fps=opts["fps"], name=opts["name"])
        except RunError as e:
            raise CommandError(str(e)) from e
        w, h = clip["frame_size"]
        self.stdout.write(f"ingested clip {clip['clip_id']!r}: {clip['n_frames']} frames at "
                          f"{clip['fps']:g} fps, {w}x{h}")
