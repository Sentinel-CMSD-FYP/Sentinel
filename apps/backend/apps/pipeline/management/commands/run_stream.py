from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.pipeline import streams
from apps.pipeline.runner import RunError


class Command(BaseCommand):
    help = ("Run the models on a live stream: a video file or frame folder (read at its own fps "
            "as a stand in for a drone feed), an rtsp:// or http:// URL, or a webcam index. "
            "Samples go out through GET /api/v1/streams/<id>/samples as they are made; at the "
            "end (or Ctrl+C) the stream is saved as a clip with its layers.")

    def add_arguments(self, parser):
        parser.add_argument("source", nargs="?", help="file, folder, URL or webcam index")
        parser.add_argument("--modules", default="all",
                            help="comma separated module names, or all (default)")
        parser.add_argument("--name", help="display name of the clip it becomes")
        parser.add_argument("--fps", type=float,
                            help="frame rate of a frame folder (default 30; a video has its own)")
        parser.add_argument("--worker", metavar="STREAM_ID",
                            help="run a stream queued by the API (the API starts this)")

    def handle(self, *args, **opts):
        if opts["worker"]:
            stream_id = opts["worker"]
        elif opts["source"]:
            source = opts["source"]
            if not (source.isdigit() or "://" in source):
                path = Path(source).resolve()
                if not path.exists():
                    raise CommandError(f"source not found: {path}")
                source = str(path)
            try:
                status = streams.create(source, opts["modules"].split(","),
                                        opts["name"] or f"Live: {Path(source).name}",
                                        fps=opts["fps"])
            except (RunError, streams.StreamBusy) as e:
                raise CommandError(str(e)) from e
            stream_id = status["stream_id"]
            self.stdout.write(f"stream {stream_id}: samples at /api/v1/streams/{stream_id}/"
                              "samples; Ctrl+C stops it")
        else:
            raise CommandError("give a source, or --worker STREAM_ID")
        status = streams.execute(stream_id)
        if status["state"] != "done":
            raise CommandError(status["error"])
        self.stdout.write(f"stream {stream_id}: {status['frames_read']} frames, "
                          f"{status['samples']} samples, clip {status['clip_id']}, "
                          f"failed {status['errors'] or 'none'}")
