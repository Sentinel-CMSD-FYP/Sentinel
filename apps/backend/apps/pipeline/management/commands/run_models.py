import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.clips.store import ClipStore, NotFound
from apps.pipeline.runner import RunError, build_modules, run_models


class Command(BaseCommand):
    help = (
        "Run perception modules on a clip under CLIPS_ROOT and add their layers to it, e.g. "
        "python manage.py run_models seq011 --modules density,infrastructure"
    )

    def add_arguments(self, parser):
        parser.add_argument("clip_id")
        parser.add_argument("--modules", required=True,
                            help="comma separated module names, e.g. density,infrastructure, "
                                 "or all")
        parser.add_argument("--device", help="cpu or cuda; default: each module's config")
        parser.add_argument("--replace", action="store_true",
                            help="overwrite layers the clip already has")
        parser.add_argument("--report", type=Path,
                            help="also write the added layers and the failed modules to this "
                                 "JSON file (read back by a run on Kaggle)")

    def handle(self, *args, **opts):
        names = [n.strip() for n in opts["modules"].split(",") if n.strip()]
        store = ClipStore(settings.CLIPS_ROOT)
        try:
            clip = store.get_clip(opts["clip_id"])
        except NotFound as e:
            raise CommandError(str(e)) from e
        errors: dict[str, str] = {}  # a failing module is noted here and the others go on
        try:
            modules = build_modules(names, opts["device"], clip.get("fps"), errors,
                                    clip.get("camera"))
            lines = []
            if modules:
                lines = run_models(store, opts["clip_id"], modules, opts["replace"], errors)
        except RunError as e:
            raise CommandError(str(e)) from e
        for line in lines:
            self.stdout.write(line)
        if opts["report"]:
            added = [m.name for m in modules if m.name not in errors]
            opts["report"].write_text(json.dumps({"layers": added, "errors": errors}, indent=2),
                                      encoding="utf-8")
        if errors:
            raise CommandError("modules failed (their layers were not added): "
                               + "; ".join(f"{name}: {why}" for name, why in errors.items()))
