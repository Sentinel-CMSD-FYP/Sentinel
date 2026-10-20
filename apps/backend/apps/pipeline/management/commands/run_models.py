from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.clips.store import ClipStore
from apps.pipeline.runner import RunError, build_modules, run_models


class Command(BaseCommand):
    help = (
        "Run perception modules on a clip under CLIPS_ROOT and add their layers to it, e.g. "
        "python manage.py run_models seq011 --modules density,infrastructure"
    )

    def add_arguments(self, parser):
        parser.add_argument("clip_id")
        parser.add_argument("--modules", required=True,
                            help="comma separated module names, e.g. density,infrastructure")
        parser.add_argument("--device", help="cpu or cuda; default: each module's config")
        parser.add_argument("--replace", action="store_true",
                            help="overwrite layers the clip already has")

    def handle(self, *args, **opts):
        names = [n.strip() for n in opts["modules"].split(",") if n.strip()]
        try:
            modules = build_modules(names, opts["device"])
            lines = run_models(ClipStore(settings.CLIPS_ROOT), opts["clip_id"], modules,
                               replace=opts["replace"])
        except RunError as e:
            raise CommandError(str(e)) from e
        for line in lines:
            self.stdout.write(line)
