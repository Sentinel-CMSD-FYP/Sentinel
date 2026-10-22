from django.core.management.base import BaseCommand, CommandError

from apps.pipeline.jobs import execute


class Command(BaseCommand):
    help = ("Worker for a run started with POST /api/v1/clips/<id>/runs (the API starts it; "
            "not meant to be typed). Runs every module and keeps .runs/<run_id>/status.json "
            "current.")

    def add_arguments(self, parser):
        parser.add_argument("run_id")

    def handle(self, *args, **opts):
        status = execute(opts["run_id"])
        if status["state"] != "done":
            raise CommandError(status["error"])
        self.stdout.write(f"run {status['run_id']}: layers {status['layers']}, "
                          f"failed {status['errors'] or 'none'}")
