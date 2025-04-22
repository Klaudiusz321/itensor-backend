#!/usr/bin/env python
import os
import cProfile
import sys

# 1) Point Django at your settings module
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "myproject.settings"      # ← change this if your settings live elsewhere
)

# 2) Now import and initialize Django
import django
django.setup()

# 3) Import the management executor
from django.core.management import execute_from_command_line

if __name__ == "__main__":
    prof = cProfile.Profile()
    prof.enable()

    # 4) Run the “runserver” command (just like `manage.py`)
    execute_from_command_line(["manage.py", "runserver"])

    prof.disable()
    prof.dump_stats("mhd_run.pstats")
    print("Wrote profiling data to mhd_run.pstats")
