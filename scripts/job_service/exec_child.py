"""Linux parent-owned executable launcher, safe to spawn from a threaded service."""

import os
import sys
from .worker import parent_death_signal

if __name__ == "__main__":
    parent_death_signal(int(sys.argv[1]))
    os.execv(sys.argv[2], sys.argv[2:])
