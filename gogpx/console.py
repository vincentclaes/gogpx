from __future__ import annotations

import itertools
import sys


class ConsoleUX:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._spinner = itertools.cycle(["|", "/", "-", "\\"])
        self._last_len = 0

    def tick(self, stage: str, note: str = "") -> None:
        if not self.enabled:
            return
        spin = next(self._spinner)
        msg = f"{spin} {stage}"
        if note:
            msg += f" · {note}"
        pad = " " * max(0, self._last_len - len(msg))
        sys.stdout.write(f"\r\033[36m{msg}\033[0m{pad}")
        sys.stdout.flush()
        self._last_len = len(msg)

    def log(self, msg: str) -> None:
        if not self.enabled:
            return
        sys.stdout.write(f"\n\033[90m{msg}\033[0m\n")
        sys.stdout.flush()

    def done(self, where: str, learned: str, next_step: str) -> None:
        if not self.enabled:
            return
        sys.stdout.write("\r\033[32mOK\033[0m\n")
        sys.stdout.write(f"Where: {where}\n")
        sys.stdout.write(f"Learned: {learned}\n")
        sys.stdout.write(f"Next: {next_step}\n")
        sys.stdout.flush()
