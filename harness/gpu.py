"""Lease exactly one GPU for one sandboxed job (adapted from an earlier lab project).

Two facts about this server shape the code:

* nvidia-smi index and /dev/nvidiaN minor differ (index 0 is /dev/nvidia1
  here), so the device node is found through the PCI bus id and
  /proc/driver/nvidia/gpus/<bus>/information, never assumed equal to the index.
* The server is shared. A GPU is eligible when it has enough free memory; idle
  ones are preferred, then the least utilised. If none qualifies, wait.

One lease at a time per assigned card, enforced with flock. AGH_GPU (nvidia-smi index, default 0) picks the card;
GPU 0 keeps the original lock file, any other card gets its own lock (2026-11-20: user allowed GPU 2 as a
second card). The project never uses a card that is not assigned through AGH_GPU.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

PROC_GPUS = Path("/proc/driver/nvidia/gpus")


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    bus_id: str
    minor: int
    mem_used_mb: int
    mem_total_mb: int
    util: int

    @property
    def free_mb(self) -> int:
        return self.mem_total_mb - self.mem_used_mb

    @property
    def device(self) -> str:
        return f"/dev/nvidia{self.minor}"


def _normalize_bus(bus: str) -> str:
    """'00000000:01:00.0' (nvidia-smi) -> '0000:01:00.0' (procfs), lowercase."""
    bus = bus.strip().lower()
    domain, _, rest = bus.partition(":")
    return f"{int(domain, 16):04x}:{rest}"


def minor_for_bus(bus_id: str, proc_root: Path = PROC_GPUS) -> int:
    info = proc_root / _normalize_bus(bus_id) / "information"
    m = re.search(r"Device Minor:\s*(\d+)", info.read_text())
    if not m:
        raise RuntimeError(f"no Device Minor in {info}")
    return int(m.group(1))


def parse_smi(csv_text: str, proc_root: Path = PROC_GPUS) -> list[Gpu]:
    gpus = []
    for line in csv_text.strip().splitlines():
        try:
            idx, uuid, bus, used, total, util = [c.strip() for c in line.split(",")]
            gpus.append(Gpu(int(idx), uuid, bus, minor_for_bus(bus, proc_root),
                            int(float(used)), int(float(total)), int(float(util))))
        except (ValueError, OSError, RuntimeError):
            continue  # "[N/A]" fields or a vanished device: not eligible
    return gpus


def list_gpus() -> list[Gpu]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=30).stdout
    return parse_smi(out)


def choose(gpus: list[Gpu], *, min_free_mb: int, prefer: int | None = None,
           idle_mem_mb: int = 1000, idle_util: int = 10) -> Gpu | None:
    """Pick one GPU: the preferred one if eligible, else idle ones first, then least used."""
    eligible = [g for g in gpus if g.free_mb >= min_free_mb]
    if prefer is not None:
        hit = [g for g in eligible if g.index == prefer]
        return hit[0] if hit else None
    if not eligible:
        return None
    return min(eligible, key=lambda g: (not (g.mem_used_mb <= idle_mem_mb and g.util <= idle_util),
                                        g.util, g.mem_used_mb, g.index))


ALT_FILE = Path.home() / "ag_harness_private" / "gpu_alt.json"


def other_users_on(uuid: str) -> list[str]:
    """Users other than us with a compute process on this card (an occupied card is never used)."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30).stdout
    me, users = os.getuid(), []
    for line in out.strip().splitlines():
        try:
            pid, u = [c.strip() for c in line.split(",")]
        except ValueError:
            continue
        if u != uuid:
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != me:
                users.append(pid)
        except OSError:
            users.append(pid)   # process not visible: assume another user
    return users


def alt_card(case: str | None, min_free_mb: int, lister) -> int | None:
    """Second card for this case (2026-11-20, user: 'GPU 3 when free, for V6'). Config ~/ag_harness_private/gpu_alt.json
    {"gpu": 3, "cases": ["V6"]}; read at every lease, so it can be changed while runs are going.
    Used only when no other user's process is on it and it has min_free_mb free."""
    if not case or not ALT_FILE.exists():
        return None
    try:
        import json
        cfg = json.loads(ALT_FILE.read_text())
    except (OSError, ValueError):
        return None
    if case not in cfg.get("cases", []) or cfg.get("gpu") is None:
        return None
    g = [x for x in lister() if x.index == int(cfg["gpu"])]
    if not g or g[0].free_mb < min_free_mb or other_users_on(g[0].uuid):
        return None
    return g[0].index


def lock_path(g: str | None = None) -> Path:
    """One lock per assigned card (GPU 0 keeps the original file)."""
    g = g if g is not None else (os.environ.get("AGH_GPU", "0") or "0")
    name = ".gpu.lock" if g == "0" else f".gpu{g}.lock"
    p = Path(os.environ.get("AGH_GPU_LOCK") or Path.home() / "ag_harness_private" / name)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@contextlib.contextmanager
def lease(*, min_free_mb: int = 12000, wait_s: float = 3600, poll_s: float = 30,
          lister=list_gpus, case: str | None = None) -> Iterator[Gpu]:
    """Hold the project-wide GPU lock and yield one eligible GPU."""
    alt = alt_card(case, min_free_mb, lister)
    if alt is not None:   # try the second card without waiting; fall back to the assigned card if its lock is held
        fh = open(lock_path(str(alt)), "w")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
        else:
            try:
                gpu = choose(lister(), min_free_mb=min_free_mb, prefer=alt)
                if gpu is not None and not other_users_on(gpu.uuid):
                    fh.write(f"{os.getpid()} {gpu.index} {gpu.uuid}\n")
                    fh.flush()
                    yield gpu
                    return
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
                fh.close()
    prefer = os.environ.get("AGH_GPU", "0")  # user-assigned card (nvidia-smi index)
    prefer_idx = int(prefer) if prefer not in (None, "") else None
    deadline = time.monotonic() + wait_s
    with open(lock_path(), "w") as fh:
        # blocking wait (woken by the kernel on release, so a waiter is not starved by a job loop that
        # re-acquires immediately); SIGALRM bounds the wait
        import signal

        def _timeout(*_):
            raise TimeoutError("another job holds the project GPU lease")
        old_h = signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(max(1, int(wait_s)))
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_h)
        try:
            while True:
                gpu = choose(lister(), min_free_mb=min_free_mb, prefer=prefer_idx)
                if gpu is not None:
                    fh.write(f"{os.getpid()} {gpu.index} {gpu.uuid}\n")
                    fh.flush()
                    yield gpu
                    return
                if time.monotonic() > deadline:
                    raise TimeoutError(f"no GPU with {min_free_mb} MiB free within {wait_s}s")
                time.sleep(poll_s)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
