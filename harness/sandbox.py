"""bwrap sandbox for LLM-written code (adapted from an earlier lab project).

The agents themselves run on the host (inside bwrap they would lose the API).
What runs here is the code they WRITE. The sandbox's job is what hooks cannot
promise:

* $HOME is replaced by an empty tmpfs, which hides the answer keys
  ($REPRO_PRIVATE), the paper and raw data, ~/.claude and every credential.
  Only the paths a job needs are bound back, read-only unless stated.
* No network unless the job is a fetch.
* Exactly one GPU device node; CUDA_VISIBLE_DEVICES is the GPU's UUID, so
  device enumeration order inside cannot select a different card.
* Fresh pid namespace and /proc, so host processes (and their /proc/<pid>/root)
  are invisible.

Not handled here: CPU pinning and memory limits (run_iteration does those with
taskset and an RSS watchdog; `prlimit --as` breaks CUDA initialisation).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


NVIDIA_SHARED = ["/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"]
SYSTEM_DIRS = ["/usr", "/etc", "/opt", "/sys", "/run/nvidia-persistenced"]
# A fetch job needs a resolver. /etc/resolv.conf is often a symlink into /run (systemd-resolved
# stub at 127.0.0.53, reachable because the network namespace is shared), and /run is not bound.
RESOLVER_DIRS = ["/run/systemd/resolve", "/run/resolvconf", "/run/NetworkManager"]
SYSTEM_LINKS = {"/bin": "usr/bin", "/sbin": "usr/sbin", "/lib": "usr/lib",
                "/lib64": "usr/lib64", "/lib32": "usr/lib32"}


def _real_system_dirs() -> list[str]:
    """/bin etc. are real directories on older distros; bind those instead of symlinking."""
    return [d for d in SYSTEM_LINKS if Path(d).is_dir() and not Path(d).is_symlink()]


@dataclass
class Mounts:
    ro: dict[str, str] = field(default_factory=dict)   # host -> sandbox
    rw: dict[str, str] = field(default_factory=dict)


def python_roots() -> list[Path]:
    """Directories the interpreter needs: the venv and the base install it links to."""
    roots = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(),
             Path(os.path.realpath(sys.executable)).parent.parent}
    return sorted(roots)


def build_argv(command: list[str], *, mounts: Mounts, network: bool = False,
               gpu_device: str | None = None, gpu_uuid: str | None = None,
               env: dict[str, str] | None = None, cwd: str = "/work",
               threads: int | None = None, scratch: Path | None = None) -> list[str]:
    """bwrap argv. `scratch` (a host dir) backs /tmp and /dev/shm.

    bwrap 0.4 cannot size a tmpfs, and tmpfs pages do not show in a process's
    RSS: an unbounded /tmp is a way to eat the shared server's RAM. With
    `scratch`, /tmp and /dev/shm live on disk in a directory the runner watches
    and deletes; without it (short host-side jobs) they stay tmpfs.
    """
    home = str(Path.home())
    argv = ["bwrap", "--die-with-parent", "--new-session",
            "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup-try"]
    if not network:
        argv.append("--unshare-net")
    # A fresh root holding only system directories: /home, /var, /root and
    # every other user's files simply do not exist inside.
    for d in SYSTEM_DIRS:
        if Path(d).exists():
            argv += ["--ro-bind", d, d]
    if network:
        for d in RESOLVER_DIRS:
            if Path(d).is_dir():
                argv += ["--ro-bind", d, d]
    for link in SYSTEM_LINKS:
        if Path(link).is_symlink():
            argv += ["--symlink", os.readlink(link), link]
    for d in _real_system_dirs():
        argv += ["--ro-bind", d, d]
    argv += ["--proc", "/proc", "--dev", "/dev"]
    if scratch is not None:
        for sub_dir in ("tmp", "shm"):
            (Path(scratch) / sub_dir).mkdir(parents=True, exist_ok=True)
        argv += ["--bind", str(Path(scratch) / "tmp"), "/tmp", "--bind", str(Path(scratch) / "shm"), "/dev/shm"]
    else:
        argv += ["--tmpfs", "/tmp", "--tmpfs", "/dev/shm"]
    argv += ["--tmpfs", home]
    masked = [home]
    # Belt and braces: if the private root is not under $HOME (tests, custom
    # REPRO_PRIVATE), hide it explicitly.
    private = str(Path.home() / "ag_harness_private")
    if not private.startswith(home.rstrip("/") + "/"):
        argv += ["--tmpfs", private]
        masked.append(private)
    for root in python_roots():
        argv += ["--ro-bind", str(root), str(root)]
    for host, inside in mounts.ro.items():
        argv += ["--ro-bind", host, inside]
    for host, inside in mounts.rw.items():
        argv += ["--bind", host, inside]
    # The masking tmpfs mounts must not become writable RAM either. Non-recursive,
    # so the read-only binds placed under $HOME (venv, interpreter) are unaffected.
    for m in masked:
        argv += ["--remount-ro", m]
    if gpu_device:
        for dev in [gpu_device, *NVIDIA_SHARED]:
            if Path(dev).exists():
                argv += ["--dev-bind", dev, dev]
    base_env = {
        "PATH": f"{Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp", "TMPDIR": "/tmp", "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "MPLCONFIGDIR": "/tmp/mpl",
        "CUDA_VISIBLE_DEVICES": gpu_uuid or "",
    }
    if threads:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            base_env[var] = str(threads)
    # bwrap 0.4 (this server) has no --clearenv; `env -i` gives the same
    # guarantee: nothing from the caller's environment (API keys, tokens,
    # REPRO_PRIVATE) reaches the job unless listed here.
    argv += ["--chdir", cwd, "--", "/usr/bin/env", "-i"]
    argv += [f"{k}={v}" for k, v in {**base_env, **(env or {})}.items()]
    return argv + command
