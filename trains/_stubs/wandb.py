"""Fallback stub for `import wandb`.

scripts/base_train.py imports wandb unconditionally, but only ever *uses* it when
--run != "dummy" (otherwise it swaps in nanochat.common.DummyWandb). This stub only
exists to satisfy the import on machines without wandb installed; the launcher puts
it on PYTHONPATH only when the real package is missing AND WANDB_RUN=dummy.
"""

__version__ = "0.0.0-stub"


class _StubRun:
    def log(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


def init(*args, **kwargs):
    print("[wandb stub] real wandb is not installed; logging to wandb is disabled.")
    return _StubRun()


def login(*args, **kwargs):
    return False
