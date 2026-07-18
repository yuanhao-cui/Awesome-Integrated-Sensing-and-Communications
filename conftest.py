"""Repository-wide pytest lifecycle hooks."""

import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def _cleanup_torch_jit_instantiator_tempdir():
    """Close a PyTorch 2.5.1 module-level temporary directory explicitly.

    ``torch.distributed.nn.jit.instantiator`` creates ``_TEMP_DIR`` at import
    time and otherwise leaves its ``TemporaryDirectory`` weakref to run during
    interpreter shutdown.  Under ``-W error`` that third-party cleanup emits a
    ResourceWarning after an otherwise clean pytest run.  Do not import the
    optional module here; clean it only when a test path already loaded it.
    """

    yield
    module = sys.modules.get("torch.distributed.nn.jit.instantiator")
    temporary_directory = getattr(module, "_TEMP_DIR", None)
    if temporary_directory is not None:
        temporary_directory.cleanup()
