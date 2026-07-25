"""Tests for the JIT (torch.compile) auto-detection and its graceful fallback."""

import pytest

from ptyrad.params.recon_params import CompilerConfigs, ReconParams
from ptyrad.runtime import jit


@pytest.fixture(autouse=True)
def clear_jit_caches():
    """The capability checks are cached per-process, so clear them around every test."""
    jit.check_jit_support.cache_clear()
    jit.probe_jit_compile.cache_clear()
    yield
    jit.check_jit_support.cache_clear()
    jit.probe_jit_compile.cache_clear()


# --------------------------------------------------------------------------------------
# Params schema
# --------------------------------------------------------------------------------------

def test_compiler_configs_defaults_to_auto():
    configs = CompilerConfigs()
    assert configs.enable == "auto"
    assert configs.auto_probe is True


def test_recon_params_defaults_to_auto():
    assert ReconParams().compiler_configs.enable == "auto"


@pytest.mark.parametrize(
    "value, expected",
    [("auto", "auto"), ("AUTO", "auto"), (" Auto ", "auto"), (True, True), (False, False)],
)
def test_compiler_configs_enable_accepted_values(value, expected):
    assert CompilerConfigs(enable=value).enable == expected


def test_compiler_configs_enable_rejects_unknown_string():
    with pytest.raises(ValueError):
        CompilerConfigs(enable="sometimes")


# --------------------------------------------------------------------------------------
# resolve_jit_enable
# --------------------------------------------------------------------------------------

def _patch_detection(monkeypatch, supported, probe_ok=True, device_type="cpu"):
    calls = {"static": 0, "probe": 0}

    def fake_static(device_type_arg="cpu", backend="inductor"):
        calls["static"] += 1
        return (supported, "static reason")

    def fake_probe(device_type_arg="cpu", backend="inductor", fullgraph=False, dynamic=None):
        calls["probe"] += 1
        return (probe_ok, "probe reason")

    monkeypatch.setattr(jit, "check_jit_support", fake_static)
    monkeypatch.setattr(jit, "probe_jit_compile", fake_probe)
    monkeypatch.setattr(jit, "resolve_device_type", lambda device=None: device_type)
    return calls


def test_auto_enables_when_supported(monkeypatch):
    calls = _patch_detection(monkeypatch, supported=True, probe_ok=True)
    assert jit.resolve_jit_enable({"enable": "auto"}) is True
    assert calls == {"static": 1, "probe": 1}


def test_auto_falls_back_when_static_check_fails(monkeypatch):
    calls = _patch_detection(monkeypatch, supported=False)
    assert jit.resolve_jit_enable({"enable": "auto"}) is False
    assert calls["probe"] == 0  # No point probing a machine that can't compile


def test_auto_falls_back_when_probe_fails(monkeypatch):
    """A machine that looks supported but whose toolchain is broken must fall back."""
    _patch_detection(monkeypatch, supported=True, probe_ok=False)
    assert jit.resolve_jit_enable({"enable": "auto"}) is False


def test_auto_probe_false_skips_the_probe(monkeypatch):
    calls = _patch_detection(monkeypatch, supported=True, probe_ok=False)
    assert jit.resolve_jit_enable({"enable": "auto", "auto_probe": False}) is True
    assert calls["probe"] == 0


def test_explicit_true_is_respected_even_when_unsupported(monkeypatch):
    calls = _patch_detection(monkeypatch, supported=False)
    assert jit.resolve_jit_enable({"enable": True}) is True
    assert calls["probe"] == 0  # Forced on, so no need to spend time probing


def test_explicit_false_skips_detection_entirely(monkeypatch):
    calls = _patch_detection(monkeypatch, supported=True)
    assert jit.resolve_jit_enable({"enable": False}) is False
    assert calls == {"static": 0, "probe": 0}


def test_missing_or_empty_configs_default_to_auto(monkeypatch):
    _patch_detection(monkeypatch, supported=True, probe_ok=True)
    assert jit.resolve_jit_enable(None) is True
    assert jit.resolve_jit_enable({}) is True


def test_unrecognized_enable_is_treated_as_auto(monkeypatch):
    """Params loaded with validate=False bypass pydantic, so be forgiving at runtime."""
    _patch_detection(monkeypatch, supported=False)
    assert jit.resolve_jit_enable({"enable": "yes-please"}) is False


# --------------------------------------------------------------------------------------
# Static capability check (no torch.compile involved)
# --------------------------------------------------------------------------------------

def test_check_jit_support_rejects_old_pytorch(monkeypatch):
    monkeypatch.setattr(jit, "_torch_version", lambda: (1, 13))
    supported, reason = jit.check_jit_support("cpu")
    assert supported is False
    assert "torch.compile" in reason


def test_check_jit_support_requires_triton_on_cuda(monkeypatch):
    monkeypatch.setattr(jit, "_has_triton", lambda: False)
    supported, reason = jit.check_jit_support("cuda")
    assert supported is False
    assert "Triton" in reason


def test_check_jit_support_rejects_old_cuda_capability(monkeypatch):
    monkeypatch.setattr(jit, "_has_triton", lambda: True)
    monkeypatch.setattr(jit, "_min_cuda_capability", lambda device_type: (6, 1))
    supported, reason = jit.check_jit_support("cuda")
    assert supported is False
    assert "compute capability" in reason


def test_check_jit_support_accepts_modern_cuda(monkeypatch):
    monkeypatch.setattr(jit, "_has_triton", lambda: True)
    monkeypatch.setattr(jit, "_min_cuda_capability", lambda device_type: (8, 6))
    supported, _ = jit.check_jit_support("cuda")
    assert supported is True


def test_check_jit_support_rejects_mps_on_old_pytorch(monkeypatch):
    monkeypatch.setattr(jit, "_torch_version", lambda: (2, 5))
    supported, reason = jit.check_jit_support("mps")
    assert supported is False
    assert "Metal" in reason


def test_check_jit_support_requires_cxx_compiler_on_cpu(monkeypatch):
    monkeypatch.setattr(jit, "_has_cxx_compiler", lambda: False)
    supported, reason = jit.check_jit_support("cpu")
    assert supported is False
    assert "C++ compiler" in reason


def test_probe_reports_failure_instead_of_raising(monkeypatch):
    """A broken toolchain must degrade into a (False, reason) verdict, never an exception."""

    def boom(x):
        raise RuntimeError("broken backend")

    monkeypatch.setattr(jit, "_jit_probe_fn", boom)
    works, reason = jit.probe_jit_compile("cpu")
    assert works is False
    assert "probe failed" in reason


def test_probe_reports_failure_on_invalid_device():
    works, reason = jit.probe_jit_compile("not-a-device")
    assert works is False
    assert "invalid device type" in reason


# --------------------------------------------------------------------------------------
# torch.compile kwargs conversion
# --------------------------------------------------------------------------------------

def test_parse_torch_compile_configs_strips_ptyrad_only_keys(monkeypatch):
    from ptyrad.solver import reconstruction

    monkeypatch.setattr(reconstruction, "resolve_jit_enable", lambda configs, device=None: True)
    user_configs = CompilerConfigs().model_dump()
    parsed = reconstruction.parse_torch_compile_configs(user_configs, device="cpu")

    assert parsed["disable"] is False
    assert "enable" not in parsed and "auto_probe" not in parsed
    assert parsed["backend"] == "inductor"
    # The user-facing params dict must not be mutated
    assert user_configs["enable"] == "auto"


def test_parse_torch_compile_configs_disables_on_fallback(monkeypatch):
    from ptyrad.solver import reconstruction

    monkeypatch.setattr(reconstruction, "resolve_jit_enable", lambda configs, device=None: False)
    parsed = reconstruction.parse_torch_compile_configs(CompilerConfigs().model_dump(), device="cpu")
    assert parsed["disable"] is True
