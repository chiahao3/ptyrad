"""Tests for the JIT (torch.compile) auto-detection and its graceful fallback."""

import pytest

from ptyrad.params.recon_params import CompilerConfigs, ReconParams
from ptyrad.runtime import jit


@pytest.fixture(autouse=True)
def clear_jit_caches():
    """The capability checks are cached per-process, so clear them around every test."""
    jit.check_jit_support.cache_clear()
    jit.smoke_test_jit_compile.cache_clear()
    yield
    jit.check_jit_support.cache_clear()
    jit.smoke_test_jit_compile.cache_clear()


# --------------------------------------------------------------------------------------
# Params schema
# --------------------------------------------------------------------------------------

def test_compiler_configs_defaults_to_auto():
    configs = CompilerConfigs()
    assert configs.enable == "auto"
    assert configs.auto_smoke_test is True


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

def _patch_detection(monkeypatch, supported, smoke_test_ok=True, device_str="cpu"):
    calls = {"static": [], "smoke_test": []}

    def fake_static(device="cpu", backend="inductor"):
        calls["static"].append(device)
        return (supported, "static reason")

    def fake_smoke_test(device="cpu", **compile_kwargs):
        calls["smoke_test"].append(device)
        calls.setdefault("smoke_kwargs", []).append(compile_kwargs)
        return (smoke_test_ok, "smoke test reason")

    monkeypatch.setattr(jit, "check_jit_support", fake_static)
    monkeypatch.setattr(jit, "smoke_test_jit_compile", fake_smoke_test)
    monkeypatch.setattr(jit, "resolve_device", lambda device=None: device_str)
    return calls


def test_auto_enables_when_supported(monkeypatch):
    calls = _patch_detection(monkeypatch, supported=True, smoke_test_ok=True)
    assert jit.resolve_jit_enable({"enable": "auto"}) is True
    assert len(calls["static"]) == 1 and len(calls["smoke_test"]) == 1


def test_detection_carries_the_selected_device_into_check_and_smoke_test(monkeypatch):
    calls = _patch_detection(monkeypatch, supported=True, smoke_test_ok=True, device_str="cuda:1")
    assert jit.resolve_jit_enable({"enable": "auto"}, device="cuda:1") is True
    assert calls["static"] == ["cuda:1"]
    assert calls["smoke_test"] == ["cuda:1"]


def test_auto_falls_back_when_static_check_fails(monkeypatch):
    calls = _patch_detection(monkeypatch, supported=False)
    assert jit.resolve_jit_enable({"enable": "auto"}) is False
    assert calls["smoke_test"] == []  # No point smoke testing a machine that can't compile


def test_auto_falls_back_when_smoke_test_fails(monkeypatch):
    """A machine that looks supported but whose toolchain is broken must fall back."""
    _patch_detection(monkeypatch, supported=True, smoke_test_ok=False)
    assert jit.resolve_jit_enable({"enable": "auto"}) is False


def test_auto_smoke_test_false_skips_the_smoke_test(monkeypatch):
    calls = _patch_detection(monkeypatch, supported=True, smoke_test_ok=False)
    assert jit.resolve_jit_enable({"enable": "auto", "auto_smoke_test": False}) is True
    assert calls["smoke_test"] == []


def test_explicit_true_is_respected_even_when_unsupported(monkeypatch):
    calls = _patch_detection(monkeypatch, supported=False)
    assert jit.resolve_jit_enable({"enable": True}) is True
    assert calls["smoke_test"] == []  # Forced on, so no need to spend time smoke testing


def test_explicit_false_skips_detection_entirely(monkeypatch):
    calls = _patch_detection(monkeypatch, supported=True)
    assert jit.resolve_jit_enable({"enable": False}) is False
    assert calls["static"] == [] and calls["smoke_test"] == []


def test_missing_or_empty_configs_default_to_auto(monkeypatch):
    _patch_detection(monkeypatch, supported=True, smoke_test_ok=True)
    assert jit.resolve_jit_enable(None) is True
    assert jit.resolve_jit_enable({}) is True


def test_unrecognized_enable_is_treated_as_auto(monkeypatch):
    """Params loaded with validate=False bypass pydantic, so be forgiving at runtime."""
    _patch_detection(monkeypatch, supported=False)
    assert jit.resolve_jit_enable({"enable": "yes-please"}) is False


# --------------------------------------------------------------------------------------
# Effective torch.compile kwargs
# --------------------------------------------------------------------------------------

def test_compile_kwargs_drop_ptyrad_only_keys():
    kwargs = jit.compile_kwargs_from_configs(CompilerConfigs().model_dump())
    assert "enable" not in kwargs and "auto_smoke_test" not in kwargs
    assert kwargs["backend"] == "inductor" and kwargs["mode"] == "default"


def test_compile_kwargs_drop_mode_when_options_are_set():
    """torch.compile raises when given both, and the schema lets users set both."""
    configs = CompilerConfigs(options={"max_autotune": True}).model_dump()
    kwargs = jit.compile_kwargs_from_configs(configs)
    assert "mode" not in kwargs
    assert kwargs["options"] == {"max_autotune": True}


def test_the_resulting_kwargs_are_accepted_by_torch_compile():
    import torch

    for configs in (CompilerConfigs(), CompilerConfigs(options={"max_autotune": False})):
        kwargs = jit.compile_kwargs_from_configs(configs.model_dump())
        torch.compile(lambda x: x + 1, disable=True, **kwargs)  # Must not raise


def test_smoke_test_receives_the_same_kwargs_as_the_reconstruction(monkeypatch):
    """A bad 'mode'/'options' must fail detection, not the first reconstruction iteration."""
    calls = _patch_detection(monkeypatch, supported=True, smoke_test_ok=True)
    configs = CompilerConfigs(mode="max-autotune").model_dump()

    assert jit.resolve_jit_enable(configs) is True
    assert calls["smoke_kwargs"] == [jit.compile_kwargs_from_configs(configs)]


def test_smoke_test_ignores_a_stray_disable_flag():
    """A disabled compile would make the smoke test vacuously succeed."""
    works, reason = jit.smoke_test_jit_compile("not-a-device", disable=True)
    assert works is False and "invalid device type" in reason


def test_smoke_test_caches_per_device_and_kwargs(monkeypatch):
    runs = []

    def fake_run(device_str, compile_kwargs):
        runs.append((device_str, dict(compile_kwargs)))
        return (True, "ok")

    monkeypatch.setattr(jit, "_run_jit_smoke_test", fake_run)

    jit.smoke_test_jit_compile("cpu", backend="inductor")
    jit.smoke_test_jit_compile("cpu", backend="inductor")  # Cached
    jit.smoke_test_jit_compile("cpu", backend="inductor", options={"a": 1})
    jit.smoke_test_jit_compile("cuda:0", backend="inductor")
    assert len(runs) == 3


def test_smoke_test_leaves_the_rng_state_untouched():
    """Detection must not shift the RNG stream that the reconstruction samples from."""
    import torch

    torch.manual_seed(1234)
    before = torch.get_rng_state()
    jit.smoke_test_jit_compile.cache_clear()
    # The 'eager' backend exercises the same trace/forward/backward path without paying for
    # inductor codegen, which would add tens of seconds to the suite on a cold cache
    works, reason = jit.smoke_test_jit_compile("cpu", backend="eager")
    assert works is True, reason
    assert torch.equal(torch.get_rng_state(), before)


# --------------------------------------------------------------------------------------
# macOS opt-in policy
# --------------------------------------------------------------------------------------

def _pretend_macos(monkeypatch):
    # Patch PtyRAD's own seam rather than `platform.system`, which PyTorch calls on import
    monkeypatch.setattr(jit, "_system", lambda: "Darwin")


def test_auto_stays_eager_on_macos(monkeypatch):
    """Triton on macOS can be silently wrong, so 'auto' must not turn JIT on there."""
    calls = _patch_detection(monkeypatch, supported=True, smoke_test_ok=True)
    _pretend_macos(monkeypatch)

    assert jit.resolve_jit_enable({"enable": "auto"}) is False
    assert calls["static"] == [] and calls["smoke_test"] == []  # Policy decides before any work


def test_detect_jit_capability_reports_the_macos_policy(monkeypatch):
    _pretend_macos(monkeypatch)
    achievable, reason = jit.detect_jit_capability()
    assert achievable is False
    assert "macOS" in reason and "'enable': true" in reason


def test_explicit_true_still_enables_jit_on_macos(monkeypatch):
    calls = _patch_detection(monkeypatch, supported=True)
    _pretend_macos(monkeypatch)
    assert jit.resolve_jit_enable({"enable": True}) is True
    assert calls["smoke_test"] == []


def test_macos_policy_does_not_change_the_capability_check(monkeypatch):
    """check_jit_support answers 'can it', not 'should we', so it stays platform-neutral."""
    _pretend_macos(monkeypatch)
    monkeypatch.setattr(jit, "_has_cxx_compiler", lambda: True)
    assert jit.check_jit_support("cpu")[0] is True


def test_auto_is_unaffected_on_other_platforms(monkeypatch):
    monkeypatch.setattr(jit, "_system", lambda: "Linux")
    _patch_detection(monkeypatch, supported=True, smoke_test_ok=True)
    assert jit.resolve_jit_enable({"enable": "auto"}) is True


# --------------------------------------------------------------------------------------
# Static capability check (no torch.compile involved)
# --------------------------------------------------------------------------------------

def test_resolve_device_preserves_and_fills_the_cuda_index(monkeypatch):
    """Detection must target the GPU the reconstruction runs on, not just its device type."""
    import torch

    assert jit.resolve_device("cuda:1") == "cuda:1"
    assert jit.resolve_device("cpu") == "cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert jit.resolve_device("cuda") == "cuda:2"  # Bare 'cuda' resolves to the current GPU
    assert jit.resolve_device() == "cuda:2"


def test_check_jit_support_rejects_old_pytorch(monkeypatch):
    monkeypatch.setattr(jit, "_torch_version", lambda: (1, 13))
    supported, reason = jit.check_jit_support("cpu")
    assert supported is False
    assert "torch.compile" in reason


def _pretend_cuda_is_available(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def test_check_jit_support_requires_the_triton_package_on_cuda(monkeypatch):
    _pretend_cuda_is_available(monkeypatch)
    monkeypatch.setattr(jit, "_has_triton_package", lambda: False)
    supported, reason = jit.check_jit_support("cuda")
    assert supported is False
    assert "triton" in reason


def test_check_jit_support_rejects_old_cuda_capability(monkeypatch):
    _pretend_cuda_is_available(monkeypatch)
    monkeypatch.setattr(jit, "_has_triton_package", lambda: True)
    monkeypatch.setattr(jit, "_cuda_capability", lambda device_str: (6, 1))
    supported, reason = jit.check_jit_support("cuda")
    assert supported is False
    assert "compute capability" in reason


def test_check_jit_support_accepts_modern_cuda(monkeypatch):
    _pretend_cuda_is_available(monkeypatch)
    monkeypatch.setattr(jit, "_has_triton_package", lambda: True)
    monkeypatch.setattr(jit, "_cuda_capability", lambda device_str: (8, 6))
    supported, _ = jit.check_jit_support("cuda")
    assert supported is True


def test_check_jit_support_rejects_cuda_when_no_gpu_is_available(monkeypatch):
    import torch

    monkeypatch.setattr(jit, "_has_triton_package", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    supported, reason = jit.check_jit_support("cuda:0")
    assert supported is False
    assert "no available CUDA device" in reason


def test_check_jit_support_reads_capability_of_the_selected_gpu(monkeypatch):
    """A weak GPU elsewhere on the host must not veto JIT on the GPU actually selected."""
    capabilities = {0: (6, 1), 1: (8, 6)}
    _pretend_cuda_is_available(monkeypatch)
    monkeypatch.setattr(jit, "_has_triton_package", lambda: True)
    monkeypatch.setattr(
        jit, "_cuda_capability", lambda device_str: capabilities[int(device_str.split(":")[1])]
    )

    assert jit.check_jit_support("cuda:1")[0] is True
    assert jit.check_jit_support("cuda:0")[0] is False


def test_triton_check_is_independent_of_the_current_cuda_device(monkeypatch):
    """torch.utils._triton.has_triton() answers for the current GPU, so it must not be used here.

    Selecting cuda:1 on a host whose current device is an older cuda:0 must still enable JIT.
    """
    import torch.utils._triton as torch_triton

    _pretend_cuda_is_available(monkeypatch)
    monkeypatch.setattr(torch_triton, "has_triton_package", lambda: True)
    monkeypatch.setattr(
        torch_triton, "has_triton", lambda: pytest.fail("has_triton() is current-device dependent")
    )
    monkeypatch.setattr(jit, "_cuda_capability", lambda device_str: (8, 6))

    assert jit.check_jit_support("cuda:1")[0] is True


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


def test_smoke_test_reports_failure_instead_of_raising(monkeypatch):
    """A broken toolchain must degrade into a (False, reason) verdict, never an exception."""

    def boom(x):
        raise RuntimeError("broken backend")

    monkeypatch.setattr(jit, "_jit_smoke_test_fn", boom)
    works, reason = jit.smoke_test_jit_compile("cpu")
    assert works is False
    assert "smoke test failed" in reason


def test_smoke_test_reports_failure_on_invalid_device():
    works, reason = jit.smoke_test_jit_compile("not-a-device")
    assert works is False
    assert "invalid device type" in reason


# --------------------------------------------------------------------------------------
# torch.compile kwargs conversion
# --------------------------------------------------------------------------------------

def test_parse_torch_compile_configs_strips_ptyrad_only_keys(monkeypatch):
    from ptyrad.solver import reconstruction

    monkeypatch.setattr(
        reconstruction, "resolve_jit_enable", lambda configs, device=None, compile_kwargs=None: True
    )
    user_configs = CompilerConfigs().model_dump()
    parsed = reconstruction.parse_torch_compile_configs(user_configs, device="cpu")

    assert parsed["disable"] is False
    assert "enable" not in parsed and "auto_smoke_test" not in parsed
    assert parsed["backend"] == "inductor"
    # The user-facing params dict must not be mutated
    assert user_configs["enable"] == "auto"


def test_parse_torch_compile_configs_disables_on_fallback(monkeypatch):
    from ptyrad.solver import reconstruction

    monkeypatch.setattr(
        reconstruction, "resolve_jit_enable", lambda configs, device=None, compile_kwargs=None: False
    )
    parsed = reconstruction.parse_torch_compile_configs(CompilerConfigs().model_dump(), device="cpu")
    assert parsed["disable"] is True
