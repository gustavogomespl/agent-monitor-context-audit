"""Hardware eligibility from synthetic nvidia-smi output, without GPU access."""

from types import SimpleNamespace

import pytest

from context_audit import colab


def _gpu_output(monkeypatch, output):
    monkeypatch.setattr(
        colab.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=output)
    )


@pytest.mark.parametrize(
    "name,memory,capability",
    [
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 97887, "12.0"),
        ("NVIDIA H100 80GB HBM3", 81559, "9.0"),
        ("NVIDIA A100 80GB PCIe", 81920, "8.0"),
    ],
)
def test_bf16_gpu_with_sufficient_vram_is_accepted_and_recorded(
    monkeypatch, name, memory, capability
):
    _gpu_output(monkeypatch, f"{name}, {memory}, synthetic-driver, {capability}\n")
    hardware = colab._hardware()
    assert hardware == {
        "name": name,
        "memory_mib": memory,
        "driver": "synthetic-driver",
        "compute_capability": capability,
    }


@pytest.mark.parametrize(
    "output",
    [
        "NVIDIA RTX 5090, 32768, synthetic-driver, 12.0\n",
        "Synthetic older architecture, 81920, synthetic-driver, 7.0\n",
        "NVIDIA H100, 81920, synthetic-driver, 9.0\n" * 2,
        "NVIDIA H100, nan, synthetic-driver, 9.0\n",
        "NVIDIA H100, 81920, synthetic-driver, N/A\n",
        "NVIDIA H100, 81920, synthetic-driver\n",
        "",
    ],
)
def test_ineligible_or_unverified_gpu_is_rejected_before_launch(monkeypatch, output):
    _gpu_output(monkeypatch, output)
    with pytest.raises(ValueError):
        colab._hardware()
