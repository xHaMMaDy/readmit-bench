"""Smoke tests — verify package imports and basic structure."""

from __future__ import annotations


def test_package_imports():
    import readmit_bench

    assert readmit_bench.__version__


def test_subpackages_importable():
    from readmit_bench import (  # noqa: F401
        api,
        data,
        drift,
        evaluation,
        explainability,
        fairness,
        features,
        models,
    )


def test_download_module_has_main():
    from readmit_bench.data import download

    assert hasattr(download, "main")
    assert hasattr(download, "list_files_for_sample")
    specs = download.list_files_for_sample(1)
    assert len(specs) == 5
    assert all(s.url.startswith("https://www.cms.gov/") for s in specs)


def test_parse_samples_arg():
    from readmit_bench.data.download import parse_samples_arg

    assert parse_samples_arg("1-5") == [1, 2, 3, 4, 5]
    assert parse_samples_arg(["1", "3", "5"]) == [1, 3, 5]
    assert parse_samples_arg(["1-3", "5"]) == [1, 2, 3, 5]
