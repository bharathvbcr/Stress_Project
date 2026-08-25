"""
Tests for the producer/consumer contract between preprocess_all_subjects() and
load_preprocessed_data().

The preprocessing pipeline writes one '{subj_id}_processed.joblib' per subject plus
a 'processed_file_map.joblib' index; the windowing/split stages consume a
{subject_id: path} map and load each subject lazily. These tests pin the loader to
that contract - it previously read a 'processed_aligned_data.joblib' aggregate that
the pipeline no longer produces.
"""

import os
import sys

import joblib
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (  # noqa: E402
    PROCESSED_FILE_MAP_NAME,
    load_preprocessed_data,
    resolve_processed_file_map,
)

SUBJECTS = ["2", "17", "NURSE_5C"]


def _make_config(tmp_path):
    processed_dir = tmp_path / "processed_data"
    static_dir = tmp_path / "static_features"
    processed_dir.mkdir()
    static_dir.mkdir()
    return {
        "save_paths": {
            "processed_data": str(processed_dir),
            "static_features_results": str(static_dir),
        }
    }, processed_dir, static_dir


def _write_subject_files(processed_dir, subjects=SUBJECTS):
    """Write per-subject payloads exactly as preprocess_all_subjects() does."""
    paths = {}
    for subj_id in subjects:
        path = processed_dir / f"{subj_id}{'_processed.joblib'}"
        joblib.dump({"signal": {"chest": {"ECG": [0.0, 1.0]}}, "subject": subj_id}, path)
        paths[subj_id] = str(path)
    return paths


def test_returns_path_map_not_signal_data(tmp_path):
    """The first return value must be paths - windowing calls joblib.load() on it."""
    config, processed_dir, _ = _make_config(tmp_path)
    written = _write_subject_files(processed_dir)
    joblib.dump(written, processed_dir / PROCESSED_FILE_MAP_NAME)

    processed_file_paths, _, _ = load_preprocessed_data(config)

    assert processed_file_paths is not None
    assert set(processed_file_paths) == set(SUBJECTS)
    for subj_id, path in processed_file_paths.items():
        assert isinstance(path, str), f"subject {subj_id} mapped to {type(path).__name__}, not a path"
        assert os.path.exists(path)
        # The downstream contract: the path must be loadable by the windowing stage.
        assert joblib.load(path)["subject"] == subj_id


def test_stale_absolute_paths_are_reanchored(tmp_path):
    """
    A map written on another machine/OS stores absolute paths that do not exist here.
    Entries must be re-anchored to the configured directory by filename.
    """
    config, processed_dir, _ = _make_config(tmp_path)
    _write_subject_files(processed_dir)
    stale = {
        subj_id: f"C:\\Users\\someone\\StressProject\\outputs\\processed_data\\{subj_id}_processed.joblib"
        for subj_id in SUBJECTS
    }
    joblib.dump(stale, processed_dir / PROCESSED_FILE_MAP_NAME)

    resolved = resolve_processed_file_map(config)

    assert set(resolved) == set(SUBJECTS)
    for path in resolved.values():
        assert os.path.exists(path)
        assert os.path.dirname(path) == str(processed_dir)


def test_falls_back_to_directory_scan_without_map(tmp_path):
    """The per-subject files are the source of truth; the map is only an index."""
    config, processed_dir, _ = _make_config(tmp_path)
    _write_subject_files(processed_dir)

    resolved = resolve_processed_file_map(config)

    assert set(resolved) == set(SUBJECTS)
    assert all(os.path.exists(p) for p in resolved.values())


def test_unresolvable_entries_are_dropped_not_reported_as_present(tmp_path):
    """A file that is gone must not appear in the map - a path that cannot be loaded
    must never be handed downstream as if it were usable."""
    config, processed_dir, _ = _make_config(tmp_path)
    written = _write_subject_files(processed_dir)
    written["99"] = str(processed_dir / "99_processed.joblib")  # never written
    joblib.dump(written, processed_dir / PROCESSED_FILE_MAP_NAME)

    resolved = resolve_processed_file_map(config)

    assert "99" not in resolved
    assert set(resolved) == set(SUBJECTS)


def test_missing_directory_returns_empty_not_crash(tmp_path):
    config = {"save_paths": {"processed_data": str(tmp_path / "does_not_exist")}}
    assert resolve_processed_file_map(config) == {}

    processed_file_paths, _, _ = load_preprocessed_data(config)
    assert not processed_file_paths


def test_legacy_aggregate_file_is_not_consulted(tmp_path):
    """
    The retired 'processed_aligned_data.joblib' monolith must not be used: loading it
    puts the whole cohort in memory (the OOM the incremental refactor removed) and
    yields signal data where downstream code expects paths.
    """
    config, processed_dir, _ = _make_config(tmp_path)
    joblib.dump(
        {subj_id: {"signal": {"chest": {"ECG": [0.0]}}} for subj_id in SUBJECTS},
        processed_dir / "processed_aligned_data.joblib",
    )

    processed_file_paths, _, _ = load_preprocessed_data(config)

    assert not processed_file_paths, "loader fell back to the retired aggregate artifact"


def test_static_features_and_rpeaks_still_load(tmp_path):
    """The other two return values keep their existing contract (real data)."""
    config, processed_dir, static_dir = _make_config(tmp_path)
    _write_subject_files(processed_dir)
    joblib.dump({"2": "static-frame"}, static_dir / "static_features_results.joblib")
    joblib.dump({"2": [10, 20, 30]}, static_dir / "r_peak_indices.joblib")

    _, static_features, r_peaks = load_preprocessed_data(config)

    assert static_features == {"2": "static-frame"}
    assert r_peaks == {"2": [10, 20, 30]}


def test_clear_files_removes_per_subject_files_and_map(tmp_path):
    config, processed_dir, static_dir = _make_config(tmp_path)
    written = _write_subject_files(processed_dir)
    joblib.dump(written, processed_dir / PROCESSED_FILE_MAP_NAME)
    joblib.dump({"2": "static-frame"}, static_dir / "static_features_results.joblib")

    load_preprocessed_data(config, clear_files=True)

    for path in written.values():
        assert not os.path.exists(path)
    assert not (processed_dir / PROCESSED_FILE_MAP_NAME).exists()
    assert not (static_dir / "static_features_results.joblib").exists()


def test_clear_files_preserves_unregenerable_legacy_aggregate(tmp_path):
    """Nothing in the pipeline writes the aggregate any more, so clearing must not
    destroy it."""
    config, processed_dir, _ = _make_config(tmp_path)
    _write_subject_files(processed_dir)
    legacy = processed_dir / "processed_aligned_data.joblib"
    joblib.dump({"2": {"signal": {}}}, legacy)

    load_preprocessed_data(config, clear_files=True)

    assert legacy.exists()


def test_tuning_has_no_import_time_data_loading():
    """
    Importing tuning must not require data on disk. It previously called
    initialize_tuning_data() at module scope, so `import tuning` raised outright
    whenever the processed artifacts were absent - which is why the notebook's
    'from tuning import run_tuning' setup cell died too.

    Checked by parsing rather than importing, so it holds without optuna installed.
    """
    import ast

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tree = ast.parse(open(os.path.join(repo_root, "tuning.py"), encoding="utf-8").read())

    module_level_calls = [
        node.value.func.id
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
    ]
    assert "initialize_tuning_data" not in module_level_calls, (
        "tuning.py loads data at import time; importing the module must stay side-effect free"
    )

    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"initialize_tuning_data", "run_tuning", "objective"} <= functions


def test_tuning_imports_without_preprocessed_data(monkeypatch):
    """Live counterpart of the parse check, when optuna is available."""
    pytest.importorskip("optuna")
    monkeypatch.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.modules.pop("tuning", None)

    import tuning  # noqa: F401  - the import itself is the assertion

    assert tuning.train_data_sampled_global is None, "import performed data loading as a side effect"
    assert callable(tuning.initialize_tuning_data)


def test_directory_with_glob_metacharacters(tmp_path):
    """A directory named like a glob pattern must still scan correctly."""
    odd_root = tmp_path / "run[1]"
    odd_root.mkdir()
    config, processed_dir, _ = _make_config(odd_root)
    _write_subject_files(processed_dir)

    resolved = resolve_processed_file_map(config)

    assert set(resolved) == set(SUBJECTS)
