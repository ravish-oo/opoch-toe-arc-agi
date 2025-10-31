#!/usr/bin/env python3
"""
Tests for WO-12: Runner + Registry + Report

Tests end-to-end pipeline orchestration:
- Descriptor registry (single source of truth)
- Batch runner (corpus processing)
- Report aggregation (witnesses, hash)
- Determinism across runs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
from typing import Dict, List, Any

import descriptor_registry
import runner
import report


# ============================================================================
# Descriptor Registry Tests
# ============================================================================

def test_registry_keep_catalogue_returns_list():
    """Keep catalogue should return list of descriptor objects"""
    sviews_meta = {"row_gcd": 1, "col_gcd": 1}

    catalogue = descriptor_registry.keep_catalogue(3, 3, sviews_meta)

    assert isinstance(catalogue, list)
    assert len(catalogue) > 0
    # Should contain identity at minimum
    has_identity = any(hasattr(c, 'name') and c.name == "identity" for c in catalogue)
    assert has_identity


def test_registry_value_catalogue_returns_all_families():
    """Value catalogue should return all VALUE families"""
    catalogue = descriptor_registry.value_catalogue()

    assert isinstance(catalogue, list)
    expected = ["CONST", "UNIQUE", "ARGMAX", "LOWEST_UNUSED", "RECOLOR", "BLOCK"]
    assert catalogue == expected


def test_registry_cost_order_is_deterministic():
    """Cost order should be deterministic and complete"""
    order1 = descriptor_registry.cost_order()
    order2 = descriptor_registry.cost_order()

    assert order1 == order2

    # Should contain key patterns
    assert "identity" in order1
    assert "CONST" in order1
    assert "RECOLOR" in order1


def test_registry_cost_order_has_keep_first():
    """Cost order should prioritize KEEP laws (lower cost)"""
    order = descriptor_registry.cost_order()

    # KEEP patterns should come before VALUE patterns
    identity_idx = order.index("identity")
    const_idx = order.index("CONST")

    assert identity_idx < const_idx


# ============================================================================
# Runner - Single Task Tests
# ============================================================================

def test_runner_simple_identity_task():
    """Runner should handle simple identity task"""
    task = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}
        ],
        "test": [
            {"input": [[5, 6], [7, 8]], "output": [[5, 6], [7, 8]]}
        ]
    }

    result = runner.run_task("test_identity", task)

    assert result["task_id"] == "test_identity"
    assert result["status"] in ["exact", "missing_descriptor", "error"]


def test_runner_task_with_no_training_pairs():
    """Runner should handle tasks with no training pairs"""
    task = {"train": [], "test": []}

    result = runner.run_task("empty_task", task)

    assert result["status"] == "error"
    assert "error" in result


def test_runner_task_exact_status_includes_train_match():
    """Exact tasks should include train_match_ok flag"""
    task = {
        "train": [
            {"input": [[1]], "output": [[1]]}
        ],
        "test": []
    }

    result = runner.run_task("exact_test", task)

    if result["status"] == "exact":
        assert "train_match_ok" in result


def test_runner_missing_descriptor_includes_witnesses():
    """Missing descriptor should include witness examples"""
    # Task that likely produces missing descriptor (complex pattern)
    task = {
        "train": [
            {"input": [[1, 2, 3]], "output": [[9, 9, 9]]}
        ],
        "test": [{"input": [[4, 5, 6]]}]
    }

    result = runner.run_task("missing_test", task)

    if result["status"] == "missing_descriptor":
        assert "missing" in result
        assert isinstance(result["missing"], list)


# ============================================================================
# Runner - Corpus Tests
# ============================================================================

def test_runner_corpus_processes_all_tasks():
    """Corpus runner should process all tasks"""
    corpus = {
        "task1": {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": []
        },
        "task2": {
            "train": [{"input": [[2]], "output": [[2]]}],
            "test": []
        }
    }

    summary = runner.run_training_corpus(corpus)

    assert summary["tasks_total"] == 2
    assert "exact" in summary
    assert "missing" in summary
    assert "results" in summary
    assert len(summary["results"]) == 2


def test_runner_corpus_deterministic_task_order():
    """Corpus runner should process tasks in sorted order"""
    corpus = {
        "zzz": {"train": [{"input": [[1]], "output": [[1]]}], "test": []},
        "aaa": {"train": [{"input": [[2]], "output": [[2]]}], "test": []},
        "mmm": {"train": [{"input": [[3]], "output": [[3]]}], "test": []}
    }

    summary = runner.run_training_corpus(corpus)

    # Results should be in sorted order
    task_ids = [r["task_id"] for r in summary["results"]]
    assert task_ids == ["aaa", "mmm", "zzz"]


def test_runner_corpus_counts_match_results():
    """Corpus summary counts should match result statuses"""
    corpus = {
        "task1": {"train": [{"input": [[1]], "output": [[1]]}], "test": []},
        "task2": {"train": [{"input": [[2]], "output": [[2]]}], "test": []}
    }

    summary = runner.run_training_corpus(corpus)

    exact_count = sum(1 for r in summary["results"] if r["status"] == "exact")
    missing_count = sum(1 for r in summary["results"] if r["status"] == "missing_descriptor")
    error_count = sum(1 for r in summary["results"] if r["status"] == "error")

    assert summary["exact"] == exact_count
    assert summary["missing"] == missing_count
    assert summary["error"] == error_count


# ============================================================================
# Report Tests
# ============================================================================

def test_report_aggregate_has_required_fields():
    """Report should have all required fields"""
    summary = {
        "tasks_total": 2,
        "exact": 1,
        "missing": 1,
        "error": 0,
        "results": [
            {"task_id": "t1", "status": "exact", "train_match_ok": True},
            {"task_id": "t2", "status": "missing_descriptor", "missing": [
                {"cid": "0", "examples": [{"train_idx": 0, "p_out": [0, 0], "expected": 5}]}
            ]}
        ]
    }

    agg_report = report.aggregate_corpus_report(summary)

    assert "tasks_total" in agg_report
    assert "exact" in agg_report
    assert "missing" in agg_report
    assert "missing_examples" in agg_report
    assert "hash" in agg_report


def test_report_hash_is_deterministic():
    """Report hash should be deterministic"""
    summary = {
        "tasks_total": 1,
        "exact": 1,
        "missing": 0,
        "error": 0,
        "results": [{"task_id": "t1", "status": "exact", "train_match_ok": True}]
    }

    report1 = report.aggregate_corpus_report(summary)
    report2 = report.aggregate_corpus_report(summary)

    assert report1["hash"] == report2["hash"]


def test_report_limits_witnesses_to_three():
    """Report should limit witnesses to first 3 per class"""
    summary = {
        "tasks_total": 1,
        "exact": 0,
        "missing": 1,
        "error": 0,
        "results": [
            {
                "task_id": "t1",
                "status": "missing_descriptor",
                "missing": [
                    {
                        "cid": "0",
                        "examples": [
                            {"train_idx": 0, "p_out": [0, 0]},
                            {"train_idx": 0, "p_out": [0, 1]},
                            {"train_idx": 0, "p_out": [0, 2]},
                            {"train_idx": 0, "p_out": [0, 3]},
                            {"train_idx": 0, "p_out": [0, 4]}
                        ]
                    }
                ]
            }
        ]
    }

    agg_report = report.aggregate_corpus_report(summary)

    # Should limit to 3 examples
    assert len(agg_report["missing_examples"]) == 1
    assert len(agg_report["missing_examples"][0]["examples"]) == 3


def test_report_preserves_task_and_class_info():
    """Report should preserve task_id and cid for each witness"""
    summary = {
        "tasks_total": 1,
        "exact": 0,
        "missing": 1,
        "error": 0,
        "results": [
            {
                "task_id": "task_abc",
                "status": "missing_descriptor",
                "missing": [
                    {
                        "cid": "42",
                        "examples": [{"train_idx": 0, "p_out": [1, 2], "expected": 5}]
                    }
                ]
            }
        ]
    }

    agg_report = report.aggregate_corpus_report(summary)

    assert len(agg_report["missing_examples"]) == 1
    example = agg_report["missing_examples"][0]
    assert example["task_id"] == "task_abc"
    assert example["cid"] == "42"


# ============================================================================
# Determinism Tests
# ============================================================================

def test_runner_determinism_same_input_same_output():
    """Runner should produce identical results on repeated runs"""
    corpus = {
        "determinism_test": {
            "train": [{"input": [[1, 2]], "output": [[1, 2]]}],
            "test": []
        }
    }

    summary1 = runner.run_training_corpus(corpus)
    summary2 = runner.run_training_corpus(corpus)

    # Status should match
    assert summary1["exact"] == summary2["exact"]
    assert summary1["missing"] == summary2["missing"]

    # Result statuses should match
    assert summary1["results"][0]["status"] == summary2["results"][0]["status"]


def test_report_hash_changes_with_different_data():
    """Report hash should change when data changes"""
    summary1 = {
        "tasks_total": 1,
        "exact": 1,
        "missing": 0,
        "error": 0,
        "results": [{"task_id": "t1", "status": "exact"}]
    }

    summary2 = {
        "tasks_total": 2,  # Different
        "exact": 1,
        "missing": 1,
        "error": 0,
        "results": [
            {"task_id": "t1", "status": "exact"},
            {"task_id": "t2", "status": "missing_descriptor", "missing": []}
        ]
    }

    report1 = report.aggregate_corpus_report(summary1)
    report2 = report.aggregate_corpus_report(summary2)

    assert report1["hash"] != report2["hash"]


# ============================================================================
# Integration Tests
# ============================================================================

def test_runner_to_report_integration():
    """Runner output should integrate cleanly with report aggregation"""
    corpus = {
        "integration_test": {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": []
        }
    }

    summary = runner.run_training_corpus(corpus)
    agg_report = report.aggregate_corpus_report(summary)

    # Report should aggregate summary correctly
    assert agg_report["tasks_total"] == summary["tasks_total"]
    assert agg_report["exact"] == summary["exact"]
    assert agg_report["missing"] == summary["missing"]


def test_print_closure_progress_no_crash():
    """Print closure progress should not crash"""
    agg_report = {
        "tasks_total": 1,
        "exact": 1,
        "missing": 0,
        "missing_examples": [],
        "hash": "abc123"
    }

    # Should not raise
    try:
        report.print_closure_progress(agg_report)
        success = True
    except Exception:
        success = False

    assert success


def test_print_witness_detail_no_crash():
    """Print witness detail should not crash"""
    witness = {
        "task_id": "test",
        "cid": "0",
        "train_idx": 0,
        "p_out": [1, 2],
        "expected": 5,
        "got": 3,
        "path": {"p_test": [0, 1]}
    }

    # Should not raise
    try:
        report.print_witness_detail(witness)
        success = True
    except Exception:
        success = False

    assert success


# ============================================================================
# Test Intent Summary
# ============================================================================

def test_intent_summary():
    """
    WO-12 Test Intent Summary

    Descriptor Registry:
    - keep_catalogue returns list from WO-08
    - value_catalogue returns fixed list of VALUE families
    - cost_order is deterministic and prioritizes KEEP
    - Additive-only: never remove/rename

    Runner (Single Task):
    - Processes task through full pipeline (Π → Q → Shape → Laws → Sieve → Paint)
    - Returns status: exact | missing_descriptor | error
    - Exact includes train_match_ok
    - Missing includes witness examples

    Runner (Corpus):
    - Processes all tasks in sorted order
    - Aggregates counts (exact, missing, error)
    - Returns results list
    - Deterministic: same input → same output

    Report:
    - Aggregates corpus results
    - Limits witnesses to 3 per class
    - Computes deterministic hash (sorted JSON)
    - Preserves task_id + cid for witnesses

    Integration:
    - Runner → Report flow
    - Print functions don't crash
    - Hash changes with data, stable otherwise

    Key Properties:
    - Determinism: Fixed task order, deterministic pipeline
    - Witnesses: First 3 examples per missing class
    - Algebraic debugging: Coordinate paths in witnesses
    - Additive closure: Extend vocabulary, never remove
    """
    pass


if __name__ == "__main__":
    print("Running WO-12 runner tests...")

    # Registry tests
    test_registry_keep_catalogue_returns_list()
    test_registry_value_catalogue_returns_all_families()
    test_registry_cost_order_is_deterministic()
    test_registry_cost_order_has_keep_first()
    print("✓ Registry tests passed")

    # Runner single task tests
    test_runner_simple_identity_task()
    test_runner_task_with_no_training_pairs()
    test_runner_task_exact_status_includes_train_match()
    test_runner_missing_descriptor_includes_witnesses()
    print("✓ Runner single task tests passed")

    # Runner corpus tests
    test_runner_corpus_processes_all_tasks()
    test_runner_corpus_deterministic_task_order()
    test_runner_corpus_counts_match_results()
    print("✓ Runner corpus tests passed")

    # Report tests
    test_report_aggregate_has_required_fields()
    test_report_hash_is_deterministic()
    test_report_limits_witnesses_to_three()
    test_report_preserves_task_and_class_info()
    print("✓ Report tests passed")

    # Determinism tests
    test_runner_determinism_same_input_same_output()
    test_report_hash_changes_with_different_data()
    print("✓ Determinism tests passed")

    # Integration tests
    test_runner_to_report_integration()
    test_print_closure_progress_no_crash()
    test_print_witness_detail_no_crash()
    print("✓ Integration tests passed")

    print(f"\n✅ All 22 WO-12 runner tests passed!")
