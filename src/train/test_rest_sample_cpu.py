"""CPU 单测：ReST 采样器的纯逻辑（不加载 torch 权重、不占 GPU）。

覆盖：过滤判据、IoU 分桶、池记录字段、数据集归属、断点续跑读取。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from .rest_sample import (
    accepted,
    build_audit_record,
    build_pool_record,
    dataset_of,
    iou_bucket,
    load_done_ids,
    select_records,
)


def fake_score(*, iou: float | None, box: list[float] | None, parse_error: str | None = None):
    if box is None:
        iou = None
    return SimpleNamespace(iou=iou, predicted_box=box, parse_error=parse_error, correct_at_0_5=bool(iou and iou >= 0.5))


RECORD = {
    "id": "synthetic-000123",
    "image": "data/synthetic_images/000123.png",
    "question": "Point to the red cube.",
    "answer": [0.1, 0.2, 0.3, 0.4],
    "width": 320,
    "height": 256,
    "source": "synthetic",
}


class TestAcceptance(unittest.TestCase):
    def test_accepts_exactly_at_tau(self):
        score = fake_score(iou=0.5, box=[0.1, 0.1, 0.2, 0.2])
        self.assertTrue(accepted(score, tau=0.5))

    def test_rejects_below_tau(self):
        score = fake_score(iou=0.4999, box=[0.1, 0.1, 0.2, 0.2])
        self.assertFalse(accepted(score, tau=0.5))

    def test_rejects_parse_failure_even_with_high_iou(self):
        score = fake_score(iou=0.9, box=[0.1, 0.1, 0.2, 0.2], parse_error="no_box")
        self.assertFalse(accepted(score, tau=0.5))

    def test_rejects_degenerate_box(self):
        score = fake_score(iou=0.9, box=[0.5, 0.5, 0.5, 0.6])
        self.assertFalse(accepted(score, tau=0.5))

    def test_rejects_missing_box(self):
        score = fake_score(iou=None, box=None)
        self.assertFalse(accepted(score, tau=0.5))


class TestBuckets(unittest.TestCase):
    def test_bucket_edges(self):
        self.assertEqual(iou_bucket(None), "parse_fail")
        self.assertEqual(iou_bucket(0.0), "lt_0.25")
        self.assertEqual(iou_bucket(0.25), "0.25_to_0.5")
        self.assertEqual(iou_bucket(0.5), "0.5_to_0.75")
        self.assertEqual(iou_bucket(0.75), "ge_0.75")


class TestDatasetAttribution(unittest.TestCase):
    def test_explicit_field_wins(self):
        self.assertEqual(dataset_of({"dataset": "refcoco", "image": "x"}), "refcoco")

    def test_infers_from_image_path(self):
        self.assertEqual(dataset_of({"image": "data/synthetic_images/a.png"}), "synthetic")
        self.assertEqual(dataset_of({"image": "data/refcoco_images/a.jpg"}), "refcoco")

    def test_unknown(self):
        self.assertEqual(dataset_of({"image": "a.png"}), "unknown")


class TestPoolRecord(unittest.TestCase):
    def test_pool_record_is_trainer_ready(self):
        line = build_pool_record(RECORD, k_index=2, predicted_box=[0.5, 0.5, 0.6, 0.6], iou=0.8)
        self.assertEqual(
            set(line),
            {"id", "image", "question", "answer", "raw_bbox", "width", "height", "source"},
        )
        self.assertEqual(line["id"], "synthetic-000123#k2")
        self.assertEqual(line["image"], RECORD["image"])
        self.assertEqual(line["answer"], [0.5, 0.5, 0.6, 0.6])
        self.assertTrue(all(isinstance(value, float) for value in line["answer"]))
        # train_sft.py 视觉 token 预检与 split 校验依赖这三个字段
        self.assertEqual(line["width"], 320)
        self.assertEqual(line["height"], 256)
        self.assertEqual(line["source"], "synthetic")
        # raw_bbox 必须由预测框推导且与 answer 自洽（dataset.validate_record 的 1.1e-8 校验）
        self.assertEqual(line["raw_bbox"], [0.5 * 320, 0.5 * 256, 0.6 * 320, 0.6 * 256])
        for got, norm, size in zip(line["raw_bbox"], line["answer"], (320, 256, 320, 256)):
            self.assertAlmostEqual(got / size, norm, places=12)
        json.dumps(line)

    def test_missing_trainer_field_raises(self):
        broken = {key: value for key, value in RECORD.items() if key != "width"}
        with self.assertRaises(ValueError):
            build_pool_record(broken, k_index=0, predicted_box=[0.1, 0.1, 0.2, 0.2], iou=0.6)

    def test_pool_answer_does_not_leak_ground_truth(self):
        """ReST 训练目标是模型自己的预测，不是 GT；混入 GT 会污染自训练。"""

        line = build_pool_record(RECORD, k_index=0, predicted_box=[0.9, 0.9, 0.95, 0.95], iou=0.55)
        self.assertNotEqual(line["answer"], RECORD["answer"])


class TestAuditRecord(unittest.TestCase):
    def test_audit_keeps_ground_truth_and_iou(self):
        score = fake_score(iou=0.8, box=[0.1, 0.1, 0.3, 0.4])
        line = build_audit_record(
            RECORD,
            k_index=1,
            accepted=True,
            output_text="[0.1, 0.1, 0.3, 0.4]",
            raw_output=" [0.1, 0.1, 0.3, 0.4]<end_of_utterance>",
            score=score,
            tau=0.5,
        )
        self.assertTrue(line["accepted"])
        self.assertEqual(line["source_id"], RECORD["id"])
        self.assertEqual(line["ground_truth"], RECORD["answer"])
        self.assertEqual(line["iou_bucket"], "ge_0.75")
        self.assertEqual(line["dataset"], "synthetic")


class TestSelection(unittest.TestCase):
    def _records(self):
        synth = [
            {"id": f"s{i}", "image": "data/synthetic_images/a.png", "question": "q", "answer": [0, 0, 1, 1]}
            for i in range(5)
        ]
        ref = [
            {"id": f"r{i}", "image": "data/refcoco_images/a.jpg", "question": "q", "answer": [0, 0, 1, 1]}
            for i in range(3)
        ]
        return synth + ref

    def test_head_limit_keeps_legacy_behaviour(self):
        picked = select_records(self._records(), limit=2, per_dataset=None, datasets=None)
        self.assertEqual([item["id"] for item in picked], ["s0", "s1"])

    def test_per_dataset_interleaves_and_balances(self):
        picked = select_records(self._records(), limit=None, per_dataset=2, datasets=None)
        self.assertEqual([item["id"] for item in picked], ["r0", "s0", "r1", "s1"])

    def test_per_dataset_respects_small_group(self):
        picked = select_records(self._records(), limit=None, per_dataset=10, datasets=None)
        self.assertEqual(len(picked), 8)

    def test_dataset_filter(self):
        picked = select_records(self._records(), limit=None, per_dataset=None, datasets=["refcoco"])
        self.assertEqual([item["id"] for item in picked], ["r0", "r1", "r2"])

    def test_empty_selection_raises(self):
        with self.assertRaises(SystemExit):
            select_records(self._records(), limit=None, per_dataset=None, datasets=["screenspot"])


class TestResume(unittest.TestCase):
    def test_reads_only_done_rows(self):
        # 用工作区内临时目录：系统 %TEMP% 在沙箱下清理会被拒（chmod PermissionError）。
        tmp_root = Path(__file__).resolve().parents[2] / "logs" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        try:
            path = tmp_root / "rest_sample_progress_test.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"source_id": "a", "status": "done"}),
                        json.dumps({"source_id": "b", "status": "started"}),
                        "not json at all",
                        json.dumps({"source_id": "c", "status": "done"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_done_ids(path), {"a", "c"})
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def test_missing_progress_file(self):
        self.assertEqual(load_done_ids(Path("does-not-exist.jsonl")), set())


if __name__ == "__main__":
    unittest.main()
