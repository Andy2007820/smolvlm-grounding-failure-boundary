"""CPU-only tests for the T15 demo core and drawing helpers.

``gradio`` is not imported anywhere in these tests: the UI wiring is exercised
through :class:`demo.app.GroundingDemo` with a fake predictor, which is exactly
the part that must be correct before a real model is attached.
"""

from __future__ import annotations

import os
import sys
import unittest

# Hide the GPU before any import that could touch torch (none should).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from PIL import Image  # noqa: E402

from demo.app import FakePredictor, GroundingDemo, build_parser, load_examples  # noqa: E402
from demo.draw import box_to_pixels, draw_box, format_result  # noqa: E402


def blank_image(width: int = 200, height: int = 120) -> Image.Image:
    return Image.new("RGB", (width, height), (255, 255, 255))


class DrawBoxTests(unittest.TestCase):
    def test_box_to_pixels_rounds_inside_image(self):
        self.assertEqual(box_to_pixels([0.1, 0.2, 0.9, 0.8], 200, 120), (20, 24, 180, 96))

    def test_box_to_pixels_rejects_invalid_box(self):
        with self.assertRaises(ValueError):
            box_to_pixels([0.9, 0.2, 0.1, 0.8], 200, 100)
        with self.assertRaises(ValueError):
            box_to_pixels([0.1, 0.2, 0.3], 200, 100)
        with self.assertRaises(ValueError):
            box_to_pixels([1.1, 0.2, 0.9, 0.8], 200, 100)

    def test_draw_box_paints_the_border_and_preserves_the_inside(self):
        image = blank_image()
        annotated = draw_box(image, [0.25, 0.25, 0.75, 0.75], width=3)
        self.assertEqual(annotated.size, image.size)
        # Border pixel (top-left corner of the box) must be colored.
        self.assertNotEqual(annotated.getpixel((50, 30)), (255, 255, 255))
        # Interior of the box must stay untouched (rectangle outline only).
        self.assertEqual(annotated.getpixel((100, 60)), (255, 255, 255))
        # The original image must not be mutated.
        self.assertEqual(image.getpixel((50, 30)), (255, 255, 255))

    def test_draw_box_with_none_returns_copy(self):
        image = blank_image()
        annotated = draw_box(image, None)
        self.assertIsNot(annotated, image)
        self.assertEqual(annotated.getpixel((0, 0)), image.getpixel((0, 0)))

    def test_draw_box_rejects_non_image(self):
        with self.assertRaises(ValueError):
            draw_box(None, [0.1, 0.1, 0.2, 0.2])

    def test_format_result_marks_parse_failure(self):
        text = format_result({"predicted_box": None, "parse_error": "expected_exactly_four_coordinates"})
        self.assertIn("expected_exactly_four_coordinates", text)

    def test_format_result_renders_box_and_raw_output(self):
        text = format_result(
            {
                "predicted_box": [0.1, 0.2, 0.3, 0.4],
                "output_text": "[0.1, 0.2, 0.3, 0.4]",
                "latency_seconds": 1.25,
            },
            pixel_box=(10, 20, 30, 40),
        )
        self.assertIn("[0.1000, 0.2000, 0.3000, 0.4000]", text)
        self.assertIn("(10, 20, 30, 40)", text)
        self.assertIn("1.250 s", text)


class DemoHandlerTests(unittest.TestCase):
    def make_demo(self) -> GroundingDemo:
        return GroundingDemo(
            FakePredictor([0.2, 0.3, 0.6, 0.7]),
            engine_label="fake",
            artifact_label="fake placeholder",
            is_fake=True,
        )

    def test_locate_returns_annotated_image_and_text(self):
        demo = self.make_demo()
        image = blank_image()
        annotated, text = demo.locate(image, "the red cup")
        self.assertIsInstance(annotated, Image.Image)
        self.assertEqual(annotated.size, image.size)
        self.assertIn("[0.2000, 0.3000, 0.6000, 0.7000]", text)
        self.assertIn("占位预测器", text)
        # Border at x=40,y=36 for a 200x120 image with box [0.2,0.3,0.6,0.7].
        self.assertNotEqual(annotated.getpixel((40, 36)), (255, 255, 255))

    def test_locate_handles_missing_image(self):
        demo = self.make_demo()
        annotated, text = demo.locate(None, "the red cup")
        self.assertIsNone(annotated)
        self.assertIn("上传", text)

    def test_locate_handles_empty_question(self):
        demo = self.make_demo()
        image = blank_image()
        annotated, text = demo.locate(image, "   ")
        self.assertIs(annotated, image)
        self.assertIn("指代表达", text)

    def test_locate_survives_predictor_exception(self):
        class BrokenPredictor:
            def predict(self, image, question):
                raise RuntimeError("boom")

        demo = GroundingDemo(BrokenPredictor(), engine_label="broken", artifact_label="broken")
        annotated, text = demo.locate(blank_image(), "the cup")
        self.assertIsInstance(annotated, Image.Image)
        self.assertIn("推理失败", text)

    def test_description_states_the_frozen_contract(self):
        text = self.make_demo().description()
        self.assertIn("p1", text)
        self.assertIn("image_seq_len `36`", text)
        self.assertIn("parse_box_with_error", text)

    def test_fake_predictor_is_labeled_as_placeholder(self):
        report = FakePredictor().load()
        self.assertTrue(report["is_fake"])
        self.assertIn("not a model", report["note"])


class DemoCliTests(unittest.TestCase):
    def test_parser_defaults_are_safe(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.engine, "hf-cpu")
        self.assertEqual(args.device, "cpu")
        self.assertFalse(args.share)
        self.assertFalse(args.smoke)

    def test_examples_are_real_images_from_the_dataset(self):
        examples = load_examples(limit=2)
        self.assertLessEqual(len(examples), 2)
        for image_path, question in examples:
            self.assertTrue(os.path.isfile(image_path), image_path)
            self.assertTrue(question.strip())

    def test_app_module_does_not_import_gradio(self):
        self.assertNotIn("gradio", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
