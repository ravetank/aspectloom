import contextlib
import io
import unittest

from PIL import Image

import aspectloom as app


class BucketSelectionTests(unittest.TestCase):
    def setUp(self):
        self.original_strategy = app.BUCKET_STRATEGY

    def tearDown(self):
        app.BUCKET_STRATEGY = self.original_strategy

    def test_exact_landscape_bucket_is_unchanged(self):
        self.assertEqual(
            app.get_nearest_bucket(1920, 1080),
            (1920, 1080, "exact", 0.0),
        )

    def test_exact_portrait_bucket_is_unchanged(self):
        self.assertEqual(
            app.get_nearest_bucket(1080, 1920),
            (1080, 1920, "exact", 0.0),
        )

    def test_aspect_strategy_returns_a_containing_bucket(self):
        app.BUCKET_STRATEGY = "aspect"
        width, height, action, waste = app.get_nearest_bucket(1500, 1000)
        self.assertGreaterEqual(width, 1500)
        self.assertGreaterEqual(height, 1000)
        self.assertEqual(action, "pad")
        self.assertGreaterEqual(waste, 0.0)
        self.assertLess(waste, 1.0)

    def test_first_fit_strategy_returns_a_containing_bucket(self):
        app.BUCKET_STRATEGY = "first_fit"
        width, height, action, _ = app.get_nearest_bucket(1500, 1000)
        self.assertGreaterEqual(width, 1500)
        self.assertGreaterEqual(height, 1000)
        self.assertEqual(action, "pad")

    def test_oversized_image_is_downscaled(self):
        _, _, action, _ = app.get_nearest_bucket(20_000, 10_000)
        self.assertEqual(action, "downscale+pad")


class ImageProcessingTests(unittest.TestCase):
    def setUp(self):
        self.original_padding_mode = app.PADDING_MODE

    def tearDown(self):
        app.PADDING_MODE = self.original_padding_mode

    def test_canvas_has_target_size_and_alpha_mask(self):
        app.PADDING_MODE = "transparent"
        source = Image.new("RGB", (100, 50), "red")
        canvas, geometry = app.build_canvas(source, 200, 100)

        self.assertEqual(canvas.mode, "RGBA")
        self.assertEqual(canvas.size, (200, 100))
        self.assertEqual(geometry, (50, 25, 100, 50))
        self.assertEqual(canvas.getchannel("A").getpixel((0, 0)), 0)
        self.assertGreater(canvas.getchannel("A").getpixel((100, 50)), 0)

    def test_composite_preserves_the_source_center(self):
        source = Image.new("RGB", (400, 300), "red")
        generated = Image.new("RGB", (800, 600), "blue")
        result = app.composite_result(generated, source, 800, 600)

        center = result.getpixel((400, 300))
        edge = result.getpixel((0, 0))
        self.assertGreater(center[0], center[2])
        self.assertGreater(edge[2], edge[0])

    def test_inpaint_dimensions_are_multiples_of_eight(self):
        width, height = app.compute_inpaint_size(4032, 3024, 0.1)
        self.assertEqual(width % 8, 0)
        self.assertEqual(height % 8, 0)
        self.assertLessEqual(max(width, height), app.MAX_INPAINT_LONG_EDGE)


class WorkflowTests(unittest.TestCase):
    def test_workflow_uses_selected_model_and_image(self):
        workflow = app.build_workflow(
            "uploaded.png",
            "Aspectloom/example",
            model_name="custom-inpaint.safetensors",
        )

        self.assertEqual(
            workflow["2"]["inputs"]["ckpt_name"],
            "custom-inpaint.safetensors",
        )
        self.assertEqual(workflow["8"]["inputs"]["image"], "uploaded.png")
        self.assertEqual(
            workflow["10"]["inputs"]["filename_prefix"],
            "Aspectloom/example",
        )


class DisplayTests(unittest.TestCase):
    def test_bucket_table_is_ascii_safe(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            app.print_bucket_table()

        output.getvalue().encode("ascii")


if __name__ == "__main__":
    unittest.main()
