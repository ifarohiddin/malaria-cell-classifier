import io
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


class FakeModel:
    """Haqiqiy modelni yuklamaslik uchun: har doim bir xil ehtimollik qaytaradi."""

    def __init__(self):
        self.probability = 0.9
        self.calls = 0

    def predict(self, batch, verbose=0):
        self.calls += 1
        assert batch.shape == (1, 128, 128, 3)
        return np.array([[self.probability]])


# `main` import paytida modelni yuklaydi, shuning uchun yuklashni almashtirib turib import qilamiz
with patch("tensorflow.keras.models.load_model", return_value=FakeModel()):
    import main


def png_bytes(mode="RGB", size=(60, 40)):
    buffer = io.BytesIO()
    Image.new(mode, size, color=128 if mode == "L" else (200, 30, 30) + ((255,) if mode == "RGBA" else ())).save(buffer, "PNG")
    return buffer.getvalue()


class PrepareImageTests(unittest.TestCase):
    class Upload:
        def __init__(self, data):
            self.file = io.BytesIO(data)

    def test_shape_and_range(self):
        array = main.prepare_image(self.Upload(png_bytes()))
        self.assertEqual(array.shape, (1, 128, 128, 3))
        self.assertTrue(0.0 <= array.min() and array.max() <= 1.0)

    def test_grayscale_and_rgba_are_converted_to_rgb(self):
        for mode in ("L", "RGBA"):
            with self.subTest(mode=mode):
                self.assertEqual(main.prepare_image(self.Upload(png_bytes(mode))).shape, (1, 128, 128, 3))


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.fake = FakeModel()
        patcher = patch.object(main, "model", self.fake)
        patcher.start()
        self.addCleanup(patcher.stop)

    def predict(self, probability, data=None, name="cell.png"):
        self.fake.probability = probability
        return self.client.post("/predict", files={"file": (name, data or png_bytes(), "image/png")})

    def test_home_page_has_upload_form(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('action="/predict"', response.text)

    def test_parasitic_prediction(self):
        response = self.predict(0.9)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Parasitic", response.text)
        self.assertIn("90.00%", response.text)
        self.assertIn("cell.png", response.text)

    def test_uninfected_prediction(self):
        response = self.predict(0.2)
        self.assertIn("Uninfected", response.text)
        self.assertIn("80.00%", response.text)

    def test_threshold_is_exclusive(self):
        response = self.predict(0.5)
        self.assertIn("Uninfected", response.text)
        self.assertIn("50.00%", response.text)

    def test_non_image_upload_returns_400_without_running_the_model(self):
        response = self.predict(0.9, data=b"bu rasm emas", name="notes.txt")
        self.assertEqual(response.status_code, 400)
        self.assertIn("rasm emas", response.text)
        self.assertEqual(self.fake.calls, 0)

    def test_missing_file_is_rejected(self):
        self.assertEqual(self.client.post("/predict").status_code, 422)


if __name__ == "__main__":
    unittest.main()
