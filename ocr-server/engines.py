from abc import ABC, abstractmethod
from math import pi
from typing import TypedDict
import asyncio
import os
import threading


import chrome_lens_py
from chrome_lens_py.utils.lens_betterproto import LensOverlayObjectsResponse
from PIL.Image import Image


class BoundingBox(TypedDict):
    x: float
    y: float
    width: float
    height: float


class Bubble(TypedDict):
    text: str
    tightBoundingBox: BoundingBox
    orientation: float
    font_size: float
    confidence: float


class Engine(ABC):
    """Base class for OCR engines. Each engine should implement the `ocr` method
    which processes an image and returns a list of Bubble objects.
    """

    @abstractmethod
    async def ocr(self, img: Image) -> list[Bubble]:
        pass


class OneOCR(Engine):
    def __init__(self):
        try:
            import oneocr

            self.engine = oneocr.OcrEngine()
        except ImportError as e:
            print(f"[Warning] OneOCR import failed: {e}")
        except Exception as e:
            print(
                f"[Warning] If you get this error please spam the Mangatan thread: {e}"
            )

        # The height of each chunk to process.
        # A value between 1000-2000 is a good starting point.
        self.CHUNK_HEIGHT = 1500
        # The pixel overlap between chunks to prevent cutting text in half.
        self.OVERLAP = 150

    async def ocr(self, img):
        chunk_image = self.process_image(img)
        # print(json.dumps(chunk_image, indent=2, ensure_ascii=False))
        return chunk_image

    def process_image(self, img: Image) -> list[Bubble]:
        full_width, full_height = img.size
        y_offset = 0
        all_transformed_results: list[Bubble] = []

        while y_offset < full_height:
            # Define the crop box for the current chunk
            box = (
                0,
                y_offset,
                full_width,
                min(y_offset + self.CHUNK_HEIGHT, full_height),
            )

            # Crop the image to get the current chunk
            chunk_image = img.crop(box)
            chunk_width, chunk_height = chunk_image.size

            # Run OCR on the smaller chunk
            results = self.engine.recognize_pil(chunk_image)
            data = self.transform(results, chunk_image.size)

            # Remap the coordinates of the detected text to be relative to the FULL image
            for item in data:
                bbox = item["tightBoundingBox"]
                # Adjust y and height based on the chunk's position and size
                bbox["y"] = (bbox["y"] * chunk_height + y_offset) / full_height
                bbox["height"] = (bbox["height"] * chunk_height) / full_height
                all_transformed_results.append(item)

            # Move to the next chunk position
            y_offset += self.CHUNK_HEIGHT - self.OVERLAP
        return all_transformed_results

    def transform(self, result, image_size) -> list[Bubble]:
        if not result or not result.get("lines"):
            return []

        image_width, image_height = image_size
        if image_width == 0 or image_height == 0:
            return []

        output_json = []

        for line in result.get("lines", []):
            text = line.get("text", "").strip()
            rect = line.get("bounding_rect")

            if not rect or not text or not line.get("words"):
                continue

            x_coords = [rect["x1"], rect["x2"], rect["x3"], rect["x4"]]
            y_coords = [rect["y1"], rect["y2"], rect["y3"], rect["y4"]]
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)
            width = x_max - x_min
            height = y_max - y_min
            snapped_angle = 90.0 if height > width else 0.0
            word_count = len(line.get("words", []))
            avg_confidence = (
                sum(word.get("confidence", 0.95) for word in line.get("words", []))
                / word_count
                if word_count > 0
                else 0.95
            )

            bubble = Bubble(
                text=text,
                tightBoundingBox=BoundingBox(
                    x=x_min / image_width,
                    y=y_min / image_height,
                    width=width / image_width,
                    height=height / image_height,
                ),
                orientation=snapped_angle,
                font_size=0.04,
                confidence=avg_confidence,
            )
            output_json.append(bubble)

        return output_json


class GoogleLens(Engine):
    def __init__(self):
        self.engine = chrome_lens_py.LensAPI()

    async def ocr(self, img):
        result = await self.engine.process_image(
            image_path=img, ocr_language="ja", output_format="lines"
        )
        return self.transform(result)

    def transform(self, result: dict) -> list[Bubble]:
        if not result.get("word_data"):
            return []

        output_json: list[Bubble] = []
        lines: list[dict] = result["line_blocks"]

        for line in lines:
            text: str = line["text"]
            geometry: dict[str, float] = line["geometry"]
            center_x = geometry["center_x"]
            center_y = geometry["center_y"]
            width = geometry["width"]
            height = geometry["height"]

            # example: 6.5; degrees from perfect vertical or horizontal line
            angle_deg = geometry["angle_deg"]

            # 90.0 is a vertical line, 0.0 is horizontal
            snapped_angle = 90.0 if height > width else 0.0

            # example: 90.0 + 6.5 = 96.5, or rotated 6.5 degrees clockwise from vertical
            actual_angle = snapped_angle + angle_deg

            bubble = Bubble(
                text=text.replace("･･･", "…"),
                tightBoundingBox=BoundingBox(
                    x=center_x - width / 2,
                    y=center_y - height / 2,
                    width=width,
                    height=height,
                ),
                orientation=round(actual_angle, 1),
                font_size=0.04,
                confidence=0.98,  # Assuming a default confidence value
            )
            output_json.append(bubble)
            # print(json.dumps(bubble, indent=2, ensure_ascii=False))

        return output_json

    # just in case we want to parse it ourselves
    def raw_transform(self, result: dict) -> list[Bubble]:
        output_json: list[Bubble] = []
        response: LensOverlayObjectsResponse = result["raw_response_objects"]

        for paragraph in response.text.text_layout.paragraphs:
            for line in paragraph.lines:
                line_text = (
                    "".join(
                        word.plain_text + (word.text_separator or "")
                        for word in line.words
                    )
                    .strip()
                    .replace("･･･", "…")
                )
                geometry = line.geometry

                bounding_box = geometry.bounding_box
                center_x = bounding_box.center_x
                center_y = bounding_box.center_y
                width = bounding_box.width
                height = bounding_box.height
                rotation_z = bounding_box.rotation_z

                bubble = Bubble(
                    text=line_text,
                    tightBoundingBox=BoundingBox(
                        x=center_x - width / 2,
                        y=center_y - height / 2,
                        width=width,
                        height=height,
                    ),
                    orientation=round(rotation_z * (180 / pi), 1),
                    font_size=0.04,
                    confidence=0.98,
                )
                output_json.append(bubble)

        return output_json


# TODO: get a mac
class AppleVision(Engine):
    def __init__(self):
        print("AppleVision is not implemented yet")
        self.engine = object()

    async def ocr(self, img: Image) -> list[Bubble]:
        print("AppleVision is not implemented yet")
        return []


class MangaOCR(Engine):
    """Local, fully-offline detect+recognize pipeline via mokuro
    (comic-text-detector for bubble/line detection + kha-white/manga-ocr for recognition).

    manga-ocr is recognition-only (crop -> string, no boxes), so we use mokuro's bundled
    comic-text-detector to find text blocks/lines, then recognize each line. We construct
    `mokuro.MangaPageOcr` for its `.text_detector`, `.mocr` and `.split_into_chunks`, but we
    do NOT call `MangaPageOcr.__call__` (it reads a FILE PATH); instead we run its pipeline
    against the in-memory PIL page. One Bubble is emitted per detected line, which matches
    what server.py's auto_merge_ocr_data() expects (it groups lines into bubbles).

    Heavy deps (torch/cv2/numpy/mokuro) are imported lazily so this module still imports on a
    lens-only image. Set MANGAOCR_FORCE_CPU=1 to skip CUDA.
    """

    # Serializes GPU/model access across worker threads — manga-ocr + comic-text-detector
    # share one CUDA context with bounded VRAM. MUST be a *threading* primitive, not
    # asyncio.Semaphore: Flask[async] under waitress runs each request in its own event loop,
    # so an asyncio semaphore created at import binds to the first loop and raises
    # "Future attached to a different loop" on every later request.
    _gpu_lock = threading.Semaphore(1)

    def __init__(self, force_cpu: bool = False):
        force_cpu = force_cpu or os.environ.get("MANGAOCR_FORCE_CPU", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        # Lazy import: only pull the torch/mokuro stack when this engine is actually selected.
        from mokuro import MangaPageOcr

        self._MangaPageOcr = MangaPageOcr
        # First construction downloads (then caches) comictextdetector.pt (~76 MB) and
        # kha-white/manga-ocr-base (~444 MB). Auto-selects CUDA when available unless force_cpu.
        self.mpocr = MangaPageOcr(
            pretrained_model_name_or_path="kha-white/manga-ocr-base",
            force_cpu=force_cpu,
            detector_input_size=1024,
            text_height=64,
            max_ratio_vert=16,
            max_ratio_hor=8,
            anchor_window=2,
            disable_ocr=False,
        )
        print("[Engine] MangaOCR ready (mokuro: comic-text-detector + manga-ocr, "
              f"force_cpu={force_cpu})")

    async def ocr(self, img: Image) -> list[Bubble]:
        # Blocking torch inference -> run off the event loop (GPU access serialized in _ocr_sync).
        return await asyncio.to_thread(self._ocr_sync, img)

    def _ocr_sync(self, img: Image) -> list[Bubble]:
        import cv2
        import numpy as np
        from PIL import Image as PILImageModule

        W, H = img.size
        if not W or not H:
            return []

        # comic-text-detector expects a BGR ndarray; PIL is RGB.
        rgb = np.asarray(img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        bubbles: list[Bubble] = []
        # Hold the GPU for the whole page: one CUDA context, one page at a time.
        with self._gpu_lock:
            # refine_mode=1 == REFINEMASK_ANNOTATION; returns (mask, mask_refined, blk_list).
            _, mask_refined, blk_list = self.mpocr.text_detector(
                bgr, refine_mode=1, keep_undetected_mask=True
            )

            for blk in blk_list:
                try:
                    vertical = bool(getattr(blk, "vertical", False))
                    block_orientation = 90.0 if vertical else 0.0
                    font_px = float(getattr(blk, "font_size", 0) or 0)
                    if font_px < 0:  # detector returns -1 when it can't estimate
                        font_px = 0
                    max_ratio = (
                        self.mpocr.max_ratio_vert if vertical else self.mpocr.max_ratio_hor
                    )

                    lines = blk.lines_array()  # per-line quad polygons (4 pts, pixel coords)
                    for line_idx, line in enumerate(lines):
                        line_crops, _ = self._MangaPageOcr.split_into_chunks(
                            bgr,
                            mask_refined,
                            blk,
                            line_idx,
                            textheight=self.mpocr.text_height,
                            max_ratio=max_ratio,
                            anchor_window=self.mpocr.anchor_window,
                        )
                        text = ""
                        for crop in line_crops:
                            if vertical:
                                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            text += self.mpocr.mocr(PILImageModule.fromarray(crop_rgb))

                        text = text.strip()
                        if not text:
                            continue

                        pts = np.asarray(line, dtype=np.float64).reshape(-1, 2)
                        x_min, y_min = float(pts[:, 0].min()), float(pts[:, 1].min())
                        x_max, y_max = float(pts[:, 0].max()), float(pts[:, 1].max())
                        bw, bh = (x_max - x_min), (y_max - y_min)
                        if bw <= 0 or bh <= 0:
                            continue

                        bubbles.append(
                            Bubble(
                                text=text,
                                tightBoundingBox=BoundingBox(
                                    x=x_min / W,
                                    y=y_min / H,
                                    width=bw / W,
                                    height=bh / H,
                                ),
                                # per-line aspect overrides the block flag when unambiguous
                                orientation=90.0 if bh > bw else block_orientation,
                                font_size=(font_px / H) if font_px > 0 else 0.04,
                                confidence=0.95,  # detector/recognizer expose no per-line score
                            )
                        )
                except Exception as e:
                    # One bad block must not drop the whole page; keep what we have.
                    print(f"[Engine] MangaOCR skipped a block: {e}")
                    continue

        return bubbles


class FallbackEngine(Engine):
    """Runs engines in order; uses the next one when an engine raises OR returns no text.

    Used for the default 'mangaocr+lens' chain: try the local manga-ocr pipeline first, and
    fall back to Google Lens if it errors or finds nothing on a page.
    """

    def __init__(self, engines: list[Engine]):
        if not engines:
            raise ValueError("FallbackEngine requires at least one engine")
        self.engines = engines

    async def ocr(self, img: Image) -> list[Bubble]:
        last: list[Bubble] = []
        for i, engine in enumerate(self.engines):
            name = type(engine).__name__
            try:
                result = await engine.ocr(img)
            except Exception as e:
                print(f"[Engine] {name} raised ({e}); falling back to next engine")
                continue
            if result:
                if i > 0:
                    print(f"[Engine] {name} (fallback #{i}) produced {len(result)} line(s)")
                return result
            if i < len(self.engines) - 1:
                print(f"[Engine] {name} found no text; trying next engine")
            else:
                print(f"[Engine] {name} (last in chain) found no text")
            last = result
        return last


_ENGINE_BUILDERS = {
    "lens": GoogleLens,
    "oneocr": OneOCR,
    "mangaocr": MangaOCR,
    "applevision": AppleVision,
}


def _build_one(name: str) -> Engine:
    name = name.strip().lower()
    builder = _ENGINE_BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"Invalid engine: {name}")
    return builder()


def initialize_engine(engine_name: str) -> Engine:
    """Build an engine, or a fallback chain via '+': e.g. 'mangaocr+lens' tries the local
    manga-ocr pipeline first and falls back to Google Lens. Engines that fail to construct
    are skipped, so a broken primary (e.g. missing model) still leaves the fallback working.
    """
    parts = [p for p in engine_name.strip().lower().split("+") if p]
    if not parts:
        raise ValueError(f"Invalid engine: {engine_name!r}")

    if len(parts) == 1:
        return _build_one(parts[0])

    built: list[Engine] = []
    for p in parts:
        try:
            built.append(_build_one(p))
        except Exception as e:
            print(f"[Engine] '{p}' failed to initialize, skipping in fallback chain: {e}")
    if not built:
        raise ValueError(f"No engine in chain {engine_name!r} could be initialized")
    if len(built) == 1:
        return built[0]
    return FallbackEngine(built)
