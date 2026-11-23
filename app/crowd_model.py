"""CSRNet counting model used by the Streamlit app.

Rebuilds the exact architecture trained in notebooks/02 and 03, loads the saved
weights, and turns a photo into a density map whose sum is the people count.
"""
import os

# oneDNN made full-image inference ~60x slower on CPU (see notebooks); must be set before importing TF.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import colormaps
from PIL import Image
from tensorflow.keras import Model, layers

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
# Trained weights are too big for the git repo; they are published as GitHub Release assets and
# downloaded on first use (e.g. on Streamlit Community Cloud). Inference-only copies, 65 MB each.
WEIGHTS_URL = "https://github.com/Charan1845/crowdcast/releases/download/weights-v1/{}"

DOWNSAMPLE = 8            # CSRNet output is 1/8 of the input size
DENSITY_SCALE = 100.0     # targets were multiplied by this during training
MAX_SIDE = 1024           # training photos were at most 1024 px on their longest side
MEAN_RGB = np.array([124, 117, 104], dtype=np.uint8)   # ImageNet mean; padding with it is ~0 after preprocessing


@dataclass(frozen=True)
class ModelInfo:
    label: str
    weights_file: str
    dataset: str
    test_mae: float
    test_mean_count: int


MODELS = {
    "dense": ModelInfo("Dense crowd", "csrnet_partA_v2_best.weights.h5", "ShanghaiTech Part A", 79.6, 433),
    "street": ModelInfo("Street scene", "csrnet_partB_best.weights.h5", "ShanghaiTech Part B", 13.4, 124),
}


def build_csrnet() -> Model:
    """Same layers, in the same order, as the training notebooks (VGG-16 up to block4_conv3 + dilated back end)."""
    vgg = tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=(None, None, 3))
    front = Model(vgg.input, vgg.get_layer("block4_conv3").output, name="vgg16_front")
    x = front.output
    for i, ch in enumerate([512, 512, 512, 256, 128, 64]):
        x = layers.Conv2D(ch, 3, padding="same", dilation_rate=2, activation="relu", name=f"backend_conv{i+1}")(x)
    out = layers.Conv2D(1, 1, name="density")(x)
    return Model(front.input, out, name="csrnet")


def ensure_weights(kind: str) -> Path:
    """Local path to the model's weights, downloading them from the GitHub Release if missing."""
    path = WEIGHTS_DIR / MODELS[kind].weights_file
    if not path.exists():
        import urllib.request
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(WEIGHTS_URL.format(MODELS[kind].weights_file), tmp)
        tmp.replace(path)   # only a complete download gets the real name
    return path


def load_model(kind: str):
    """A compiled (graph-mode) CSRNet: same results as calling the Keras model directly, but ~40% less
    peak memory on a full-size photo (909 vs 1,524 MB) and much faster, which is what fits on free hosting."""
    path = ensure_weights(kind)
    model = build_csrnet()
    model.load_weights(str(path))
    run = tf.function(lambda x: model(x, training=False),
                      input_signature=[tf.TensorSpec([1, None, None, 3], tf.float32)])
    run.keras_model = model   # keep a reference so the weights stay alive
    return run


def prepare_image(image: Image.Image) -> tuple[Image.Image, bool]:
    """RGB, and shrunk so the longest side is at most MAX_SIDE. Returns (image, was_resized)."""
    image = image.convert("RGB")
    if max(image.size) <= MAX_SIDE:
        return image, False
    scale = MAX_SIDE / max(image.size)
    new_size = (round(image.width * scale), round(image.height * scale))
    return image.resize(new_size, Image.LANCZOS), True


def predict_density(model: Model, image: Image.Image) -> np.ndarray:
    """Density map at 1/8 resolution (cropped to the image area). Its sum is the predicted count."""
    arr = np.asarray(image, dtype=np.uint8)
    h, w = arr.shape[:2]
    hp, wp = (int(np.ceil(v / DOWNSAMPLE)) * DOWNSAMPLE for v in (h, w))
    padded = np.empty((hp, wp, 3), dtype=np.uint8)
    padded[:] = MEAN_RGB
    padded[:h, :w] = arr
    x = tf.keras.applications.vgg16.preprocess_input(padded.astype(np.float32))[None]
    dmap = model(tf.constant(x))[0, ..., 0].numpy() / DENSITY_SCALE   # model = compiled CSRNet from load_model
    return dmap[: int(np.ceil(h / DOWNSAMPLE)), : int(np.ceil(w / DOWNSAMPLE))]


# ---------------------------------------------------------------------------------------------
# Hybrid counting: a person detector for small groups, CSRNet for crowds.
# CSRNet was trained on crowds filmed from above, so on everyday photos of a few people (group
# shots, a team lined up) it "sees" heads in textures and overcounts. A detector trained on
# everyday photos (COCO) counts small groups exactly. The thresholds below were tuned on
# validation photos only (notebook 06): COCO groups, ShanghaiTech A and B.
# ---------------------------------------------------------------------------------------------
# EfficientDet-Lite4 (TensorFlow Lite): as accurate as the full EfficientDet-D2 on our validation photos
# (MAE 1.47 vs 1.48) but ~64 MB of memory instead of ~1.35 GB, which is what fits on free hosting.
DETECTOR_URL = "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite4/detection/default/2?lite-format=tflite"
DETECTOR_FILE = WEIGHTS_DIR / "detector" / "efficientdet_lite4.tflite"
PERSON_CLASS = 0            # "person" in the TFLite model's label map
DET_SCORE = 0.25            # keep detections at least this confident (lowest validation error)
DET_MAX = 25                # the TFLite model returns at most 25 objects; a full 25 may mean more people
SMALL_GROUP_MAX = 40        # if the Street CSRNet sees at most this many people, trust the detector's exact count
DENSE_SWITCH = 250          # if the Dense CSRNet sees more than this, it's a dense crowd: use the Dense model
                            # (the Street model badly undercounts dense crowds, so it can't make this call)
# Larger groups (notebook 07, tuned on validation photos only). The detector sees a 640 px copy of the photo and
# returns 25 objects at most, so once it finds several people it looks again at four overlapping tiles.
TILE_TRIGGER = 5            # run the tiles when the whole photo shows at least this many people
TILE_OVERLAP = 0.25         # how much neighbouring tiles overlap, so nobody is lost on a tile edge
FULL_IOU, FULL_IOM = 0.5, 0.85   # duplicate-box limits for the whole-photo pass
TILE_IOU, TILE_IOM = 0.4, 0.7    # stricter limits once tiles are merged in (one person can appear in two tiles)
CUT_MARGIN = 0.02           # a box this close to an inner tile edge (share of tile size) was cut off by that edge
JOIN_ALIGN = 0.5            # two cut boxes are one person if they line up by at least this much along the edge
GROUP_MIN = 8               # CSRNet says "crowd", but the detector found at least this many people...
GROUP_COVER = 0.15          # ...and their boxes cover under this share of the photo: it's a group, trust the detector


def ensure_detector() -> Path:
    """Local copy of the EfficientDet-Lite4 TFLite model, downloaded from TensorFlow Hub if missing."""
    if not DETECTOR_FILE.exists():
        import urllib.request
        DETECTOR_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = DETECTOR_FILE.with_suffix(".part")
        urllib.request.urlretrieve(DETECTOR_URL, tmp)
        if tmp.stat().st_size < 1_000_000:   # an error page, not the model
            tmp.unlink()
            raise RuntimeError("Could not download the person detector from TensorFlow Hub.")
        tmp.replace(DETECTOR_FILE)
    return DETECTOR_FILE


class PersonDetector:
    """EfficientDet-Lite4 via the TFLite interpreter. A lock keeps it safe when several visitors count at once."""

    def __init__(self, path: Path):
        import threading, warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # tf.lite.Interpreter prints a deprecation notice
            self.interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=4)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.size = int(self.input["shape"][1])
        self.outputs = self.interpreter.get_output_details()   # boxes, classes, scores, count
        self.lock = threading.Lock()

    def __call__(self, image: Image.Image) -> np.ndarray:
        """Person boxes as (top, left, bottom, right, score) in pixels, confidence >= DET_SCORE."""
        x = np.asarray(image.convert("RGB").resize((self.size, self.size)), dtype=np.uint8)[None]
        with self.lock:
            self.interpreter.set_tensor(self.input["index"], x)
            self.interpreter.invoke()
            boxes, classes, scores = (self.interpreter.get_tensor(o["index"])[0] for o in self.outputs[:3])
        keep = (classes == PERSON_CLASS) & (scores >= DET_SCORE)
        pixels = boxes[keep] * np.array([image.height, image.width, image.height, image.width])   # boxes are 0..1
        return np.column_stack([pixels, scores[keep]])


def load_detector() -> PersonDetector:
    return PersonDetector(ensure_detector())


def merge_boxes(boxes: np.ndarray, iou_limit: float, contain_limit: float) -> np.ndarray:
    """Keep the most confident box for each person. A box is a duplicate if it overlaps a kept one by more than
    iou_limit, or if more than contain_limit of the smaller box lies inside it (a person cut in half by a tile edge)."""
    kept = []
    for box in boxes[np.argsort(-boxes[:, 4])]:
        duplicate = False
        for other in kept:
            h = max(0, min(box[2], other[2]) - max(box[0], other[0]))
            w = max(0, min(box[3], other[3]) - max(box[1], other[1]))
            shared = w * h
            a = (box[2] - box[0]) * (box[3] - box[1])
            b = (other[2] - other[0]) * (other[3] - other[1])
            if shared / (a + b - shared + 1e-9) > iou_limit or shared / (min(a, b) + 1e-9) > contain_limit:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return np.array(kept, dtype=np.float32).reshape(-1, 5)


def tile_grid(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Four overlapping tiles as (left, top, right, bottom)."""
    tile_w, tile_h = width / (2 - TILE_OVERLAP), height / (2 - TILE_OVERLAP)
    return [(round(i * tile_w * (1 - TILE_OVERLAP)), round(j * tile_h * (1 - TILE_OVERLAP)),
             round(i * tile_w * (1 - TILE_OVERLAP) + tile_w), round(j * tile_h * (1 - TILE_OVERLAP) + tile_h))
            for j in range(2) for i in range(2)]


def join_cut_boxes(tiles: list[tuple[tuple[int, int, int, int], np.ndarray]], width: int, height: int) -> np.ndarray:
    """A person too big for one tile is found as two halves. Join halves that meet across a tile edge into one box."""
    items = []
    for index, ((left, top, right, bottom), found) in enumerate(tiles):
        margin = CUT_MARGIN * max(right - left, bottom - top)
        for box in found:
            items.append((box, index, bottom < height and box[2] >= bottom - margin, top > 0 and box[0] <= top + margin,
                          right < width and box[3] >= right - margin, left > 0 and box[1] <= left + margin))
    pairs = []
    for i, (a, tile_a, cut_bottom, _, cut_right, _) in enumerate(items):
        for j, (b, tile_b, _, cut_top, _, cut_left) in enumerate(items):
            if tile_a == tile_b:
                continue
            if cut_bottom and cut_top:       # top and bottom halves must line up left to right
                start, end = 1, 3
            elif cut_right and cut_left:     # left and right halves must line up top to bottom
                start, end = 0, 2
            else:
                continue
            shared = min(a[end], b[end]) - max(a[start], b[start])
            span = max(a[end], b[end]) - min(a[start], b[start])
            if shared > 0 and shared / span >= JOIN_ALIGN:
                pairs.append((shared / span, i, j))
    used, joined = set(), []
    for _, i, j in sorted(pairs, key=lambda pair: -pair[0]):
        if i not in used and j not in used:
            used |= {i, j}
            a, b = items[i][0], items[j][0]
            joined.append([min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), max(a[4], b[4])])
    rest = [item[0].tolist() for n, item in enumerate(items) if n not in used]
    return np.array(joined + rest, dtype=np.float32).reshape(-1, 5)


def detect_people(detector: PersonDetector, image: Image.Image) -> tuple[np.ndarray, int]:
    """Returns (merged person boxes, how many the whole-photo pass found). Tiles run only for groups of 5 or more."""
    raw = detector(image)
    whole = merge_boxes(raw, FULL_IOU, FULL_IOM)
    if len(whole) < TILE_TRIGGER:
        return whole, len(whole)
    tiles = []
    for left, top, right, bottom in tile_grid(image.width, image.height):
        found = detector(image.crop((left, top, right, bottom))).copy()
        found[:, [0, 2]] += top
        found[:, [1, 3]] += left
        tiles.append(((left, top, right, bottom), found))
    joined = join_cut_boxes(tiles, image.width, image.height)
    return merge_boxes(np.concatenate([raw, joined]), TILE_IOU, TILE_IOM), len(whole)


def box_coverage(boxes: np.ndarray, image: Image.Image) -> float:
    """Total box area as a share of the photo. Real crowds fill the frame with boxes; a group in a scene doesn't."""
    return float(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).sum() / (image.width * image.height))


def draw_boxes(image: Image.Image, boxes: np.ndarray) -> Image.Image:
    """The photo with a numbered box around each detected person."""
    from PIL import ImageDraw
    out = image.copy()
    draw = ImageDraw.Draw(out)
    width = max(2, round(min(image.size) / 250))
    for i, (top, left, bottom, right) in enumerate(sorted(boxes[:, :4].tolist(), key=lambda b: b[1]), start=1):
        draw.rectangle([left, top, right, bottom], outline="#38BDAE", width=width)
        label = str(i)
        tw, th = draw.textbbox((0, 0), label)[2:]
        draw.rectangle([left, top, left + tw + 6, top + th + 4], fill="#38BDAE")
        draw.text((left + 3, top + 1), label, fill="#0D1117")
    return out


@dataclass
class CountResult:
    count: float
    method: str              # "detector", "street" or "dense"
    overlay: Image.Image     # boxes (detector) or heatmap (density)
    street_estimate: float   # CSRNet Street model's count, used for routing


def count_auto(image: Image.Image, models: dict, detector) -> CountResult:
    """Pick the right method for the photo and count. `models` holds the "street" and "dense" CSRNets."""
    street_map = predict_density(models["street"], image)
    street = float(max(street_map.sum(), 0))
    boxes, whole_photo = detect_people(detector, image)
    if street <= SMALL_GROUP_MAX:
        # A full 25 may mean the detector ran out of slots: trust it only if the tiles found more people.
        if whole_photo < DET_MAX or len(boxes) > DET_MAX:
            return CountResult(float(len(boxes)), "detector", draw_boxes(image, boxes), street)
        return CountResult(street, "street", heatmap_overlay(image, street_map), street)
    dense_map = predict_density(models["dense"], image)
    dense = float(max(dense_map.sum(), 0))
    if dense > DENSE_SWITCH:   # checked first: in a packed crowd the detector finds only a few faces
        return CountResult(dense, "dense", heatmap_overlay(image, dense_map), street)
    if len(boxes) >= GROUP_MIN and box_coverage(boxes, image) < GROUP_COVER:   # CSRNet overcounting a group
        return CountResult(float(len(boxes)), "detector", draw_boxes(image, boxes), street)
    return CountResult(street, "street", heatmap_overlay(image, street_map), street)


def heatmap_overlay(image: Image.Image, dmap: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """Blend a coloured density map over the photo (transparent where nobody was found)."""
    d = np.maximum(dmap, 0)
    top = np.quantile(d, 0.98)   # scale to the busy areas, not the single hottest pixel
    d = np.clip(d / top, 0, 1) if top > 0 else d
    heat = Image.fromarray(np.uint8(d * 255)).resize(image.size, Image.BILINEAR)
    h = np.asarray(heat, dtype=np.float32) / 255
    colours = colormaps["turbo"](h)[..., :3]
    weight = (alpha * np.sqrt(h))[..., None]   # stronger colour where denser, none where empty
    base = np.asarray(image, dtype=np.float32) / 255
    blended = base * (1 - weight) + colours * weight
    return Image.fromarray(np.uint8(np.clip(blended, 0, 1) * 255))
