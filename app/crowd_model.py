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


def load_model(kind: str) -> Model:
    path = ensure_weights(kind)
    model = build_csrnet()
    model.load_weights(str(path))
    return model


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
    dmap = model(x, training=False)[0, ..., 0].numpy() / DENSITY_SCALE
    return dmap[: int(np.ceil(h / DOWNSAMPLE)), : int(np.ceil(w / DOWNSAMPLE))]


# ---------------------------------------------------------------------------------------------
# Hybrid counting: a person detector for small groups, CSRNet for crowds.
# CSRNet was trained on crowds filmed from above, so on everyday photos of a few people (group
# shots, a team lined up) it "sees" heads in textures and overcounts. A detector trained on
# everyday photos (COCO) counts small groups exactly. The thresholds below were tuned on
# validation photos only (notebook 06): COCO groups, ShanghaiTech A and B.
# ---------------------------------------------------------------------------------------------
DETECTOR_URL = "https://tfhub.dev/tensorflow/efficientdet/d2/1?tf-hub-format=compressed"
DETECTOR_DIR = WEIGHTS_DIR / "detector" / "efficientdet_d2"
PERSON_CLASS = 1            # COCO label id for "person"
DET_SCORE = 0.30            # keep detections at least this confident
SMALL_GROUP_MAX = 40        # if the Street CSRNet sees at most this many people, trust the detector's exact count
DENSE_SWITCH = 250          # if the Dense CSRNet sees more than this, it's a dense crowd: use the Dense model
                            # (the Street model badly undercounts dense crowds, so it can't make this call)


def ensure_detector() -> Path:
    """Local copy of the EfficientDet-D2 SavedModel, downloaded from TensorFlow Hub if missing."""
    if not (DETECTOR_DIR / "saved_model.pb").exists():
        import tarfile, tempfile, urllib.request
        DETECTOR_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "detector.tar.gz"
            urllib.request.urlretrieve(DETECTOR_URL, archive)
            with tarfile.open(archive) as tar:
                tar.extractall(DETECTOR_DIR, filter="data")
    return DETECTOR_DIR


def load_detector():
    return tf.saved_model.load(str(ensure_detector()))


def detect_people(detector, image: Image.Image) -> np.ndarray:
    """Person boxes as (top, left, bottom, right) in pixels, confidence >= DET_SCORE."""
    out = detector(tf.convert_to_tensor(np.asarray(image, dtype=np.uint8)[None], tf.uint8))
    keep = (out["detection_classes"][0].numpy() == PERSON_CLASS) & (out["detection_scores"][0].numpy() >= DET_SCORE)
    boxes = out["detection_boxes"][0].numpy()[keep]        # normalised 0..1
    return boxes * np.array([image.height, image.width, image.height, image.width])


def draw_boxes(image: Image.Image, boxes: np.ndarray) -> Image.Image:
    """The photo with a numbered box around each detected person."""
    from PIL import ImageDraw
    out = image.copy()
    draw = ImageDraw.Draw(out)
    width = max(2, round(min(image.size) / 250))
    for i, (top, left, bottom, right) in enumerate(sorted(boxes, key=lambda b: b[1]), start=1):
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
    if street <= SMALL_GROUP_MAX:
        boxes = detect_people(detector, image)
        return CountResult(float(len(boxes)), "detector", draw_boxes(image, boxes), street)
    dense_map = predict_density(models["dense"], image)
    dense = float(max(dense_map.sum(), 0))
    if dense > DENSE_SWITCH:
        return CountResult(dense, "dense", heatmap_overlay(image, dense_map), street)
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
