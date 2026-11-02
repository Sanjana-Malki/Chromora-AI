"""Image colorization.

Primary model: DDColor-tiny (ONNX) via onnxruntime.
Fallback: the original Zhang et al. OpenCV DNN model.
"""

import os

import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DDCOLOR_PATH = os.path.join(MODEL_DIR, "ddcolor-tiny-fp16.onnx")

PROTOTXT_PATH = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
CAFFEMODEL_PATH = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
PTS_IN_HULL_PATH = os.path.join(MODEL_DIR, "pts_in_hull.npy")

NETWORK_INPUT_SIZE = (224, 224)

SATURATION_SCALE = 0.90
CAST_CORRECTION = 0.25
AB_SMOOTH_KERNEL = 5

_onnx_session = None
_net_zhang = None
_model_error = None
_model_type = None


def _load_ddcolor():
    global _onnx_session
    if not os.path.isfile(DDCOLOR_PATH):
        return False, "Missing file: %s" % DDCOLOR_PATH
    try:
        import onnxruntime

        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        _onnx_session = onnxruntime.InferenceSession(
            DDCOLOR_PATH,
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        return True, None
    except Exception as exc:  # pylint: disable=broad-except
        _onnx_session = None
        return False, "Failed to load DDColor model (%s): %s" % (DDCOLOR_PATH, exc)


def _load_zhang():
    global _net_zhang
    missing = [p for p in (PROTOTXT_PATH, CAFFEMODEL_PATH, PTS_IN_HULL_PATH) if not os.path.isfile(p)]
    if missing:
        return False, "Missing Zhang model file(s): %s" % ", ".join(missing)
    try:
        net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
        pts_in_hull = np.load(PTS_IN_HULL_PATH)
        pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
        net.getLayer(net.getLayerId("class8_ab")).blobs = [pts_in_hull.astype(np.float32)]
        net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
            np.full([1, 313], 2.606, dtype=np.float32)
        ]
        _net_zhang = net
        return True, None
    except Exception as exc:  # pylint: disable=broad-except
        _net_zhang = None
        return False, "Failed to load Zhang model: %s" % exc


try:
    ddcolor_ok, ddcolor_err = _load_ddcolor()
    if ddcolor_ok:
        _model_type = "ddcolor"
    else:
        zhang_ok, zhang_err = _load_zhang()
        if zhang_ok:
            _model_type = "zhang"
        else:
            errors = [e for e in (ddcolor_err, zhang_err) if e]
            _model_error = (
                "No colorization model found. Expected %s or the Zhang model files in %s. %s"
                % (os.path.basename(DDCOLOR_PATH), MODEL_DIR, " ".join(errors))
            ).strip()
except Exception as exc:  # pylint: disable=broad-except
    _model_error = str(exc)


def is_model_loaded():
    return _model_type is not None


def get_model_error():
    return _model_error
def _colorize_ddcolor(image_bgr, output_path):
    height, width = image_bgr.shape[:2]

    img_f = (image_bgr / 255.0).astype(np.float32)
    orig_l = cv2.cvtColor(img_f, cv2.COLOR_BGR2Lab)[:, :, :1]

    resized = cv2.resize(img_f, (512, 512), interpolation=cv2.INTER_AREA)
    resized_l = cv2.cvtColor(resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    gray_lab = np.concatenate(
        (resized_l, np.zeros_like(resized_l), np.zeros_like(resized_l)), axis=-1
    )
    gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)

    tensor = gray_rgb.transpose(2, 0, 1)[None].astype(np.float32)

    input_name = _onnx_session.get_inputs()[0].name
    output = _onnx_session.run(None, {input_name: tensor})[0][0]
    output_ab = output.transpose(1, 2, 0)

    output_ab = cv2.resize(output_ab, (width, height), interpolation=cv2.INTER_CUBIC)

    output_lab = np.concatenate((orig_l, output_ab), axis=-1)
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    output_bgr = np.clip(output_bgr, 0.0, 1.0)
    output_bgr = (output_bgr * 255.0).round().astype(np.uint8)

    ok = cv2.imwrite(output_path, output_bgr)
    if not ok:
        raise RuntimeError("Failed to save the colorized image to disk.")
    return output_path
def _colorize_zhang(image_bgr, output_path):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image_rgb = (gray_bgr[:, :, [2, 1, 0]] * 1.0 / 255.0).astype(np.float32)

    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
    lightness = image_lab[:, :, 0]
    original_shape = image_rgb.shape[:2]

    resized_rgb = cv2.resize(image_rgb, NETWORK_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    resized_lab = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2Lab)
    resized_l = resized_lab[:, :, 0] - 50.0

    blob = cv2.dnn.blobFromImage(resized_l)
    _net_zhang.setInput(blob)
    prediction = _net_zhang.forward()
    ab_decoded = prediction[0, :, :, :].transpose((1, 2, 0))

    if AB_SMOOTH_KERNEL and AB_SMOOTH_KERNEL > 1:
        ab_decoded = cv2.GaussianBlur(ab_decoded, (AB_SMOOTH_KERNEL, AB_SMOOTH_KERNEL), 0)

    ab_decoded = ab_decoded * SATURATION_SCALE

    mean_a = float(np.mean(ab_decoded[:, :, 0]))
    mean_b = float(np.mean(ab_decoded[:, :, 1]))
    ab_decoded[:, :, 0] -= CAST_CORRECTION * mean_a
    ab_decoded[:, :, 1] -= CAST_CORRECTION * mean_b

    ab_resized = cv2.resize(ab_decoded, (original_shape[1], original_shape[0]),
                            interpolation=cv2.INTER_CUBIC)

    output_lab = np.concatenate((lightness[:, :, np.newaxis], ab_resized), axis=2)
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_Lab2BGR)
    output_bgr = np.clip(output_bgr, 0.0, 1.0)
    output_bgr = (output_bgr * 255.0).astype(np.uint8)

    ok = cv2.imwrite(output_path, output_bgr)
    if not ok:
        raise RuntimeError("Failed to save the colorized image to disk.")
    return output_path


def colorize_image(input_path, output_path):
    if _model_type is None:
        raise RuntimeError("Colorization model is not loaded: %s" % (_model_error or "unknown error"))

    image_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not read the uploaded image. The file may be corrupt or unsupported.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if _model_type == "ddcolor":
        return _colorize_ddcolor(gray_bgr, output_path)
    return _colorize_zhang(gray_bgr, output_path)
