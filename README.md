# Chromora AI – AI Image Colorization & Sketch Generation

A beginner-friendly web application that converts grayscale images into color images
using a pretrained convolutional neural network (CNN). The app is built with Python,
Flask, OpenCV, and NumPy. It runs completely on your local machine and uses only
HTML and CSS for the user interface (no JavaScript).

## Project Overview

Users upload a grayscale image through a simple web form. The image is processed by
a pretrained colorization model (Zhang et al., 2016) running through OpenCV's DNN
module. The original and colorized images are shown side by side, and the
colorized result can be downloaded.

## Objectives

- Demonstrate how a pretrained deep learning model can be integrated into a web app.
- Provide a clean, understandable codebase suitable for a software engineering fresher.
- Run fully offline on a Windows laptop.

## Features

1. Upload grayscale images using an HTML form.
2. Accept JPG, JPEG, and PNG images.
3. Validate file extensions and reject invalid uploads.
4. Show clear error messages for invalid files or processing failures.
5. Colorize images using the pretrained CNN model.
6. Display original and colorized images side by side.
7. Download the generated color image.
8. Return to the upload page to colorize another image.
9. Responsive, professional interface built with HTML and CSS only.
10. Save generated files in local `static/uploads` and `static/results` folders.
11. Secure filename handling and path-traversal protection.

## Technologies Used

- Python 3.11+ (tested with 3.13)
- Flask (web framework)
- OpenCV (image processing and model inference)
- NumPy (array operations)
- Pretrained CNN model (Zhang et al. colorization, via OpenCV DNN)
- HTML and CSS (UI, no JavaScript)

## How the AI Model Works

The model is **not** trained from scratch in this project. It uses **pretrained
weights** from the well-known colorization model by Richard Zhang, Phillip Isola,
and Alexei A. Efros ("Colorful Image Colorization", ECCV 2016).

The model is a convolutional neural network that predicts the two color channels
(`a` and `b` in the Lab color space) from the lightness channel (`L`) of an
image. OpenCV's DNN module runs the network.

Pipeline:

1. Read the image and convert it to RGB, normalised to `[0, 1]`.
2. Convert to the Lab color space. Keep the original `L` (lightness) channel.
3. Resize the image to 224x224 and extract its `L` channel.
4. Mean-centre the `L` channel (subtract 50).
5. Run the network to predict the `ab` color channels at 224x224.
6. Resize the predicted `ab` channels back to the original image size.
7. Recombine with the original `L` channel to form the output Lab image.
8. Convert back to BGR and save.

Using the original `L` channel preserves edges and texture detail.

### Model Files

| File | Purpose |
|------|---------|
| `colorization_deploy_v2.prototxt` | Network architecture (layer definitions) for OpenCV DNN. |
| `colorization_release_v2.caffemodel` | Pretrained weights (the learned parameters). |
| `pts_in_hull.npy` | 313 cluster centres in the `ab` color space; injected into the network as convolution weights. |

### Download Sources

- `colorization_deploy_v2.prototxt` and `pts_in_hull.npy`:
  - https://github.com/richzhang/colorization/tree/caffe/colorization/models
  - https://github.com/richzhang/colorization/tree/caffe/colorization/resources
  - Raw: https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt
  - Raw: https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy
- `colorization_release_v2.caffemodel` (original Berkeley link is no longer available):
  - GitHub release mirror: https://github.com/spmallick/learnopencv/releases/tag/Colorization
  - Direct asset: https://github.com/spmallick/learnopencv/releases/download/Colorization/colorization_release_v2.caffemodel
"""
The model files are large (~123 MB for the caffemodel). They are NOT committed to
this repository; download them as described in the installation section.

## Project Structure

```
AI-Image-Colorization/
├── app.py
├── colorizer.py
├── requirements.txt
├── README.md
├── models/
│   ├── colorization_deploy_v2.prototxt
│   ├── colorization_release_v2.caffemodel
│   └── pts_in_hull.npy
├── templates/
│   ├── index.html
│   └── result.html
└── static/
    ├── css/
    │   └── style.css
    ├── uploads/
    └── results/
```

## Installation (Windows + VS Code)

### 1. Check Python
Open a terminal and run:
```
python --version
```
You should see Python 3.11 or newer. If not, download from https://www.python.org/downloads/
and make sure "Add Python to PATH" is checked during installation.

### 2. Create the project directory
The project already exists at:
```
C:\Users\<you>\...\AI-Image-Colorization
```
Open this folder in VS Code: File -> Open Folder.

### 3. Create and activate a virtual environment
In the VS Code terminal:
```
python -m venv venv
venv\Scripts\activate
```
You should see `(venv)` before your prompt.

### 4. Install required packages
```
pip install -r requirements.txt
```

### 5. Download the pretrained model files
Place the three files inside the `models/` folder.

Using PowerShell (curl.exe):
```
cd models
curl.exe -L -o colorization_deploy_v2.prototxt https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt
curl.exe -L -o pts_in_hull.npy https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy
curl.exe -L -o colorization_release_v2.caffemodel https://github.com/spmallick/learnopencv/releases/download/Colorization/colorization_release_v2.caffemodel
cd ..
```

### 6. Run the Flask application
```
python app.py
```

### 7. Open in browser
Visit: http://127.0.0.1:5000

### 8. Test with a grayscale image
- Prepare a grayscale JPG/PNG image.
- Upload it through the form.
- Wait for the page to reload with the result.
- View original and colorized side by side and download the colorized image.

## Troubleshooting

- **"No module named cv2"**: Run `pip install opencv-python` inside the activated venv.
- **Model file missing**: Make sure all three files are in `models/` with exact names.
- **"The uploaded file is not a valid image"**: The file may be corrupt or not a real image. Try a different JPG/PNG.
- **Port already in use**: Change `port=5000` in `app.py` to another port (e.g. 5001).
- **Slow first request**: The model loads at startup; the first colorization may take a few seconds.
- **Colorization looks odd**: The model predicts plausible colors; accuracy is not guaranteed.

## Limitations

- Colors are predicted, not historically accurate.
- Works best on natural photographs; may produce artifacts on line art or text.
- Fixed 224x224 network input; very large images are resized internally but output keeps original dimensions.
- No user accounts, history, or database.

## Future Improvements

- Batch processing of multiple images.
- Higher resolution / better model variants.
- Image comparison slider.
- Docker packaging for easy deployment.
- Model fine-tuning on domain-specific data.

## Model Attribution

- Zhang, R., Isola, P., & Efros, A. A. (2016). Colorful Image Colorization. ECCV.
- Original project: https://github.com/richzhang/colorization
- OpenCV sample: https://github.com/opencv/opencv/blob/master/samples/dnn/colorization.py
- Caffemodel mirror: https://github.com/spmallick/learnopencv/releases/tag/Colorization

## License

This project is provided for educational purposes. The pretrained model retains its
original license from the authors.

## Color Quality Improvements

The base Zhang et al. model can sometimes produce artificial colour casts (for example
greenish backgrounds or purple hair). To improve results without using hardcoded
colours or artificial overlays, the updated `colorizer.py`:

- Converts the uploaded image to a true grayscale first, so source colour casts
  cannot leak into the prediction.
- Lightly smooths the predicted ``ab`` channels to reduce blotchy artefacts.
- Applies a small saturation scaling (0.90) to keep colours natural.
- Applies a gentle grey-world correction on ``ab`` to reduce strong colour casts.

These are model-derived adjustments. They do not recover the original colours
exactly; colourization remains a prediction.

## Model Recommendation

For a Windows laptop with limited hardware, the OpenCV DNN Zhang et al. model is
the most practical choice: it is small, CPU-friendly, and needs no GPU or paid
API. DeOldify produces more vivid results but requires PyTorch/FastAI, several
gigabytes of dependencies and weights, and is slow on CPU, so it is not
recommended for this beginner project. The project keeps the OpenCV model.


## Updated Colorization Model (DDColor)

The colorizer now uses **DDColor-tiny** (ICCV 2023) exported to ONNX and run
with `onnxruntime` on CPU. The legacy Zhang et al. Caffe model is kept as a
fallback when the ONNX file is missing.

- Model file: `models/ddcolor-tiny-fp16.onnx` (~130 MB)
- Source: https://huggingface.co/edgetools/ddcolor
- Direct download: https://huggingface.co/edgetools/ddcolor/resolve/main/ddcolor-tiny-fp16.onnx
- Expected SHA256: `2653da00dc15e54a45e5200b61dbf82ee9ceaf56b02bb9b9657569ac775e82e6`
- Extra dependency: `onnxruntime` (already in `requirements.txt`)

The ONNX model predicts chroma (ab) at 512x512 while preserving the original
image's luminance, which keeps edges, textures and lighting intact and reduces
colour bleeding compared with the old model.

## Updated Pencil Sketch

`sketcher.py` now combines:

- XDoG line art for soft, natural outlines
- Dodge-blend core lines
- Tonal shading driven by the original luminance
- Graphite grain and cross-hatching in shadowed regions
- A subtle paper texture

The result is intended to look like a handmade graphite sketch rather than a
simple inverted-image filter.

---

## Application Screenshots

### 1. Home Page
![Chromora AI Home Page](Screenshorts/Home%20Page.png)

### 2. Image Colorization & Pencil Sketch
![Colorization and Sketch Result](Screenshorts/Colored%20%26%20Sketch%20Image.png)

### 3. Sign In Page
![Sign In](Screenshorts/Sign%20In.png)

### 4. Sign Up Page
![Sign Up](Screenshorts/Sign%20Up.png)

### Account Page
![Account Page](Screenshorts/Account%20Page.png)

