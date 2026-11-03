# Skin Cancer Classification with a Convolutional Neural Network

A notebook-based machine-learning project exploring binary classification of skin-lesion images into **Benign** and **Malignant** categories using a convolutional neural network (CNN) built with TensorFlow/Keras.

> **Medical disclaimer:** This repository is an educational machine-learning project, not a medical device. Its predictions must not be used to diagnose, screen for, or make treatment decisions about skin cancer. A qualified healthcare professional should assess any concerning lesion.

## Project goals

- Prepare lesion images for a consistent model input size.
- Train and evaluate a CNN on labeled benign and malignant images.
- Save a trained model for later experimentation or inference.
- Provide a reproducible starting point for learning about computer vision in a medical-imaging context.

## Repository contents

| File | Description |
| --- | --- |
| [`ML_Final_Project.ipynb`](./ML_Final_Project.ipynb) | Main notebook for the project workflow. Open it to review the data-loading, model, training, and evaluation steps. |
| [`ML_Project_Image_Resize.ipynb`](./ML_Project_Image_Resize.ipynb) | Image-preparation notebook. The included example resizes images to 224 × 224 pixels using Pillow. Its input and output paths must be changed for your environment. |
| [`skin_cancer_best_model.h5`](./skin_cancer_best_model.h5) | Saved Keras model checkpoint (HDF5 format). Use only with compatible preprocessing and class-label mapping. |

The dataset itself is not included in this repository. The notebooks may expect files or directories that you provide separately.

## Dataset

The project is organized around two labels:

- **Benign** — images labeled as benign in the dataset.
- **Malignant** — images labeled as malignant in the dataset.

The resize notebook shows a folder-based workflow and a target image size of 224 × 224. Prepare your data so the directory names, label mapping, and folder layout match the data-loading code in `ML_Final_Project.ipynb`. For example, a suitable arrangement for a folder-based loader may look like:

```text
data/
└── Resized/
    ├── Benign/
    │   ├── image_001.jpg
    │   └── ...
    └── Malignant/
        ├── image_101.jpg
        └── ...
```

This is an illustrative layout; use the exact paths and structure expected by the notebook. Before training, verify the dataset's source, license, labels, and permitted use. Do not assume that all images in a class are representative of the wider population.

## Requirements

- Python 3 and Jupyter Notebook, or a Google Colab runtime.
- TensorFlow/Keras and the Python packages imported by the notebooks.
- Pillow for the image-resizing notebook.
- The labeled dataset, stored locally or made accessible to the notebook.

The repository does not currently include a pinned dependency file. Install packages as needed based on the notebook imports and your environment. For example:

```bash
python -m venv .venv
# Activate the virtual environment before continuing.
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux:       source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install jupyter pillow tensorflow
```

Depending on the notebook, you may also need packages such as NumPy, Matplotlib, or scikit-learn. If TensorFlow does not install in your local Python environment, use a supported Python/TensorFlow combination or run the notebook in Google Colab. A GPU can speed up training, but is not required for every workflow.

## Getting started

### Option A: Google Colab

1. Open `ML_Final_Project.ipynb` in GitHub and select **Open in Colab**, or upload the notebook to Colab.
2. If available, select a GPU runtime through **Runtime → Change runtime type**.
3. Make the dataset available to the runtime (for example, upload it or mount cloud storage).
4. Review the notebook from top to bottom. Update its dataset paths and any download or extraction steps to match your data source.
5. Run the cells in order. Check that the labels and image counts are correct before starting a training run.

### Option B: Run locally

1. Clone the repository and enter its directory:

   ```bash
   git clone https://github.com/Lakitha-Senarathna/Skin-Cancer-CNN.git
   cd Skin-Cancer-CNN
   ```

2. Create and activate a Python environment, then install Jupyter and the packages required by the notebooks (see [Requirements](#requirements)).
3. Place the dataset where the notebook can access it, and update any machine-specific paths. The resize notebook currently contains example paths specific to a Windows computer; replace them before running it elsewhere.
4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `ML_Project_Image_Resize.ipynb` if you need to resize images, then open and run `ML_Final_Project.ipynb`.

## Workflow

1. **Prepare the data:** Obtain a properly licensed, labeled dataset and confirm the class names and labels. Keep training and evaluation data separated as intended by the main notebook.
2. **Resize images:** The preparation notebook demonstrates resizing images to 224 × 224 pixels. Change its input/output directories to paths that exist in your environment.
3. **Inspect preprocessing:** Confirm the color-channel handling, pixel scaling/normalization, image dimensions, and label order in the main notebook. Inference must use the same preprocessing and label mapping as training.
4. **Train and evaluate:** Run the main notebook's data-loading, model-training, and evaluation cells. Record the data split, random seed, settings, and evaluation results when reporting experiments.
5. **Save and reuse:** The repository includes `skin_cancer_best_model.h5`. Review the notebook to confirm how that checkpoint was produced and which preprocessing and class-index mapping it expects before using it.

## Using the saved model

The model file is in Keras HDF5 format. A typical load operation is:

```python
import tensorflow as tf

model = tf.keras.models.load_model("skin_cancer_best_model.h5")
```

Loading the model alone is not enough to make a valid prediction. Match the input image size, color format, normalization/rescaling, and output-to-class mapping to the exact training pipeline in `ML_Final_Project.ipynb`. Do not infer the meaning of an output index without checking the notebook's class mapping. For an image with a different shape or preprocessing, the prediction may be invalid or misleading.

## Evaluation and responsible use

Overall accuracy alone is not sufficient for a medical-image classification task. When evaluating or extending this work, consider reporting a confusion matrix, per-class precision and recall, sensitivity/recall for the malignant class, specificity, and ROC-AUC where appropriate. Evaluate on data that was not used for training or model selection, and check for data leakage (including duplicate or near-duplicate images across splits), class imbalance, and differences between the training data and intended use.

Performance on a dataset does not establish clinical validity or safety. Dataset quality, label reliability, image acquisition, demographic representation, and external validation all affect results. Do not present this model as a substitute for clinical examination, pathology, or professional medical advice.

## Reproducibility and limitations

- The notebooks may contain environment-specific paths and dataset setup steps; review and edit these before execution.
- The dataset, dataset citation, exact split, and full dependency versions are not documented in this repository. Add those details when known to make experiments reproducible.
- Training results can vary with data, random initialization, preprocessing, package versions, and hardware.
- The included model should be treated as an experimental artifact; its provenance and performance should be verified before reuse.
- Resizing an image changes its dimensions and may affect image details. Apply consistent preprocessing to training, validation, test, and inference data.

## Contributing

Suggestions and improvements are welcome. Useful contributions include documenting the dataset and license, adding a reproducible dependency file, removing machine-specific paths, documenting class mappings and metrics, and improving validation practices. Please avoid sharing private or identifiable patient data.

## License and dataset attribution

No project license or dataset citation is specified here. Before redistributing this code, model, or any data, check the applicable licenses and permissions and add the appropriate license and dataset attribution information.

## Author

Created by [Lakitha-Senarathna](https://github.com/Lakitha-Senarathna).
