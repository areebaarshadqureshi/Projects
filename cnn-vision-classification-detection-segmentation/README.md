# CNN Vision Suite: Classification, Detection, Segmentation & Style Transfer

Five CNN-based computer vision systems built in a single notebook, covering image classification, facial expression recognition, object detection, semantic segmentation, and neural style transfer.

## Task 1: Image Classification (CIFAR-10)

Trained a CNN from scratch to classify images into 10 categories (airplane, automobile, bird, etc.) from the CIFAR-10 dataset.

- **Loss function:** Cross-entropy loss
- **Result:** 84.8% test accuracy
- Includes preprocessing, training curves, and prediction visualizations.

## Task 2: Facial Expression Recognition (FER2013)

Trained a CNN to classify facial expressions (happy, sad, angry, etc.) from the FER2013 dataset.

- **Loss function:** Cross-entropy loss
- **Result:** 61.8% test accuracy, 0.59 macro F1 score
- Includes a full classification report and sample prediction visualizations.

## Task 3: Object Detection (YOLOv8)

Fine-tuned a pre-trained YOLOv8n model on a custom car-detection dataset, converting bounding-box annotations to YOLO format and training from there.

- **Loss function:** Combined localization loss and confidence loss (YOLO detection loss)
- **Result:** 98.0% mAP50, 67.9% mAP50-95, 99.9% precision, 95.0% recall
- Includes bounding-box visualizations on test images and training loss curves (box, classification, and confidence loss).

## Task 4: Image Segmentation (U-Net)

Built and trained a U-Net architecture from scratch for binary semantic segmentation on an image/mask dataset.

- **Loss function:** Combined BCE and Dice loss
- **Result:** 0.56 Dice coefficient, 0.39 IoU (best epoch, restored via early stopping)
- Includes accuracy/loss curves and side-by-side visualizations of input image, true mask, and predicted mask.

## Task 5: Neural Style Transfer

Implemented neural style transfer using a pre-trained VGG19 as a feature extractor, combining the content of one image with the style of another through iterative optimization.

- **Loss function:** Total variation loss, style loss, and content loss
- Includes the stylized output image after optimization.

## Tools & Libraries

Python, TensorFlow/Keras, PyTorch, Ultralytics YOLOv8, OpenCV, scikit-learn, Matplotlib, Seaborn, Pandas, NumPy

## Notes

This notebook was originally built as a coursework assignment on convolutional neural networks and is shared here as a portfolio reference covering five distinct CV task types in one place.
