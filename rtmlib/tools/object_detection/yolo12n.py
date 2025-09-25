from typing import List, Tuple, Optional
import numpy as np
import torch
import cv2
from ultralytics import YOLO
import tempfile
import os

from ..base import BaseTool


class YOLO12n(BaseTool):
    """YOLO12n object detector with person filtering and TensorRT engine support."""

    COCO_PERSON_CLASS = 0  # Person class in YOLO COCO dataset (class 0)

    def __init__(self,
                 model_path: Optional[str] = None,
                 model_input_size: tuple = (640, 640),
                 score_thr: float = 0.5,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu',
                 export_format: str = 'engine'):
        """Initialize YOLO12n detector.

        Args:
            model_path: Path to model file. If None, will download yolo12n.pt.
            model_input_size: Input size for the model (height, width).
            score_thr: Score threshold for filtering detections.
            backend: Backend for inference ('onnxruntime', 'pytorch', or 'tensorrt').
            device: Device for inference ('cpu' or 'cuda').
            export_format: Format for export when using ONNX provider ('engine', 'onnx').
        """
        self.model_input_size = model_input_size
        self.score_thr = score_thr
        self.backend = backend
        self.device = device
        self.export_format = export_format

        # Initialize YOLO model
        if model_path is None:
            import os
            # Try to find the model file in multiple possible locations
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # First, try relative to the package installation (for installed package)
            package_model_path = os.path.join(current_dir, '..', '..', '..', '..', 'models', 'person_detector_yolo12.pt')
            package_model_path = os.path.abspath(package_model_path)

            # Second, try relative to project root (for development)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            dev_model_path = os.path.join(project_root, 'models', 'person_detector_yolo12.pt')

            # Choose the path that exists
            if os.path.exists(package_model_path):
                model_path = package_model_path
            elif os.path.exists(dev_model_path):
                model_path = dev_model_path
            else:
                # Fallback: create models directory relative to package
                models_dir = os.path.join(os.path.dirname(current_dir), 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, 'person_detector_yolo12.pt')
                raise FileNotFoundError(f"Model file not found. Please place 'person_detector_yolo12.pt' in: {models_dir}")

        self.model = YOLO(model_path)

        # Export to desired format if using ONNX provider
        if backend == 'onnxruntime':
            self.engine_path = self._export_to_engine()
            # Reinitialize with engine model
            self.model = YOLO(self.engine_path)
        elif backend == 'onnxruntimea':
            self.onnx_path = self._export_to_onnx()
            # Use base class initialization for ONNX
            super().__init__(self.onnx_path, model_input_size, backend=backend, device=device)

    def _export_to_engine(self) -> str:
        """Export YOLO model to TensorRT engine format."""
        try:
            # Get the model path and generate expected engine path
            model_path = str(self.model.ckpt_path)
            engine_path = model_path.replace('.pt', '.engine')

            # Check if engine file already exists
            if os.path.exists(engine_path):
                print(f"TensorRT engine already exists: {engine_path}")
                return engine_path

            print(f"Exporting YOLO model to TensorRT engine: {engine_path}")

            # Export to TensorRT engine format
            exported_engine_path = self.model.export(
                format='engine',
                imgsz=self.model_input_size,
                device=self.device,
                half=True if self.device == 'cuda' else False,
                workspace=4  # 4GB workspace for TensorRT
            )
            print(f"Successfully exported model to TensorRT engine: {exported_engine_path}")
            return exported_engine_path
        except Exception as e:
            print(f"Failed to export to TensorRT engine: {e}")
            print("Falling back to PyTorch inference")
            self.backend = 'pytorch'
            return None

    def _export_to_onnx(self) -> str:
        """Export YOLO model to ONNX format."""
        try:
            # Export to ONNX format
            onnx_path = self.model.export(
                format='onnx',
                imgsz=self.model_input_size,
                opset=11,
                simplify=True,
                dynamic=False
            )
            print(f"Successfully exported model to ONNX: {onnx_path}")
            return onnx_path
        except Exception as e:
            print(f"Failed to export to ONNX: {e}")
            print("Falling back to PyTorch inference")
            self.backend = 'pytorch'
            return None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Run detection on input image.

        Args:
            image: Input image as numpy array (BGR format from OpenCV).

        Returns:
            Filtered bounding boxes for person class only in format [[x1, y1, x2, y2], ...].
        """
        # if self.backend == 'onnxruntime' and self.export_format == 'onnx' and hasattr(self, 'session'):
        #     # Use ONNX runtime inference
        #     image_processed, ratio = self.preprocess(image)
        #     outputs = self.inference(image_processed)[0]
        #     results = self.postprocess(outputs, ratio)
        # else:
        # Use YOLO native inference (PyTorch or TensorRT engine)
        # Run inference with YOLO
        results = self.model(image, conf=self.score_thr, verbose=False)

        # Filter for person class only and convert to the expected format
        person_boxes = self._filter_person_detections(results[0])
        return person_boxes

        return results

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """Preprocess image for ONNX inference.

        Args:
            img: Input image.

        Returns:
            Tuple of (preprocessed_image, scale_ratio).
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize and pad image to model input size
        h, w = img_rgb.shape[:2]
        target_h, target_w = self.model_input_size

        # Calculate scale ratio
        ratio = min(target_h / h, target_w / w)
        new_h, new_w = int(h * ratio), int(w * ratio)

        # Resize image
        resized_img = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image
        padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 114

        # Place resized image in center
        start_y = (target_h - new_h) // 2
        start_x = (target_w - new_w) // 2
        padded_img[start_y:start_y + new_h, start_x:start_x + new_w] = resized_img

        # Normalize and transpose for model input
        padded_img = padded_img.astype(np.float32) / 255.0
        padded_img = np.transpose(padded_img, (2, 0, 1))  # HWC to CHW
        padded_img = np.expand_dims(padded_img, axis=0)  # Add batch dimension

        return padded_img, ratio

    def postprocess(self, outputs: np.ndarray, ratio: float) -> np.ndarray:
        """Postprocess ONNX inference outputs.

        Args:
            outputs: Raw model outputs from ONNX.
            ratio: Scale ratio from preprocessing.

        Returns:
            Filtered bounding boxes for person class.
        """
        # YOLO output format: [batch, num_detections, (x, y, w, h, conf, class_probs...)]
        if len(outputs.shape) == 3 and outputs.shape[1] > 0:
            detections = outputs[0]  # Remove batch dimension

            # Extract confidence scores and class predictions
            conf_scores = detections[:, 4]  # Confidence scores
            class_probs = detections[:, 5:]  # Class probabilities

            # Get class predictions
            class_ids = np.argmax(class_probs, axis=1)
            class_confs = np.max(class_probs, axis=1)

            # Calculate final confidence (obj_conf * class_conf)
            final_confs = conf_scores * class_confs

            # Filter by confidence threshold
            valid_mask = final_confs > self.score_thr

            # Filter for person class only (class 0)
            person_mask = class_ids == self.COCO_PERSON_CLASS

            # Combine masks
            final_mask = valid_mask & person_mask

            if np.any(final_mask):
                # Extract valid detections
                valid_detections = detections[final_mask]

                # Convert from center format (x, y, w, h) to corner format (x1, y1, x2, y2)
                boxes = valid_detections[:, :4]
                x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                # Scale boxes back to original image size
                boxes = np.column_stack([x1, y1, x2, y2]) / ratio

                return boxes

        return np.array([]).reshape(0, 4)

    def _filter_person_detections(self, results) -> np.ndarray:
        """Filter YOLO results to only include person class.

        Args:
            results: Detection results from YOLO model.

        Returns:
            Filtered bounding boxes for person class in format [[x1, y1, x2, y2], ...].
        """
        if results.boxes is None or len(results.boxes) == 0:
            return np.array([]).reshape(0, 4)

        # Get boxes data
        boxes = results.boxes

        # Filter for person class (class 0 in COCO) and confidence threshold
        person_mask = (boxes.cls == self.COCO_PERSON_CLASS) & (boxes.conf > self.score_thr)

        if not person_mask.any():
            return np.array([]).reshape(0, 4)

        # Extract bounding boxes for person detections
        # boxes.xyxy contains [x1, y1, x2, y2] format bounding boxes
        person_boxes = boxes.xyxy[person_mask].cpu().numpy()

        return person_boxes