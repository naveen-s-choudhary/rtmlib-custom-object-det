from typing import List, Tuple, Optional
import numpy as np
import torch
import cv2
from PIL import Image
from rfdetr import RFDETRNano as RFDETRNanoModel
from rfdetr.util.coco_classes import COCO_CLASSES
import tempfile
import os

from ..base import BaseTool


class RFDETRNano(BaseTool):
    """RF-DETR Nano object detector with person filtering and ONNX support."""
    
    COCO_PERSON_CLASS = 1  # Person class in RF-DETR COCO dataset (class 1, not 0)
    
    def __init__(self,
                 onnx_model: Optional[str] = None,
                 model_input_size: tuple = (640, 640),
                 score_thr: float = 0.5,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):
        """Initialize RFDETRNano detector.
        
        Args:
            onnx_model: Path to ONNX model. If None, will use PyTorch model and export to ONNX.
            model_input_size: Input size for the model (height, width).
            score_thr: Score threshold for filtering detections.
            backend: Backend for inference ('onnxruntime' or 'pytorch').
            device: Device for inference ('cpu' or 'cuda').
        """
        self.model_input_size = model_input_size
        self.score_thr = score_thr
        self.backend = backend
        self.device = device
        
        if backend == 'onnxruntime' and onnx_model:
            # Use existing ONNX model
            super().__init__(onnx_model, model_input_size, backend=backend, device=device)
            self.model = None
        else:
            # Use PyTorch model (RF-DETR doesn't support direct ONNX export yet)
            self.model = RFDETRNanoModel()
            self.model.optimize_for_inference()
            
            # Note: ONNX export is not supported yet, so we'll use PyTorch backend
            if backend == 'onnxruntime':
                print("Warning: ONNX export not yet implemented for RFDETRNano. Using PyTorch backend instead.")
                self.backend = 'pytorch'
    
    def _export_to_onnx(self) -> str:
        """Export RFDETRNano model to ONNX format."""
        # For now, RF-DETR ONNX export is not implemented
        # This would require access to the underlying PyTorch model
        # which may not be directly exposed by the rfdetr package
        raise NotImplementedError("ONNX export for RFDETRNano is not yet implemented")
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Run detection on input image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV).
            
        Returns:
            Filtered bounding boxes for person class only in format [[x1, y1, x2, y2], ...].
        """
        if self.backend == 'onnxruntime' and hasattr(self, 'session'):
            # Use ONNX runtime inference (if we have a valid ONNX model)
            image_processed, ratio = self.preprocess(image)
            outputs = self.inference(image_processed)[0]
            results = self.postprocess(outputs, ratio)
        else:
            # Use RF-DETR PyTorch inference
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image (RF-DETR expects PIL Image)
            pil_image = Image.fromarray(image_rgb)
            
            # Run inference with RF-DETR
            detections = self.model.predict(pil_image, threshold=self.score_thr)
            
            # Filter for person class only and convert to the expected format
            results = self._filter_person_detections(detections)
        
        return results
    
    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """Preprocess image for inference.
        
        Args:
            img: Input image.
            
        Returns:
            Tuple of (preprocessed_image, scale_ratio).
        """
        # Resize and pad image to model input size
        h, w = img.shape[:2]
        target_h, target_w = self.model_input_size
        
        # Calculate scale ratio
        ratio = min(target_h / h, target_w / w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        
        # Resize image
        import cv2
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 114
        
        # Place resized image in center
        start_y = (target_h - new_h) // 2
        start_x = (target_w - new_w) // 2
        padded_img[start_y:start_y + new_h, start_x:start_x + new_w] = resized_img
        
        return padded_img, ratio
    
    def postprocess(self, outputs: np.ndarray, ratio: float) -> np.ndarray:
        """Postprocess ONNX inference outputs.
        
        Args:
            outputs: Raw model outputs.
            ratio: Scale ratio from preprocessing.
            
        Returns:
            Filtered bounding boxes for person class.
        """
        # This is a simplified postprocessing - adjust based on actual ONNX output format
        # The exact format depends on how the ONNX model was exported
        
        if len(outputs.shape) == 3 and outputs.shape[1] > 0:
            # Assume format: [batch, num_detections, (x1, y1, x2, y2, score, class)]
            detections = outputs[0]  # Remove batch dimension
            
            # Filter by score threshold
            valid_detections = detections[detections[:, 4] > self.score_thr]
            
            # Filter for person class only (class 0 in COCO)
            person_detections = valid_detections[valid_detections[:, 5] == self.COCO_PERSON_CLASS]
            
            if len(person_detections) > 0:
                # Scale boxes back to original image size
                boxes = person_detections[:, :4] / ratio
                return boxes
        
        return np.array([]).reshape(0, 4)
    
    def _filter_person_detections(self, detections) -> np.ndarray:
        """Filter detections to only include person class.
        
        Args:
            detections: Detection results from RF-DETR model.
            
        Returns:
            Filtered bounding boxes for person class in format [[x1, y1, x2, y2], ...].
        """
        if detections is None or len(detections.class_id) == 0:
            return np.array([]).reshape(0, 4)
        
        # Filter for person class (class_id == 0 in COCO)
        person_mask = detections.class_id == self.COCO_PERSON_CLASS
        
        if not person_mask.any():
            return np.array([]).reshape(0, 4)
        
        # Extract bounding boxes for person detections
        # detections.xyxy contains [x1, y1, x2, y2] format bounding boxes
        person_boxes = detections.xyxy[person_mask]
        
        return person_boxes
