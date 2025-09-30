from typing import List, Tuple, Optional, Iterator, Dict, Any
import numpy as np
import torch
import cv2
from ultralytics import YOLO
import tempfile
import os
import time
from dataclasses import dataclass
from enum import Enum

from ..base import BaseTool


class ProcessingStatus(Enum):
    """Processing status for batch operations."""
    SUCCESS = "success"
    PARTIAL_FAILURE = "partial_failure"
    COMPLETE_FAILURE = "complete_failure"
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"


@dataclass
class BatchResult:
    """Comprehensive batch processing result."""
    results: List[Optional[Any]]  # None for failed items
    status: ProcessingStatus
    successful_indices: List[int]
    failed_indices: List[int]
    error_details: Dict[int, Exception]
    processing_time: float
    memory_peak_usage: Optional[float] = None

    def get_successful_results(self) -> List[Tuple[int, Any]]:
        """Get (index, result) pairs for successful processing."""
        return [(i, self.results[i]) for i in self.successful_indices]

    def get_failure_summary(self) -> Dict[str, List[int]]:
        """Categorize failures by error type."""
        summary = {}
        for idx, error in self.error_details.items():
            error_type = type(error).__name__
            summary.setdefault(error_type, []).append(idx)
        return summary


class MemoryManager:
    """Memory management for batch processing."""

    def __init__(self, memory_limit_gb: float = 8.0):
        self.memory_limit_gb = memory_limit_gb

    def estimate_yolo_memory(self, batch_size: int, input_size: Tuple[int, int]) -> float:
        """Estimate memory usage for YOLO batch processing in GB."""
        # Base memory usage for YOLO12n model
        base_memory = 2.1  # GB

        # Per-image memory cost (input + intermediate activations)
        h, w = input_size
        per_image_memory = (h * w * 3 * 4) / (1024**3)  # Input image in float32
        per_image_memory += 0.012  # Estimated intermediate activations

        return base_memory + (batch_size * per_image_memory)

    def adaptive_batch_size(self, total_images: int, input_size: Tuple[int, int],
                           max_batch_size: int = 32) -> int:
        """Calculate optimal batch size within memory constraints."""
        for batch_size in range(min(max_batch_size, total_images), 0, -1):
            estimated_memory = self.estimate_yolo_memory(batch_size, input_size)
            if estimated_memory <= self.memory_limit_gb:
                return batch_size
        return 1  # Fallback to single image processing

    def create_processing_chunks(self, total_images: int, input_size: Tuple[int, int],
                               max_batch_size: int = 32) -> List[int]:
        """Create memory-efficient processing chunks."""
        optimal_batch_size = self.adaptive_batch_size(total_images, input_size, max_batch_size)
        chunks = []
        remaining = total_images

        while remaining > 0:
            chunk_size = min(optimal_batch_size, remaining)
            chunks.append(chunk_size)
            remaining -= chunk_size

        return chunks


class YOLO12n(BaseTool):
    """YOLO12n object detector with person filtering and TensorRT engine support."""

    COCO_PERSON_CLASS = 0  # Person class in YOLO COCO dataset (class 0)

    def __init__(self,
                 model_path: Optional[str] = None,
                 model_input_size: tuple = (640, 640),
                 score_thr: float = 0.5,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu',
                 export_format: str = 'engine',
                 max_batch_size: int = 32,
                 memory_limit_gb: float = 8.0):
        """Initialize YOLO12n detector.

        Args:
            model_path: Path to model file. If None, will download yolo12n.pt.
            model_input_size: Input size for the model (height, width).
            score_thr: Score threshold for filtering detections.
            backend: Backend for inference ('onnxruntime', 'pytorch', or 'tensorrt').
            device: Device for inference ('cpu' or 'cuda').
            export_format: Format for export when using ONNX provider ('engine', 'onnx').
            max_batch_size: Maximum batch size for batch processing.
            memory_limit_gb: Memory limit in GB for batch processing.
        """
        self.model_input_size = model_input_size
        self.score_thr = score_thr
        self.backend = backend
        self.device = device
        self.export_format = export_format
        self.max_batch_size = max_batch_size

        # Initialize batch processing components
        self.memory_manager = MemoryManager(memory_limit_gb)

        # Initialize YOLO model
        if model_path is None:
            import os
            # Try to find the model file in multiple possible locations
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # First, try in tools/models directory (current_dir is tools/object_detection)
            tools_models_path = os.path.join(current_dir, '..', 'models', 'person_detector_yolo12.pt')
            tools_models_path = os.path.abspath(tools_models_path)

            # Second, try relative to project root (for development)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            dev_model_path = os.path.join(project_root, 'models', 'person_detector_yolo12.pt')

            # Choose the path that exists
            if os.path.exists(tools_models_path):
                model_path = tools_models_path
            elif os.path.exists(dev_model_path):
                model_path = dev_model_path
            else:
                # Fallback: create models directory in tools
                models_dir = os.path.join(os.path.dirname(current_dir), 'models')
                os.makedirs(models_dir, exist_ok=True)
                model_path = os.path.join(models_dir, 'person_detector_yolo12.pt')
                raise FileNotFoundError(f"Model file not found. Please place 'person_detector_yolo12.pt' in: {models_dir}")

        self.model_path = model_path  # Store the original model path
        self.model = YOLO(model_path)

        # Export to desired format if using ONNX provider
        if backend == 'onnxruntime':
            self.engine_path = self._export_to_engine()
            # Reinitialize with engine model
            if self.engine_path:
                self.model = YOLO(self.engine_path)
        elif backend == 'onnxruntimea':
            self.onnx_path = self._export_to_onnx()
            # Use base class initialization for ONNX
            super().__init__(self.onnx_path, model_input_size, backend=backend, device=device)

    def _export_to_engine(self) -> str:
        """Export YOLO model to TensorRT engine format."""
        try:
            # Generate model-specific engine file name
            model_path = str(self.model.ckpt_path)
            base_path = model_path.replace('.pt', '')

            # Include model input size and device in engine file name for uniqueness
            size_str = f"{self.model_input_size[0]}x{self.model_input_size[1]}"
            device_str = "cuda" if self.device == 'cuda' else "cpu"
            precision_str = "fp16" if self.device == 'cuda' else "fp32"

            engine_path = f"{base_path}_{size_str}_{device_str}_{precision_str}.engine"

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

            # Rename the exported engine to our specific naming convention
            import shutil
            if exported_engine_path != engine_path and os.path.exists(exported_engine_path):
                shutil.move(exported_engine_path, engine_path)
                print(f"Renamed engine file to: {engine_path}")

            print(f"Successfully exported model to TensorRT engine: {engine_path}")
            return engine_path
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

    def __call_batch__(self, images: List[np.ndarray],
                       batch_size: Optional[int] = None,
                       progress_callback: Optional[callable] = None) -> BatchResult:
        """Run batch detection on multiple images.

        Args:
            images: List of input images as numpy arrays (BGR format from OpenCV).
            batch_size: Override automatic batch size selection. If None, uses memory-optimized batch size.
            progress_callback: Optional callback function for progress reporting.

        Returns:
            BatchResult containing filtered bounding boxes for person class only for each image.
        """
        if len(images) == 0:
            return BatchResult(
                results=[],
                status=ProcessingStatus.SUCCESS,
                successful_indices=[],
                failed_indices=[],
                error_details={},
                processing_time=0.0
            )

        start_time = time.time()

        try:
            # Determine optimal batch processing strategy
            if batch_size is None:
                processing_chunks = self.memory_manager.create_processing_chunks(
                    len(images), self.model_input_size, self.max_batch_size
                )
            else:
                # Use user-specified batch size
                processing_chunks = []
                remaining = len(images)
                while remaining > 0:
                    chunk_size = min(batch_size, remaining)
                    processing_chunks.append(chunk_size)
                    remaining -= chunk_size

            # Process images in chunks
            all_results = []
            successful_indices = []
            failed_indices = []
            error_details = {}
            processed_count = 0

            for chunk_size in processing_chunks:
                chunk_start = processed_count
                chunk_end = processed_count + chunk_size
                image_chunk = images[chunk_start:chunk_end]

                try:
                    # Process chunk with YOLO batch inference
                    chunk_results = self._process_image_chunk(image_chunk)

                    # Add successful results
                    for i, result in enumerate(chunk_results):
                        global_idx = chunk_start + i
                        all_results.append(result)
                        successful_indices.append(global_idx)

                    # Update progress
                    if progress_callback:
                        progress = (chunk_end) / len(images)
                        progress_callback(progress)

                except Exception as e:
                    # Handle chunk failure - try processing individually
                    print(f"Batch processing failed for chunk {chunk_start}-{chunk_end}, trying individual processing: {e}")

                    for i, image in enumerate(image_chunk):
                        global_idx = chunk_start + i
                        try:
                            individual_result = self.__call__(image)  # Use single-image processing
                            all_results.append(individual_result)
                            successful_indices.append(global_idx)
                        except Exception as individual_error:
                            all_results.append(None)
                            failed_indices.append(global_idx)
                            error_details[global_idx] = individual_error

                processed_count += chunk_size

            # Determine overall status
            if len(failed_indices) == 0:
                status = ProcessingStatus.SUCCESS
            elif len(successful_indices) > 0:
                status = ProcessingStatus.PARTIAL_FAILURE
            else:
                status = ProcessingStatus.COMPLETE_FAILURE

            processing_time = time.time() - start_time

            return BatchResult(
                results=all_results,
                status=status,
                successful_indices=successful_indices,
                failed_indices=failed_indices,
                error_details=error_details,
                processing_time=processing_time
            )

        except Exception as e:
            # Complete failure
            processing_time = time.time() - start_time
            return BatchResult(
                results=[None] * len(images),
                status=ProcessingStatus.COMPLETE_FAILURE,
                successful_indices=[],
                failed_indices=list(range(len(images))),
                error_details={i: e for i in range(len(images))},
                processing_time=processing_time
            )

    def _process_image_chunk(self, image_chunk: List[np.ndarray]) -> List[np.ndarray]:
        """Process a chunk of images with YOLO batch inference.

        Args:
            image_chunk: List of images to process in a single batch.

        Returns:
            List of person bounding boxes for each image.
        """
        # Use YOLO's native batch processing capabilities
        # YOLO can accept a list of images directly
        results_batch = self.model(image_chunk, conf=self.score_thr, verbose=False)

        # Process each result to filter for person class
        processed_results = []
        for results in results_batch:
            person_boxes = self._filter_person_detections(results)
            processed_results.append(person_boxes)

        return processed_results

    def process_batch_stream(self, image_iterator: Iterator[np.ndarray],
                           chunk_size: int = 100) -> Iterator[List[np.ndarray]]:
        """Stream batch processing for large datasets.

        Args:
            image_iterator: Iterator yielding images.
            chunk_size: Number of images to process in each batch.

        Yields:
            List of detection results for each processed chunk.
        """
        chunk = []

        for image in image_iterator:
            chunk.append(image)

            if len(chunk) >= chunk_size:
                batch_result = self.__call_batch__(chunk)
                yield [result for result in batch_result.results if result is not None]
                chunk = []

        # Process remaining images
        if chunk:
            batch_result = self.__call_batch__(chunk)
            yield [result for result in batch_result.results if result is not None]