import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self):
        try:
            import torch
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=torch.cuda.is_available())
            self.enabled = True
        except ImportError:
            logger.warning("PaddleOCR not installed. OCR will be disabled.")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            self.enabled = False

    def preprocess_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
        return denoised

    def extract_text(self, image_path: str) -> Dict[str, Any]:
        if not self.enabled:
            return {"text": "", "confidence": 0.0, "boxes": []}
            
        try:
            preprocessed = self.preprocess_image(image_path)
            result = self.ocr.ocr(preprocessed, cls=True)
            if not result or not result[0]:
                return {"text": "", "confidence": 0.0, "boxes": []}
            
            lines = result[0]
            extracted_text = []
            total_conf = 0.0
            boxes = []
            
            for line in lines:
                box = line[0]
                text = line[1][0]
                conf = line[1][1]
                extracted_text.append(text)
                total_conf += conf
                boxes.append({"box": box, "text": text, "confidence": conf})
                
            avg_conf = total_conf / len(lines) if lines else 0.0
            return {
                "text": "\n".join(extracted_text),
                "confidence": avg_conf,
                "boxes": boxes
            }
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return {"text": "", "confidence": 0.0, "boxes": []}
