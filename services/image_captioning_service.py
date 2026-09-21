import logging
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)

class ImageCaptioningService:
    def __init__(self):
        try:
            import torch
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model.to(self.device)
            self.enabled = True
        except ImportError:
            logger.warning("transformers or torch not installed. Image captioning disabled.")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize BLIP2: {e}")
            self.enabled = False

    def generate_caption(self, image_path: str) -> Optional[str]:
        if not self.enabled:
            return None
            
        try:
            raw_image = Image.open(image_path).convert('RGB')
            inputs = self.processor(raw_image, return_tensors="pt").to(self.device, self.model.dtype)
            
            out = self.model.generate(**inputs, max_new_tokens=50)
            caption = self.processor.decode(out[0], skip_special_tokens=True).strip()
            return caption
        except Exception as e:
            logger.error(f"Caption generation failed for {image_path}: {e}")
            return None
