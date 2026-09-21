import logging
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)

class FormulaService:
    def __init__(self):
        try:
            import torch
            from transformers import NougatProcessor, VisionEncoderDecoderModel
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = NougatProcessor.from_pretrained("facebook/nougat-small")
            self.model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-small")
            self.model.to(self.device)
            self.enabled = True
        except ImportError:
            logger.warning("transformers not installed. Formula recognition disabled.")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Nougat: {e}")
            self.enabled = False

    def extract_formula(self, image_path: str) -> Optional[str]:
        if not self.enabled:
            return None
            
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            outputs = self.model.generate(
                pixel_values,
                max_length=200,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            )
            sequence = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            sequence = self.processor.post_process_generation(sequence, fix_markdown=False)
            return sequence
        except Exception as e:
            logger.error(f"Formula extraction failed for {image_path}: {e}")
            return None
