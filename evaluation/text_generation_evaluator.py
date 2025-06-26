from paddleocr import PaddleOCR
from paddleocr import FormulaRecognition
from Tools.ocr.doc_parsing_evaluator import ParsingEvaluator
from Tools.ocr.ocr_evaluator import OcrEvaluator
import os
import re
import json
from typing import Dict, List, Optional, Tuple, Any
import cv2

class TextGenerationEvaluator:
    
    def __init__(self):
        self._ocr_models = {}
        self._formula_model = None
        

    def _get_ocr_model(self, lang: str) -> PaddleOCR:
        if lang not in self._ocr_models.keys():
            self._ocr_models[lang] = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang=lang
            )
        return self._ocr_models[lang]
    
    def _get_formula_model(self) -> FormulaRecognition:
        if self._formula_model is None:
            self._formula_model = FormulaRecognition(model_name="PP-FormulaNet-L")
        return self._formula_model
    
    def remove_english(self, text: str) -> str:
        return re.sub(r'[A-Za-z]', '', text)
    
    def ocr_images(self, img_path: str, lang: str = 'fr', ocr_type: str = "doc") -> str:
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image path not found: {img_path}")

        image = cv2.imread(img_path)
        if ocr_type == "formula":
            model = self._get_formula_model()
            output = model.predict(input=image, batch_size=1)
            return output[0].get("rec_formula", "")
        else:
            ocr = self._get_ocr_model(lang)
            results = ocr.predict(input=image)
            texts = results[0].get("rec_texts", [])
            return " ".join(texts)
    
    def evaluate_metrics(self, texts: str, gt_texts: str, ocr_type: str, lang: str) -> Any:
        gt_info = {"image_name": gt_texts}
        response_info = {"image_name": texts}
        
        if ocr_type == "formula":
            evaluator = ParsingEvaluator()
            score = evaluator.evaluate_single_formula_sample(gt=gt_texts, pred=texts)
            return score, "NED"
        elif ocr_type == "doc":
            evaluator = ParsingEvaluator()
            score = evaluator.evaluate_single_doc_sample(gt=gt_texts, pred=texts)
            return score, "NED"
        else:
            evaluator = OcrEvaluator()
            results = evaluator.evaluate(
                response_info=response_info, 
                gt_info=gt_info, 
                lang=lang
            )["summary"]["mirco_f1_score"]
            return results, "micro_f1_score"
    
    def evaluate(self, generated_image_path, eval_material_path: str) -> Dict[str, Any]:
        with open(eval_material_path, "r", encoding="utf-8") as f:
            eval_info = json.load(f)
        
        if isinstance(eval_info, dict):
            gt_text = eval_info.get("gt_text")
            ocr_type = eval_info.get("ocr_type")
            lang = eval_info.get("lang")
        else:
            raise ValueError("The content of eval_material_path must be a dictionary containing the fields 'gt_text', 'ocr_type', and 'lang'.")
        
        
        if not os.path.exists(generated_image_path):
            raise FileNotFoundError(f"can not find image: {generated_image_path}")
        
        pred_text = self.ocr_images(generated_image_path, lang=lang, ocr_type=ocr_type)
        
        if (lang == "ch" or lang == "japan") and ocr_type == "normal":
            pred_text = self.remove_english(pred_text)
        
        metrics, metric_name = self.evaluate_metrics(pred_text, gt_text, ocr_type, lang)

        return f"recognized text: {pred_text}, {metric_name} score: {metrics}"
