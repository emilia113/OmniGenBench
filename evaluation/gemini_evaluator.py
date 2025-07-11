import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import google.generativeai as genai
import random
class EvaluationType(Enum):
    INSTRUCTION_FOLLOWING = "INSTRUCTION_FOLLOWING"
    AESTHETIC = "AESTHETIC"
    REALISM = "REALISM"


@dataclass
class EvaluationConfig:
    model_name: str = "Gemini 2.5 Pro 06-05"
    supported_extensions: tuple = (".png", ".jpg", ".jpeg", ".webp")
    score_range: tuple = (1, 5)







class GeminiImageEvaluator:
    
    INSTRUCTIONS = {
        EvaluationType.INSTRUCTION_FOLLOWING: [
    """Image-Generation Alignment Evaluation Prompt (with Criteria Elements Checklist)

You are an expert image evaluator. For each task, you will receive:

1. Prompt Text – a natural-language description specifying what the image should depict.
2. Criteria Elements Checklist – a list of specific visual elements that should appear in a correct image for this prompt.
3. (Optional) Reference Images(input conditions used during generation, if any)
4. (Optional) Auxiliary Information(additional information recognized from images by detection models or OCR models)
5. Candidate Image – the image generated by the model.

Your job is to judge how well the candidate image matches the prompt, using the Criteria Elements Checklist as concrete evidence.

Evaluation Dimension:
Alignment Between Candidate Image and Prompt Description
Compare the Candidate Image against every item in the Criteria Elements Checklist. Consider objects, attributes, spatial relations, and implied scene dynamics. The more checklist items the image satisfies, the better its alignment with the Prompt Text.

Five-Level Scoring Rubric:
Score 5 – Completely satisfies: The image fulfills all checklist items; every required element, attribute, and relation is present and correct.
Score 4 – Mostly satisfies: The image fulfills most checklist items; only minor or non-critical details are missing or slightly off.
Score 3 – Half satisfies: About half of the checklist items are correctly depicted; several important elements are absent or inaccurate.
Score 2 – Slightly satisfies: Only a few checklist items appear correctly; most required elements are missing or wrong.
Score 1 – Does not satisfy: The image fails to meet any checklist item; scene is unrelated to the prompt.

Example:
Prompt Text:
“An egg is completely broken, with eggshell scattered around and egg white and yolk clearly spilling out.”

Criteria Elements Checklist:
- Broken eggshell pieces scattered
- Egg white visible outside shell
- Egg yolk visible outside shell

Output Format:
1. Provide a step-by-step explanation of how you reached your decision, explicitly referencing which checklist items are present or missing.
2. Conclude with a single line:
Final Score: X
Replace X with 1, 2, 3, 4, or 5.

Input:
Prompt Text: """,

    """Criteria Elements Checklist:""",
    """- Reference Images(input conditions used during generation, if any):""",
    """- Auxiliary Information(additional information recognized from images by detection models or OCR models):""",
    """- A Candidate Image(generated by a model):""",
],

        EvaluationType.AESTHETIC: [
"""You are an expert image evaluator. For each task, you will be provided with an image. Your task is to independently assess the image along the following dimension and assign an integer score from 1 to 5:

Evaluation Dimension: Aesthetic Quality

Assess the overall artistic appeal and visual harmony of the image. Consider the image's composition, color and lighting coordination, and attention to visual detail.

Key Considerations:

1. Composition Quality
Is the visual layout well-organized and balanced? Does the image show a clear structure with appropriate placement of elements?

2. Color and Lighting Harmony
Are the colors well-matched and consistent? Does the image exhibit a coherent tone, with harmonious color combinations and natural transitions of light and shadow?

3. Detail Refinement
Are the edges, textures, and materials handled carefully? Is there a sense of visual polish and completeness, without obvious flaws or careless execution?

Scoring Guidelines:

5 – The image is beautifully composed, with excellent visual balance, harmonious color and lighting, and highly refined details throughout.
4 – The image is aesthetically pleasing, with mostly strong composition and coordinated colors, though some minor areas may lack polish.
3 – The image shows basic aesthetic appeal, but has noticeable issues in layout, color harmony, or detail quality.
2 – The image lacks cohesion, with disorganized composition, clashing colors, or rough details.
1 – The image is visually unappealing or poorly executed, showing major flaws in composition, color use, or detail.

Output Format
After the evaluation, conclude clearly with the final score, formatted as:
Final Score: X

Image:"""
],

        EvaluationType.REALISM: [
"""You are an expert image evaluator. For each task, you will be shown an image generated by a model.Your job is to independently assess the image along the following dimension and assign an integer score from 1 to 5:

Evaluation Dimension: Realism and Generation Quality

I. Input

1. The image itself
2. The corresponding generation prompt (or textual description)

II. Evaluation Process

1. Determine the image type

* 3D Scene Image: Includes real-world or 3D-rendered scenes/people/objects, involving perspective, lighting, materials, and other three-dimensional elements.
* 2D Flat Image: Includes visualizations, scientific diagrams, flowcharts, instructional illustrations, document screenshots, etc., based on flat layout and graphic/textual content.

2. Apply the appropriate realism evaluation criteria

A. For 3D Scene Images

* Sharpness: Is the image clear overall? Are details sharp? Is there noticeable blur in the whole or parts of the image?
* Natural Plausibility:
  * Do visual elements follow common sense (e.g., no extra fingers/limbs, no distorted or deformed anatomy)?
  * Are key elements present? Are there any extraneous or illogical objects?
* Physical and Visual Consistency:
  * Are lighting direction, intensity, and shadows coherent?
  * Do perspective and scale follow real-world photography or 3D logic?
  * Are physical laws (gravity, reflection, refraction, etc.) properly reflected?

B. For 2D Flat Images

* No distortion: Is the image sharp overall? Is there any obvious blur or compression artifact?
* Graphic and Text Quality:
  * Are edges smooth without jagged or pixelated artifacts?
  * Is the text legible? Any misalignment, ghosting, gibberish, or character errors?
* Information Plausibility and Consistency:
  * For instruction manuals, rule diagrams, or scientific concept illustrations — is the content logical and factually accurate?
  * Are legends, axes, units, and labels consistent and correct?

III. Scoring Criteria

Scoring for 3D Scene Images

Score 5: Image is sharp, visually coherent, and all elements are physically realistic.
Score 4: Image is clear, most elements look real, only minor and negligible issues.
Score 3: Image is mostly clear, but contains several notable unrealistic or inconsistent elements.
Score 2: Image is clearly blurry or includes major distortions / misalignments / physics violations.
Score 1: Image is extremely blurry, incoherent, or highly implausible; realism is nearly absent.

Scoring for 2D Flat Images

Score 5: Image and text are sharp, undistorted, with fully accurate and realistic information.
Score 4: Clear image with only minor distortion or layout imperfections.
Score 3: Mostly clear but contains visible edge distortion, character errors, or mild info inconsistency.
Score 2: Noticeable blur, compression artifacts, or multiple layout/text errors; partially inaccurate content.
Score 1: Severely blurry, structurally chaotic, or contains numerous factual errors; hard to interpret or unreliable.

IV. Output Format

Final Score: X

Only output this single line (replace X with an integer between 1 and 5). No additional explanation is required.

Input:
Prompt Text:""",
    """Image:"""
]

    }

        
    def __init__(self,
                 config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.model = genai.GenerativeModel(self.config.model_name)
        
        self._setup_logging()
        self.evaluation_type_map = [
            EvaluationType.INSTRUCTION_FOLLOWING,
            EvaluationType.AESTHETIC,
            EvaluationType.REALISM
        ]
    
    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.logger.handlers.clear()
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _img_to_vision_dict_genai(self, img_path: str) -> Dict[str, Any]:
        img_path = Path(img_path)
        with img_path.open("rb") as f:
            data = f.read()
        
        mime = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        return {"mime_type": mime, "data": data}
        
    def _prepare_content(self, 
                        evaluation_type: EvaluationType,
                        generated_image_path: str,
                        prompt: Optional[str] = None,
                        condition_image_path_list: Optional[str] = None,
                        eval_material_path: Optional[List[str]] = None,
                        auxiliary_info: Optional[Union[List[str], str]] = None) -> List[Any]:
        instruction_template = self.INSTRUCTIONS[evaluation_type]
        instruction_content = []
        
        if evaluation_type == EvaluationType.INSTRUCTION_FOLLOWING:
            if prompt is None:
                raise ValueError("Prompt is required for INSTRUCTION_FOLLOWING evaluation")
            if eval_material_path is None:
                raise ValueError("eval_material_path is required for INSTRUCTION_FOLLOWING evaluation")
            
            instruction_content.append(instruction_template[0] + prompt)
            
            with open(eval_material_path, "r", encoding="utf-8") as f:
                criteria = json.load(f)
                if isinstance(criteria, list):
                    criteria_str = criteria[0]
                elif isinstance(criteria, dict):
                    criteria_str = criteria.get("juding_criteria", "")
                    if isinstance(criteria_str, list):
                        criteria_str = criteria_str[0]
                elif isinstance(criteria, str):
                    criteria_str = criteria
                else:
                    raise ValueError(f"Invalid format for eval_material_path: {eval_material_path}")
            instruction_content.append(instruction_template[1] + criteria_str)

            
            
            if condition_image_path_list is not None:
                instruction_content.append(instruction_template[2])
                
                for cond_img_path in condition_image_path_list:
                    cond_img_dict = self._img_to_vision_dict_genai(cond_img_path)
                    instruction_content.append(cond_img_dict)
            
            if auxiliary_info is not None:
                instruction_content.append(instruction_template[3])
                instruction_content.append(auxiliary_info)
            
            instruction_content.append(instruction_template[4])
            
            
        else:
            if evaluation_type == EvaluationType.REALISM:
                if prompt is None:
                    raise ValueError("Prompt is required for REALISM evaluation")
                instruction_content.append(instruction_template[0] + prompt + instruction_template[1])
            else:
                instruction_content.append(instruction_template[0])
        
        gen_img_dict = self._img_to_vision_dict_genai(generated_image_path)
        instruction_content.append(gen_img_dict)
        
        return instruction_content
        
    def _call_gemini_api(self, content: list) -> Optional[str]:
        resp = self.model.generate_content(content, generation_config={"temperature": 0.2})
        result = resp.text.strip()
        self.logger.info(f"Gemini returned result:\n{result}")
        return result
    
    def _extract_score(self, reply: str) -> Optional[int]:
        pattern = rf"Final\s*Score\s*:\s*([{self.config.score_range[0]}-{self.config.score_range[1]}])"
        match = re.search(pattern, reply, re.I)
        return int(match.group(1)) if match else None
    
    def evaluate_single_image(self, 
                evaluation_type: EvaluationType,
                generated_image_path: str,
                prompt: Optional[str] = None,
                condition_image_path_list: Optional[str] = None,
                eval_material_path: Optional[List[str]] = None,
                auxiliary_info: Optional[Union[List[str], str]] = None) -> Optional[Dict[str, Any]]:

        if not Path(generated_image_path).exists():
            raise FileNotFoundError(f"Generated image file does not exist: {generated_image_path}")
        
        
        content = self._prepare_content(
            evaluation_type=evaluation_type,
            generated_image_path=generated_image_path,
            prompt=prompt,
            condition_image_path_list=condition_image_path_list,
            eval_material_path=eval_material_path,
            auxiliary_info=auxiliary_info
        )
        reply = self._call_gemini_api(content)
        
        if not reply:
            return None
        

        score = self._extract_score(reply)
        
        return score
            
    
    def evaluate(self, 
            evaluation_type: str,
            generated_image_path: str,
            prompt: Optional[str] = None,
            eval_material_path: Optional[List[str]] = None,
            condition_image_path_list: Optional[str] = None,
            auxiliary_info: Optional[Union[List[str], str]] = None) -> Optional[Dict[str, Any]]:

        
        if evaluation_type not in self.evaluation_type_map:
            raise ValueError(f"unsupport: {evaluation_type}. support: {list(self.evaluation_type_map.keys())}")
        
        
        
        return self.evaluate_single_image(
            evaluation_type=evaluation_type,
            generated_image_path=generated_image_path,
            prompt=prompt,
            eval_material_path=eval_material_path,
            condition_image_path_list=condition_image_path_list,
            auxiliary_info=auxiliary_info
        )

