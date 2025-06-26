import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from gemini_evaluator import GeminiImageEvaluator, EvaluationType
from consistency_evaluator import CLIPIEvaluator, DINOEvaluator, CLIPTEvaluator
from dsg_evaluator import DSGEvaluator
from text_generation_evaluator import TextGenerationEvaluator
from tqdm import tqdm
import pprint
import argparse

TASK_METRICS = {
    'STEM_Driven_Reasoning_Generation': 'MLLM_judgement', 
    'Spatial_Reasoning_Generation': 'MLLM_judgement',
    'Situational_Reasoning_Generation': 'MLLM_judgement',
    'World_Knowledge_Anchored_Generation': 'MLLM_judgement',
    'Appearance_Compliance_Generation/Controlled_Visuals_Generation/Object_Instance_Count_Control_Generation': 'DSG',
    'Appearance_Compliance_Generation/Controlled_Visuals_Generation/Action_Cardinality_Control_Generation': 'DSG',
    'Appearance_Compliance_Generation/Controlled_Visuals_Generation/Relational_Spatial_Positioning_Composition': 'DSG',
    'Appearance_Compliance_Generation/Controlled_Visuals_Generation/Multi-Object_Generation_with_Attribute_Diversity': 'DSG',
    'Appearance_Compliance_Generation/Controlled_Visuals_Generation/Absolute_Spatial_Positioning_Generation': 'DSG',
    'Appearance_Compliance_Generation/Text_Rendering/Academic_Document_Generation': 'NED',
    'Appearance_Compliance_Generation/Text_Rendering/STEM_Formula_Generation': 'NED',
    'Appearance_Compliance_Generation/Text_Rendering/Scene_Text_Generation_in_Multiple_Languages': 'micro_f1_score',
    'Appearance_Compliance_Generation/Text_Rendering/Poster_Generation_in_Multiple_Languages': 'micro_f1_score',
    'Appearance_Compliance_Generation/Text_Rendering/Typeface_Rendering': 'MLLM_judgement',
    'Dynamics_Consistency_Generation/Fine-grained_Image_Editing': 'MLLM_judgement',
    'Dynamics_Consistency_Generation/Story_Generation': 'MLLM_judgement',
    'Dynamics_Consistency_Generation/Subject-driven_Generation': 'MLLM_judgement',
}

METRIC_REGISTRY = {
    'MLLM_judgement': GeminiImageEvaluator,
    'CLIP_I': CLIPIEvaluator,
    'CLIP_T': CLIPTEvaluator,
    'DINO': DINOEvaluator,
    'NED' : TextGenerationEvaluator,
    'micro_f1_score' : TextGenerationEvaluator,
    'DSG': DSGEvaluator,
}

TASK_CATEGORIES = [
    "Dynamics_Consistency_Generation",
    "Appearance_Compliance_Generation",
    "World_Knowledge_Anchored_Generation",
    "Spatial_Reasoning_Generation",
    "Situational_Reasoning_Generation",
    "STEM_Driven_Reasoning_Generation",
]

def set_api_keys(genai_key: str) -> None:

    import google.generativeai as genai
    genai.configure(api_key=genai_key)



class BenchmarkDataStructure:
    
    def __init__(self, benchmark_root: str, generated_image_root: str, eval_material_root: str):

        self.benchmark_root = Path(benchmark_root)
        self.generated_image_root = Path(generated_image_root)
        self.eval_material_root = Path(eval_material_root)
        self.benchmark_text_root = Path(self.benchmark_root, "text")
        self.benchmark_image_root = Path(self.benchmark_root, "image")
        self.task_to_ids = {}
        self._build_structure()
    
    def _build_structure(self):
        if not self.benchmark_text_root.exists():
            raise ValueError(f"Text directory not found: {self.benchmark_text_root}")

        for root, dirs, files in os.walk(self.benchmark_text_root):
            if dirs:
                continue

            leaf_dir = Path(root)
            prompt_files = [
                f for f in leaf_dir.glob("prompt_*.json")
                if "conclusion" not in f.name
            ]
            if not prompt_files:
                continue

            task_type = str(leaf_dir.relative_to(self.benchmark_text_root))
            ids = []
            for prompt in prompt_files:

                idx = int(prompt.stem.split("_", 1)[1])
                ids.append(idx)

            if ids:
                ids.sort()
                self.task_to_ids[task_type] = ids
                print(f"Found task '{task_type}' with {len(ids)} prompts")

    

    
    def get_file_paths(self, task_type: str, prompt_id: int) -> Dict[str, Any]:

        task_dir = self.benchmark_text_root / task_type
        prompt_path = task_dir / f"prompt_{prompt_id}.json"
        eval_material_path = self.eval_material_root / task_type / f"eval_material_{prompt_id}.json"
        generated_image_path = self.generated_image_root / task_type / f"sample_{prompt_id}" / "result_1.png"
        

        
        
        result = {
            "prompt_path": prompt_path,
            "eval_material_path": eval_material_path,
            "generated_image_path": generated_image_path,
            "conditional_image_paths": [],
            "prompt": "",
            "task_type": task_type
        }

        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
            
            result["prompt"] = prompt_data.get("text", "")
            
            image_path_list = [str(self.benchmark_image_root / path) for path in prompt_data.get("image_path", [])]
            result["conditional_image_paths"] = image_path_list

                        
    
        
        return result

    def get_task_types(self) -> List[str]:
        return list(self.task_to_ids.keys())
    
    def get_task_ids(self, task_type: str) -> List[int]:
        return self.task_to_ids.get(task_type, [])

class MetricManager:
    
    def __init__(self):
        self.metrics = {}
        self._initialize_all_metrics()
    
    def _initialize_all_metrics(self):

        all_metric_names = set()
        for metric_name in TASK_METRICS.values():
            all_metric_names.add(metric_name)
        
        for metric_name in all_metric_names:
            metric_cls = METRIC_REGISTRY.get(metric_name)
            self.metrics[metric_name] = metric_cls()
    
    def get_metric(self, metric_name: str):
        return self.metrics.get(metric_name)
    
    def find_matching_task(self, task_type: str) -> Optional[str]:

        if task_type in TASK_METRICS:

            return task_type

        task_parts = task_type.split('/')
        for i in range(len(task_parts) - 1, 0, -1):
            parent_task = '/'.join(task_parts[:i])
            if parent_task in TASK_METRICS:
                return parent_task
        
        return None
    
 
    def _load_json(self, path: str) -> Dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def evaluate_instruction_following(self, task_type: str, data: Dict) -> float:

        matching_task = self.find_matching_task(task_type)
        if matching_task is None:
            raise ValueError(f"No matching task found for: {task_type}")
        
        metric_name = TASK_METRICS[matching_task]
        
        metric_instance = self.get_metric(metric_name)
        if not metric_instance:
            print(f"Warning: Metric '{metric_name}' not initialized")
    
    
        if metric_name == 'MLLM_judgement':
            score = metric_instance.evaluate(
                evaluation_type=EvaluationType.INSTRUCTION_FOLLOWING,
                prompt=data['prompt'],
                eval_material_path=data['eval_material_path'],
                condition_image_path_list=data['conditional_image_paths'],
                generated_image_path=data['generated_image_path'],
            )
        
        elif metric_name == 'CLIP_I' or metric_name == 'DINO':
            score = metric_instance.evaluate(
                condition_image_path_list=data['conditional_image_paths'], 
                generated_image_path=data['generated_image_path']
            )
            print(f"CLIP_I score: {score}")
        
        elif metric_name == 'CLIP_T'  or metric_name == 'DSG':

            score = metric_instance.evaluate(
                generated_image_path=data['generated_image_path'],
                eval_material_path=data['eval_material_path']
            )
        elif metric_name == 'NED' or metric_name == 'micro_f1_score':
            auxiliary_info = metric_instance.evaluate(
                generated_image_path=data['generated_image_path'],
                eval_material_path=data['eval_material_path']
            )
            score = self.get_metric('MLLM_judgement').evaluate(
                evaluation_type=EvaluationType.INSTRUCTION_FOLLOWING,
                prompt=data['prompt'],
                eval_material_path=data['eval_material_path'],
                condition_image_path_list=data['conditional_image_paths'],
                generated_image_path=data['generated_image_path'],
                auxiliary_info=auxiliary_info,
            )
            
        else:
            raise ValueError(f"Unknown metric type: {metric_name}")
                
        
        return score

    def evaluate_aesthetics(self, data: Dict) -> float:
        metric_instance = self.get_metric("MLLM_judgement")
        score = metric_instance.evaluate(
            evaluation_type=EvaluationType.AESTHETIC,
            generated_image_path=data['generated_image_path'],
        )
        return score

    def evaluate_realism(self, data: Dict) -> float:
        metric_instance = self.get_metric("MLLM_judgement")
        score = metric_instance.evaluate(
            evaluation_type=EvaluationType.REALISM,
            prompt=data['prompt'],
            generated_image_path=data['generated_image_path'],
        )
        return score


class CompleteEvaluator:
    
    def __init__(self, benchmark_root: str, generated_image_root: str, eval_material_root: str):

        self.benchmark_root = Path(benchmark_root)
        self.metric_manager = MetricManager()
        self.generated_image_root = Path(generated_image_root)
        self.eval_material_root = Path(eval_material_root)
        self.cache_path = Path(benchmark_root, "..", "all_task_scores.json")
        
        
        self.benchmark_data = BenchmarkDataStructure(
            benchmark_root=self.benchmark_root,
            generated_image_root=self.generated_image_root,
            eval_material_root=self.eval_material_root
        )
        
    
    def evaluate_task(self, task_type: str, only_instruction_following: bool = False) -> Dict[str, List[Any]]:
        task_ids = self.benchmark_data.get_task_ids(task_type)
        if not task_ids:
            raise ValueError(f"No prompts found for task: {task_type}")
        
        eval_types = ["instruction_following"] if only_instruction_following else ["instruction_following", "aesthetics", "realism"]
        results = {eval_type: [] for eval_type in eval_types}
        
        
        for prompt_id in task_ids:
            data = self.benchmark_data.get_file_paths(task_type, prompt_id)
            
            for eval_type in eval_types:
                if eval_type == "instruction_following":
                    result = self.metric_manager.evaluate_instruction_following(task_type, data)
                elif eval_type == "aesthetics":
                    result = self.metric_manager.evaluate_aesthetics(data)
                elif eval_type == "realism":
                    result = self.metric_manager.evaluate_realism(data)
                results[eval_type].append(result)

        
            
            
        return results

    def _load_cache(self) -> Dict:
        """加载缓存文件"""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self, data: Dict) -> None:

        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def _validate_cache(self, cache: Dict, expected_eval_types: List[str]) -> None:

        if not cache:
            return

        expected_set = set(expected_eval_types)
        for task_type, task_result in cache.items():
            for eval_type, eval_results in task_result.items():
                if not isinstance(eval_results, list):
                    raise ValueError(
                        f"Cache for task {task_type}, eval_type {eval_type} is not a list. "
                        f"Please delete {self.cache_path} or ensure the cache format is correct."
                    )
            cached_set = set(task_result.keys())
            if cached_set != expected_set:
                raise ValueError(
                    f"Cache evaluation types {cached_set} do not match expected types {expected_set} for task {task_type}. "
                    f"Please delete {self.cache_path} or ensure the parameters are consistent."
                )
    
    def evaluate_all_tasks(self, only_instruction_following: bool = False) -> Dict[str, Dict[str, List[Any]]]:

        eval_types = ["instruction_following"] if only_instruction_following else ["instruction_following", "aesthetics", "realism"]

        all_results = self._load_cache()
        self._validate_cache(all_results, eval_types)
        
        for task_type in tqdm(self.benchmark_data.task_to_ids.keys(), desc="Evaluating all tasks"):
            cached = all_results.get(task_type)
            if cached and all(eval_type in cached for eval_type in eval_types):
                continue
            
            try:
                task_results = self.evaluate_task(task_type, only_instruction_following)
                all_results[task_type] = task_results
                self._save_cache(all_results)
            except Exception as e:
                print(f"Error evaluating task {task_type}: {e}")
                all_results[task_type] = {eval_type: [] for eval_type in eval_types}
                
        
        return all_results

    def calculate_scores_by_category(self, all_results: Dict[str, Dict[str, List]], only_instruction_following: bool = False) -> Dict[str, List[float]]:

        from collections import defaultdict
        
        eval_types = ["instruction_following"] if only_instruction_following else ["instruction_following", "aesthetics", "realism"]
        
        category_scores = defaultdict(lambda: defaultdict(list))
        
        for task_type, task_results in all_results.items():
            category = task_type.split('/')[0]
            if category not in TASK_CATEGORIES:
                continue
                
            for eval_type in eval_types:
                if eval_type not in task_results:
                    continue
                    
                for result in task_results[eval_type]:
                    if result is None:
                        continue
                    if isinstance(result, dict):
                        valid_scores = [v for v in result.values() if v is not None]
                        if valid_scores:
                            score = sum(valid_scores) / len(valid_scores)
                            category_scores[category][eval_type].append(score)
                    else:
                        category_scores[category][eval_type].append(result)
        
        final_scores = {}
        for category in TASK_CATEGORIES:
            scores_list = []
            for eval_type in eval_types:
                scores = category_scores[category][eval_type]
                avg_score = sum(scores) / len(scores) * 20 if scores else None
                scores_list.append(avg_score)
            
            final_scores[category] = scores_list
            
        final_results = {}
        for category, scores_list in final_scores.items():
            if only_instruction_following:
                final_results[category] = scores_list[0] if scores_list and scores_list[0] is not None else None
            else:
                if scores_list and len(scores_list) >= 3 and all(s is not None for s in scores_list[:3]):
                    final_results[category] = scores_list[0] * 0.8 + scores_list[1] * 0.1 + scores_list[2] * 0.1
                else:
                    final_results[category] = None
        
        return final_results



if __name__ == "__main__":
    
    """
    python cal_metrics.py \
        --genai_key_file ./my_genai_key.txt \
        --benchmark_root ../OmniGenBench/benchmark \
        --generated_image_root ../generated_images/GenerationModel \
        --eval_material_root ../OmniGenBench/eval_material \
        --only_instruction_following
    """
    
    parser = argparse.ArgumentParser(description="Evaluation system, provide the path to the txt file containing the genai_key")
    parser.add_argument("--genai_key_file", type=str, required=True, help="Path to the txt file containing the genai_key, the file should contain one line with the genai_key")
    parser.add_argument("--benchmark_root", type=str, default="../OmniGenBench/benchmark", help="Root directory of the benchmark")
    parser.add_argument("--generated_image_root", type=str, default="../generated_images/FakeModel", help="Root directory of generated images")
    parser.add_argument("--eval_material_root", type=str, default="../OmniGenBench/eval_material", help="Root directory of evaluation materials")
    parser.add_argument("--only_instruction_following", action="store_true", help="Only evaluate instruction_following scores")
    args = parser.parse_args()

    with open(args.genai_key_file, "r", encoding="utf-8") as f:
        genai_key = f.readline().strip()
    if not genai_key:
        raise ValueError("txt file does not contain genai_key")
    set_api_keys(genai_key)

    evaluator = CompleteEvaluator(args.benchmark_root, args.generated_image_root, args.eval_material_root)

    all_results = evaluator.evaluate_all_tasks(only_instruction_following=args.only_instruction_following)

    pp = pprint.PrettyPrinter(indent=4, width=100, compact=False)

    for task_type, task_results in all_results.items():
        if all(result is None for result in task_results["instruction_following"]):
            print(task_type)

    category_scores = evaluator.calculate_scores_by_category(all_results)

    print("Category scores:")
    pp.pprint(category_scores)
    

    