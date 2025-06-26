import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable, Union, Tuple
from PIL import Image
from tqdm import tqdm
import argparse




SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}



class BenchmarkRunner:
    
    def __init__(self, model_name: str, prompt_root: str, image_root: str, log_save_dir: str, result_save_dir: str):
        self.model_name = model_name
        self.prompt_root = prompt_root
        self.image_root = image_root
        self.log_save_dir = log_save_dir
        self.result_save_dir = result_save_dir
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.model_name)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            os.makedirs(self.log_save_dir, exist_ok=True)
            handler = logging.FileHandler(f"{self.log_save_dir}/{self.model_name}.log", 'a', 'utf-8')
            handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logger.addHandler(handler)
            
        return logger
    
    def _extract_prompt_text_and_images(self, prompt_dict: Dict) -> Tuple[str, List[str], str]:

        text = prompt_dict.get('text', '')
        image_paths = [os.path.join(self.image_root, img_path) for img_path in prompt_dict.get('image', [])]
        prompt_id = prompt_dict.get('prompt_id', 'unknown')
        task_rel_path = self._get_task_rel_path(prompt_dict, prompt_id)
        prompt_id_num = prompt_id.split("/")[-1]
        save_dir = os.path.join(self.result_save_dir, self.model_name, task_rel_path, f"sample_{prompt_id_num}")
        return text, image_paths, save_dir


    def run(self, generate_func: Callable[[str, List[str]], List[Union[Image.Image, str]]]) -> Dict[str, int]:

        prompt_files = self._get_all_prompt_files()
        print(f"\n📌 total {len(prompt_files)} prompts")

        total_count = 0
        success_count = 0

        for prompt_file in tqdm(prompt_files, desc="benchmark testing", ncols=100):
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_dict = json.load(f)
            prompt_id = prompt_dict.get('prompt_id', os.path.relpath(prompt_file, self.prompt_root))
            prompt_dict['prompt_id'] = prompt_id

            text, image_paths, save_dir = self._extract_prompt_text_and_images(prompt_dict)

            if self._is_prompt_finished(Path(save_dir)):
                self.logger.info(f"✓ skip finished: {prompt_id}")
                success_count += 1
                total_count += 1
                continue

            try:
                generated_images = generate_func(text, image_paths)
                if generated_images:
                    self._save_images(generated_images, save_dir)
                    success_count += 1
                    self.logger.info(f"  ✓ {prompt_id}")
                else:
                    self.logger.info(f"  ✗ {prompt_id} - no images generated")
            except Exception as e:
                self.logger.info(f"  ✗ {prompt_id} - error: {str(e)}")
            total_count += 1

        print(f"\n✅ done! total: {total_count}, success: {success_count}")

        return {
            'total': total_count,
            'success': success_count,
            'failed': total_count - success_count
        }

    def _get_all_prompt_files(self) -> List[Path]:
        prompt_files = []
        root_path = Path(self.prompt_root)

        if root_path.exists():
            for json_file in root_path.rglob("*"):
                if json_file.is_file() and json_file.name.startswith("prompt_") and json_file.name.endswith(".json") and "conclusion" not in json_file.name:
                    prompt_files.append(json_file.resolve())
        return sorted(prompt_files)

    def _get_task_rel_path(self, prompt_dict: Dict, prompt_id: str) -> str:
        task_rel_path = prompt_dict.get('task_rel_path', None)
        if task_rel_path is None:
            if "/" in prompt_id:
                task_rel_path = "/".join(prompt_id.split("/")[:-1])
            else:
                task_rel_path = ""
        return task_rel_path

    def _is_prompt_finished(self, save_dir: Path) -> bool:
        if not save_dir.exists():
            return False
            
        for i in range(1, 10):
            for ext in SUPPORTED_EXTENSIONS:
                result_file = save_dir / f"result_{i}{ext}"
                if result_file.exists():
                    return True
        return False

    def _save_images(self, images: list, save_dir: str):
        print(save_dir)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            if isinstance(img, Image.Image):
                img.save(save_dir / f"result_{i+1}.png")
            else:
                Image.fromarray(img).save(save_dir / f"result_{i+1}.png")




class GenerationModel:
    def __init__(self):
        self.model = None
    def generate(self, text: str, image_paths: List[str]) -> List[Image.Image]:
        pass
    
    

if __name__ == "__main__":
    """
    python generate_image.py \
        --prompt_root /your/path/to/benchmark/text \
        --image_root /your/path/to/benchmark/image \
        --log_save_dir ./log \
        --result_save_dir ./generated_images
    """
    
    
    parser = argparse.ArgumentParser(description="OmniGenBench generator runner script")
    parser.add_argument("--prompt_root", type=str, default="/your/path/to/benchmark/text", help="Root directory for prompts")
    parser.add_argument("--image_root", type=str, default="/your/path/to/benchmark/image", help="Root directory for conditional images")
    parser.add_argument("--log_save_dir", type=str, default="./log", help="Directory to save logs")
    parser.add_argument("--result_save_dir", type=str, default="./generated_images", help="Directory to save generated images")
    args = parser.parse_args()

    fake_model = GenerationModel()
    runner = BenchmarkRunner(
        model_name=f"{fake_model.__class__.__name__}",
        prompt_root=args.prompt_root,
        image_root=args.image_root,
        log_save_dir=args.log_save_dir,
        result_save_dir=args.result_save_dir
    )
    results = runner.run(
        generate_func=fake_model.generate
    )