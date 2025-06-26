
from copy import deepcopy
from PIL import Image
import openai
import base64
import io
import json
import google.generativeai as genai

# Function to encode the image
def encode_image(image_input):
    # Check if the input is a string (assuming it is a path)
    if isinstance(image_input, str):
        with open(image_input, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    # Check if the input is a PIL Image object
    elif isinstance(image_input, Image.Image):
        img_byte_arr = io.BytesIO()
        image_input.save(img_byte_arr, format=image_input.format)
        return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    else:
        raise ValueError("Invalid input: must be a file path or a PIL Image object.")

class GPT4o:
    def __init__(self, ckpt='gpt-4o'):
        
        assert openai.api_key is not None, "OpenAI API key is not set"

    def vqa(self, image, question):

        base64_image = encode_image(image)

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": f"Answer only with 'yes' or 'no'. Do not give other outputs or punctuation marks. Question: {question}"},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                    },
                ],
                }
            ],
            max_tokens=20,
            )
        
        answer = response.choices[0].message.content
        
        answer = answer.lower().strip()

        # remove punctuation marks
        answer = answer.replace(".", "").replace(",", "").replace("?", "").replace("!", "")

        return answer


class Gemini_2_5:
    def __init__(self, model="Gemini 2.5 Pro 06-05"):
        self.model = genai.GenerativeModel(model)
        
    def vqa(self, question, image_path=None):
        content = [f"Answer only with 'yes' or 'no'. Do not give other outputs or punctuation marks. Question: {question}"]
        
        # 如果有图片，添加图片
        if image_path:
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            mime = "image/jpeg" if str(image_path).endswith((".jpg", ".jpeg")) else "image/png"
            content.append({"mime_type": mime, "data": image_data})
        
        try:
            response = self.model.generate_content(content)
            return response.text.strip()
        except Exception as e:
            print(f"Error: {e}")
            return None


class DSGEvaluator:

    def __init__(self):
        self.vqa_model = Gemini_2_5()

    def get_dsg_result(self, eval_material_path: str) -> dict:
        with open(eval_material_path, "r", encoding="utf-8") as f:
            dsg_result = json.load(f)
            
        qid2tuple = dsg_result['qid2tuple']
        qid2dependency = dsg_result['qid2dependency']
        qid2question = dsg_result['qid2question']
        
        return qid2tuple, qid2dependency, qid2question
    
    
    def evaluate(self, generated_image_path, eval_material_path: str, verbose: bool = False) -> dict:
        qid2tuple, qid2dependency, qid2question = self.get_dsg_result(eval_material_path)
        
        generated_image = Image.open(generated_image_path).convert('RGB')
        
        if verbose:
            print("#"*10, "1) VQA 生成答案并打分", "#"*10)

        qid2answer = {}
        qid2scores = {}
        # 回答每个问题并初判
        for qid, question in qid2question.items():
            
            answer = self.vqa_model.vqa(generated_image, question)
            
            qid2answer[qid] = answer
            qid2scores[qid] = float(answer.lower().strip() == 'yes')

        avg_without_dep = sum(qid2scores.values()) / len(qid2scores) * 4 + 1 if qid2scores else 0.0

    
        return avg_without_dep