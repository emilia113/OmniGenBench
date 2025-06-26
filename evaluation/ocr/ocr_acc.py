import os
import re
import copy
from paddleocr import PaddleOCR

# get the keywords from the gt text
def get_gt_key_words(text: str):
    words = []
    text = text
    matches = re.findall(r"'(.*?)'", text) # find the keywords enclosed by ''
    if matches:
        for match in matches:
            words.extend(match.split())
   
    return words

def ocr_images(img_path, lang='fr'):
    """
    输入一系列图像路径，返回每张图片识别到的字符串，按顺序组成列表。
    :param img_paths: 图像文件路径列表
    :param lang: 语言，默认为法语（fr）
    :return: 每张图片识别到的字符串列表
    """
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang=lang)
    pred = ocr.predict(input=img_path)[0]["rec_texts"]
    return pred


def get_recall(pred, gt):
    """
    pred: list of predicted texts
    gt: list of ground truth texts
    """

    pred = [p.strip().lower() for p in pred]
    gt = [g.strip().lower() for g in gt]

    gt_orig = copy.deepcopy(gt)

    gt_length = len(gt)

    for p in pred:
        if p in gt_orig:
            gt_orig.remove(p)

    r = (gt_length - len(gt_orig)) / (gt_length + 1e-8)

    return r


files = os.listdir('/path/to/MaskTextSpotterV3/tools/ocr_result')
print(len(files))

