import argparse
import json
import os
import requests
from pathlib import Path
import torch
from utils.myDataset import *  # 假设你已经导入了数据集相关的类

# 调用 API 生成提示词
def generate_prompts(class_name, num_prompts, api_key):
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a creative assistant helping to generate detailed prompts for machine learning tasks."
            },
            {
                "role": "user",
                "content": f"Write {num_prompts} detailed descriptions of {class_name}, including its unique features, using the format {class_name}, which has..., The description I need is for image generation, so the description you give must be a clear visual feature that can help the generator understand the content to the greatest extent and reduce ambiguity. And you need to estimate your own adjectives to ensure that the features produced by your adjectives in the generated image are not distorted or exaggerated. The entry format and length give you an example: abyssinian, which has a sleek, golden-tan coat, slender frame, tall ears, bright amber eyes, and long, agile legs.. The entry length should remain roughly the same, and each output should only have the description I need, and no other content.Generate only the description of the newly updated class each time to avoid repeating the description of the class that has already been generated above.Try to generate ten from different angles, don’t make them too similar and the text shouldn’t be too long.The generated text should not contain garbled characters or **, which may affect the function of the transcription json file."
            }
        ],
        "model": "grok-beta",
        "stream": False,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        response_json = response.json()
        return response_json["choices"][0]["message"]["content"].split('\n')
    else:
        print(f"API error for class '{class_name}': {response.status_code} - {response.text}")
        return [f"Error generating prompt for {class_name}"]

# 主函数
def main():
    # 设置解析器
    parser = argparse.ArgumentParser(description="Generate prompts for a dataset using XAI API.")
    parser.add_argument('--dataset', default='ImageNet', help='数据集：CUB/FLO/PET/FOOD/ImageNet')
    parser.add_argument('--image_root', default='D:\\PYproject\\clipI\\dataset', help='数据集根目录路径')
    parser.add_argument('--output_dir', default='D:\\PYproject\\clipI\\dataset\\prompts', help='保存生成提示词的目录')
    parser.add_argument('--api_key', default='xai-dyz0BBSyZyAyalw8tRVYBh0kGUGo2DW9P3enDGd9Y9jSdO2aiQnhiHSDfXgO0fS3WVLJ5a9uDNX7MCGa', help='XAI API 密钥')
    parser.add_argument('--num_prompts', type=int, default=10, help='每个类生成的提示词数量')
    args = parser.parse_args()

    # 根据选择的数据集加载
    if args.dataset == "CUB":
        mydataset = CUBDataset(args)
    elif args.dataset == "FLO":
        mydataset = FLODataset(args)
    elif args.dataset == "PET":
        mydataset = PETDataset(args)
    elif args.dataset == "FOOD":
        mydataset = FOODDataset(args)
    elif args.dataset == "ImageNet":
        mydataset = ImageNetDataset(args)
    else:
        raise ValueError(f"未知数据集: {args.dataset}")

    # 提取类名
    all_names = mydataset.all_names
    print(f"从 {args.dataset} 数据集中加载了 {len(all_names)} 个类。")

    # 输出文件路径
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"prompts_for_{args.dataset}.json"

    # 读取已有的提示词文件
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            prompts_dict = json.load(f)
    else:
        prompts_dict = {}

    # 检查缺失类
    missing_classes = [name for name in all_names if name not in prompts_dict]
    print(f"需要补充提示词的类：{len(missing_classes)} 个")

    # 补充缺失的提示词
    for class_name in missing_classes:
        prompts = generate_prompts(class_name, args.num_prompts, args.api_key)
        prompts_dict[class_name] = prompts
        print(f"为类 '{class_name}' 补充了提示词")

        # 立刻写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(prompts_dict, f, ensure_ascii=False, indent=4)
    
    sorted_prompts_dict = {name: prompts_dict.get(name, []) for name in all_names}

    # 保存排序后的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_prompts_dict, f, ensure_ascii=False, indent=4)

    print(f"提示词已保存到 {output_file}，并按类名排序完成。")

if __name__ == "__main__":
    main()
