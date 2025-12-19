import argparse
import json
import os
import requests
from pathlib import Path
import torch
from utils.myDataset import * 


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


def main():
    
    parser = argparse.ArgumentParser(description="Generate prompts for a dataset using XAI API.")
    parser.add_argument('--dataset', default='ImageNet', help='CUB/FLO/PET/FOOD/ImageNet')
    parser.add_argument('--image_root', default='D:\\PYproject\\clipI\\dataset', help='')
    parser.add_argument('--output_dir', default='D:\\PYproject\\clipI\\dataset\\prompts', help='')
    parser.add_argument('--api_key', default='', help='')
    parser.add_argument('--num_prompts', type=int, default=10, help='')
    args = parser.parse_args()

    
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
        raise ValueError(f"unknown: {args.dataset}")

  
    all_names = mydataset.all_names
  

  
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"prompts_for_{args.dataset}.json"

   
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            prompts_dict = json.load(f)
    else:
        prompts_dict = {}

 
    missing_classes = [name for name in all_names if name not in prompts_dict]


   
    for class_name in missing_classes:
        prompts = generate_prompts(class_name, args.num_prompts, args.api_key)
        prompts_dict[class_name] = prompts
      

       
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(prompts_dict, f, ensure_ascii=False, indent=4)
    
    sorted_prompts_dict = {name: prompts_dict.get(name, []) for name in all_names}

    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_prompts_dict, f, ensure_ascii=False, indent=4)

    

if __name__ == "__main__":
    main()
