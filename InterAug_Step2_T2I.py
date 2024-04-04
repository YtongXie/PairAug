import os
import argparse
from diffusers import StableDiffusionPipeline
import torch
import json

def parse_args():
    parser = argparse.ArgumentParser(description="GPT-3.5.")
    parser.add_argument(
        "--start_index",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--end_index",
        type=int,
        required=True,
    )                          # number of total argumented reports
    parser.add_argument(
        "--save_path",
        type=str,
        default='data/InterAug/InterAug_images',
        required=False,
    )
    args = parser.parse_args()
    return args

args = parse_args()

txt_path = 'data/InterAug/InterAug_reports'

model_path = "pretrained/roentgen"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

for _, _, files in os.walk(txt_path):
    final_files = files

print('finish data processing')
start_index = -1

os.makedirs(args.save_path, exist_ok=True)

times = 1   ### number of generated images for each report 

while start_index < len(final_files):
    start_index += 1
    if start_index >= args.start_index:  
        try:
            f = open(os.path.join(txt_path, final_files[start_index]), "r")
            prompt = f.read()
            name = final_files[start_index]

            for t in range(times):
                image = pipe([prompt], num_inference_steps=50, height=512, width=512, guidance_scale=4).images[0]
                image.save(os.path.join(args.save_path, name[0:-4] + '_t' + str(t) + ".png"))

            print(name)
            if start_index > args.end_index:  
                break

        except Exception as e:
            print(f"Error encountered as {e}")

print("FINISHED")
