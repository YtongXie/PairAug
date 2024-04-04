import os
import openai
import tiktoken
import json
import time
import argparse

openai.api_key = "xxxxxxx"   # input your openai api key


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chatgpt_completion(model_new="gpt",sys_msg='helper',prompt_new="Hello_World", temperature_new=0.05, top_p_new=1, n_new=1, max_tokens_new=1):
    Chat_Completion = openai.ChatCompletion.create(
        model=model_new,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt_new}
        ],
        temperature=temperature_new,
        top_p=top_p_new,
        n=n_new,
        max_tokens=max_tokens_new,
        presence_penalty=0,
        frequency_penalty=0
    )
    return Chat_Completion


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
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default='data/IntraAug_reports',
        required=False,
    )
    args = parser.parse_args()
    return args


args = parse_args()

os.makedirs(args.save_path, exist_ok=True)

final_files = []
with open('metadata_train.jsonl','r') as json_file:     # metadata_snets.jsonl == mimic cxr reports
    json_list = list(json_file)
    for json_str in json_list:
        result = json.loads(json_str)
        final_files.append(result)

print('finish data processing')
start_index = -1
pred = {}
while start_index < len(final_files):
    start_index += 1
    if start_index > args.start_index:  
        try:
            prompt = final_files[start_index]['text']
            if len(prompt.split(' ')) < 20:
                continue 

            guide = "Following is an original chest X-ray report. Generate one reasonable augmentation that is limited to 50 words while conveying patient's changes in symptom than the original report." #### use

            completion = chatgpt_completion(model_new="gpt-3.5-turbo",prompt_new=prompt,sys_msg= guide,max_tokens_new=300,temperature_new=0.5)
            rewrite_finding = completion.choices[0].message.content
            
            pred.update({final_files[start_index]['file_name']: rewrite_finding})
            file_name  = os.path.join(args.save_path, final_files[start_index]['file_name'][0:-4] + "_augmented.txt")
            with open(file_name, "w") as f:
                f.write(rewrite_finding)
            print(start_index)

            if start_index > args.end_index: 
                break

        except Exception as e:
            print(f"Error encountered as {e}")
            print("Wait for 30s before retrying.")
            time.sleep(30)

with open("IntraAug_reports.json","w") as filehandle:
    json.dump(pred, filehandle)


print("FINISHED")

   










