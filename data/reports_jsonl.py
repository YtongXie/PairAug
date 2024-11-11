import os
import re
import pandas as pd
from tqdm import tqdm
import json
from nltk.tokenize import RegexpTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

split = 'train'
MIMIC_CXR_DATA_DIR = '2019.MIMIC-CXR-JPG/2.0.0'     ### Original MIMIC-CXR-JPG data folder
MIMIC_CXR_MASTER_CSV = os.path.join(MIMIC_CXR_DATA_DIR, "master.csv")
#### The ‘master.csv’ can be downloaded from this link: https://drive.google.com/file/d/1-ZW_BsoPH1bzsNo-EelJIhgm6Wk6jaFq/view?usp=sharing
MIMIC_CXR_VIEW_COL = "ViewPosition"
MIMIC_CXR_PATH_COL = "Path"
MIMIC_CXR_SPLIT_COL = "split"

df = pd.read_csv(MIMIC_CXR_MASTER_CSV)
df = df[df["ViewPosition"].isin(["PA", "AP"])]
df[MIMIC_CXR_PATH_COL] = df[MIMIC_CXR_PATH_COL].apply(
    lambda x: os.path.join(MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:])))

df = df[df[MIMIC_CXR_SPLIT_COL] == split]
df.reset_index(drop=True, inplace=True)

output_file = 'metadata_' + split + '.jsonl'

with open(output_file, 'w') as outfile:

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        # pick impression, findings, last_paragraph
        captions = ""
        captions += row["impression"]
        captions += " "
        captions += row["findings"]

        # use space instead of newline
        captions = captions.replace("\n", " ")

        # split sentences
        splitter = re.compile("[0-9]+\.")
        captions = splitter.split(captions)
        captions = [point.split(".") for point in captions]
        captions = [sent for point in captions for sent in point]

        cnt = 0
        study_sent = []
        # create tokens from captions
        for cap in captions:
            if len(cap) == 0:
                continue

            cap = cap.replace("\ufffd\ufffd", " ")

            tokenizer = RegexpTokenizer(r"\w+")
            tokens = tokenizer.tokenize(cap.lower())
            if len(tokens) <= 1:
                continue

            # filter tokens for current sentence
            included_tokens = []
            for t in tokens:
                t = t.encode("ascii", "ignore").decode("ascii")
                if len(t) > 0:
                    included_tokens.append(t)

            if len(included_tokens) > 0:
                study_sent.append(" ".join(included_tokens))

            cnt += len(included_tokens)

        if cnt >= 3:
            series_sents = list(filter(lambda x: x != "", study_sent))
            sent = ",".join(series_sents)


            metadata_entry = {
                'file_name': row['Path'][row['Path'].find('files')+6::].replace('/','_'),  
                'text': sent,
            }

            outfile.write(json.dumps(metadata_entry) + '\n')
