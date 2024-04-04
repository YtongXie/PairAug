import os
from tqdm import tqdm
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from PIL import Image
import numpy as np
import torch.nn.functional as F
import shutil
from numpy import linalg as LNG 

model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()
model.cuda()
processor = MedCLIPProcessor()

image_path_ori = 'data/MIMIC_images_ori'
image_path_p2p = 'data/IntraAug/IntraAug_images'
text_path_ori = 'data/MIMIC_reports_ori'
text_path_gpt = 'data/IntraAug/IntraAug_reports'

save_img_path = 'data/IntraAug/IntraAug_images_fliter'
os.makedirs(save_img_path, exist_ok=True)

save_text_path = 'data/IntraAug/IntraAug_reports_fliter'
os.makedirs(save_text_path, exist_ok=True)

for _, _, file in os.walk(image_path_p2p):
    final_files = file

clip_sim_pair = []
clip_sim_diff = []
clip_sim_image = []
file_names = []

for file_i in tqdm(final_files):

    txt_ori = open(os.path.join(text_path_ori, file_i[0:-17] + '.txt')).read()
    txt_gpt = open(os.path.join(text_path_gpt, file_i[0:-7] + '.txt')).read()
    
    image_ori = Image.open(os.path.join(image_path_ori, file_i[0:-17] + '.jpg')).resize([224,224])
    image_p2p = Image.open(os.path.join(image_path_p2p, file_i)).resize([224,224])

    # prepare for the demo image and texts
    inputs_ori = processor(
        text=[txt_ori],
        images=image_ori,
        return_tensors="pt",
        padding=True
        )

    inputs_gen = processor(
        text=[txt_gpt],
        images=image_p2p,
        return_tensors="pt",
        padding=True
        )

    image_features_0 = model.encode_image(inputs_ori.data['pixel_values'].cuda())         #### features from original images
    image_features_1 = model.encode_image(inputs_gen.data['pixel_values'].cuda())         #### features from synthesis images
    text_features_0 = model.encode_text(inputs_ori.data['input_ids'].cuda()[:,0:512], inputs_ori.data['attention_mask'].cuda()[:,0:512])         #### features from original reports
    text_features_1 = model.encode_text(inputs_gen.data['input_ids'].cuda()[:,0:512], inputs_gen.data['attention_mask'].cuda()[:,0:512])         #### features from synthesis reports

    clip_sim_pair.append(F.cosine_similarity(image_features_1, text_features_1).cpu().data.numpy()[0])     #### semantic alignment between synthesis images and corresponding synthesis reports
    clip_sim_diff.append(F.cosine_similarity(image_features_1 - image_features_0, text_features_1 - text_features_0).cpu().data.numpy()[0])    #### the consistency of the change between two images with the change between the corresponding two reports.
    clip_sim_image.append(F.cosine_similarity(image_features_1, image_features_0).cpu().data.numpy()[0])     #### similarity between synthesis images and original images

    file_names.append(file_i)


threshold = 0.003    ##### threshold
scalar = 1.    ##### scalar to balance the number of generated pairs

for i in range(len(file_names)):
    if (clip_sim_image[i] > ((np.array(clip_sim_image)/scalar).mean() - threshold)) and (clip_sim_diff[i] > ((np.array(clip_sim_diff)/scalar).mean() - threshold)) and (clip_sim_pair[i] > ((np.array(clip_sim_pair)/scalar).mean() - threshold)):
        if os.path.isfile(os.path.join(image_path_p2p, file_names[i])) and os.path.isfile(os.path.join(text_path_gpt, file_names[i][0:-7] + '.txt')):
            shutil.copy(os.path.join(image_path_p2p, file_names[i]), os.path.join(save_img_path, file_names[i]))
            shutil.copy(os.path.join(text_path_gpt, file_names[i][0:-7] + '.txt'), os.path.join(save_text_path, file_names[i][0:-7] + '.txt'))

print('finished!')
