import os
from tqdm import tqdm
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
from PIL import Image
import shutil

model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()
model.cuda()
processor = MedCLIPProcessor()

image_path = 'data/InterAug/InterAug_images'
text_path = 'data/InterAug/InterAug_reports'

save_img_path = 'data/InterAug/InterAug_images_fliter'
os.makedirs(save_img_path, exist_ok=True)

save_text_path = 'data/InterAug/InterAug_reports_fliter'
os.makedirs(save_text_path, exist_ok=True)


for _, _, file in os.walk(image_path):
    final_files = file

logits = []
file_names = []
for file_i in tqdm(final_files):
    txt = open(os.path.join(text_path, file_i[0:-7] + '.txt')).read()
    image = Image.open(os.path.join(image_path, file_i))
    inputs = processor(
        text=[txt],
        images=image,
        return_tensors="pt",
        padding=True
        )
    # pass to MedCLIP model
    outputs = model(**inputs)
    logits.append(outputs['logits'][0].cpu().data.numpy())
    file_names.append(file_i)


th = 0.3
for i in range(len(file_names)):
    if logits[i] > th:
        shutil.copy(os.path.join(image_path, file_names[i]), os.path.join(save_img_path, file_names[i]))
        shutil.copy(os.path.join(text_path, file_names[i][0:-7] + '.txt'), os.path.join(save_text_path, file_names[i][0:-7] + '.txt'))


