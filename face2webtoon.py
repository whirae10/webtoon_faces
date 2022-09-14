import os
import cv2
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from tqdm import tqdm
import pickle
from model import Encoder, Generator
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.image

device = 'cuda:0'
image_size = 256
torch.set_grad_enabled(False)

ae_model_path = './checkpoint/last.pt'  # './checkpoint/002000.pt'

tmp_df = pd.read_pickle('./vector.pkl')
module = hub.load("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2")


def load_image(path, size):
    image = image2tensor(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

    w, h = image.shape[-2:]
    if w != h:
        crop_size = min(w, h)
        left = (w - crop_size) // 2
        right = left + crop_size
        top = (h - crop_size) // 2
        bottom = top + crop_size
        image = image[:, :, left:right, top:bottom]

    if image.shape[-1] != size:
        image = torch.nn.functional.interpolate(image, (size, size), mode="bilinear", align_corners=True)

    return image


def image2tensor(image):
    image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.
    return (image - 0.5) / 0.5


def get_cos_sim(image, num):
    image = np.array([image])
    output = np.array(module(image))
    cos_sim_array = []

    for i in range(len(tmp_df['output'])):
        cos_sim_array.append(cosine_similarity(output, tmp_df['output'][i]))

    tmp_df['cos_sim'] = cos_sim_array
    # tmp_df_cos = tmp_df.sort_values(by='cos_sim', ascending=False)
    # top3 필요시에는 이 코드 실행 tmp_df_cos = tmp_df_cos[:3]
    tmp_df_cos_title = tmp_df.groupby(['title']).sort_values(by='cos_sim', ascending=False)
    tmp_df_cos_char = tmp_df.groupby(['title', 'charc']).sort_values(by='cos_sim', ascending=False).reset_index()

    result = []
    for idx, row in tmp_df_cos_title[:num].iterrows():
        best_char = tmp_df_cos_char[tmp_df_cos_char.title == str(row['title'])].iloc[0]['charc']
        result.append([row['title'], row['cos_sim'], best_char])

    return result


encoder = Encoder(32).to(device)
generator = Generator(32).to(device)

ckpt = torch.load(ae_model_path, map_location=device)
encoder.load_state_dict(ckpt["e_ema"])
generator.load_state_dict(ckpt["g_ema"])

encoder.eval()
generator.eval()

print(f'[SwapAE model loaded] {ae_model_path}')


def tensor2image(tensor):
    tensor = tensor.clamp(-1., 1.).detach().squeeze().permute(1, 2, 0).cpu().numpy()
    return tensor * 0.5 + 0.5


def run(image):
    inputs = load_image(image, image_size).to(device)
    input_structure, input_texture = encoder(inputs)
    fake_img = generator(input_structure, input_texture)
    fake_img = tensor2image(fake_img)
    fake_img = matplotlib.image.imsave(r'D:\cp1\cp1_webtoonize\static\ml_img\002.png', fake_img)
    return fake_img
