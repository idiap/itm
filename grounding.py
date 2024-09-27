# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Alina Elena Baia <alina.baia@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy

from PIL import Image
import supervision as sv
import ast
from collections import Counter


from groundingdino.util.inference import load_model, load_image, predict, annotate
HOME = "./"
CONFIG_PATH = os.path.join(HOME, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))


WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))

model = load_model(CONFIG_PATH, WEIGHTS_PATH)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-dataset', type=str, default='../data/dataset_train/')
  parser.add_argument("-csv", "--csv_name", help="csv file with the image descriptions", type=str, default ="../generated_data/extracted_keywords.csv")
  parser.add_argument("-output", "--csv_name_output", help="csv file with the extracted keywords", type=str, default ="final_image_tags.csv")


  args = parser.parse_args()


  dataset_dir = args.dataset
  csv_name = args.csv_name
  csv_name_output = args.csv_name_output

  df = pd.read_csv(csv_name)
  df["final_tags"] = np.NaN
  df['final_tags'] = df['final_tags'].astype('object')

  images_name = list(df["image_name"])

  if not os.path.exists("../generated_data"):
    os.makedirs("../generated_data")
    
  for idx in tqdm(range(len(images_name))):
 
    img = images_name[idx]
    IMAGE_NAME = img
    IMAGE_PATH = os.path.join(dataset_dir, img)

    words = df.loc[df["image_name"] == IMAGE_NAME, "keywords"].item()

    words = words.strip(".")
    words = words.split(", ")

    tmp_phrases = []
    count_boxes = 0
    for word in words:

      TEXT_PROMPT = word
  
      BOX_TRESHOLD =0.35
      TEXT_TRESHOLD =0.25

      image_source, image = load_image(IMAGE_PATH)

      boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
      )

      count_boxes = count_boxes + len(boxes)
      if len(phrases) !=0:
        for p in phrases:
          tmp_phrases.append(p)

    final_tags = ", ".join(list(Counter(tmp_phrases).keys()))
    df.loc[df["image_name"] == img, "final_tags"] = final_tags

  df.to_csv('../generated_data/{}'.format(csv_name_output), encoding='utf-8', index = False)
  print(df.head())
