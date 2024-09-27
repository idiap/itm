# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Alina Elena Baia <alina.baia@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import os
import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

import imagebind.data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("-csv", "--csv_name", help="csv file with the image tags", type=str, default ="./generated_data/final_image_tags.csv")
  parser.add_argument("-output", "--output_file_name", help="file with the generated embeddings", type=str, default ="tags_embeddings")
  parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=20)
  args = parser.parse_args()

  if not os.path.exists("./generated_data"):
    os.makedirs("./generated_data")

  csv_name = args.csv_name
  output = args.output_file_name
  batch_size = args.batch_size


  df_tags = pd.read_csv(csv_name)
  images_name = df_tags["image_name"].tolist()
  tags = df_tags["final_tags"].tolist()

  nr_iterations = int(np.ceil(len(tags)/batch_size))

  tags_embeddings = []
  for i in tqdm(range(nr_iterations)):
    start_index = i * batch_size
    end_index = (i * batch_size) + batch_size

    inputs = {
      ModalityType.TEXT: imagebind.data.load_and_transform_text(tags[start_index:end_index], device),
      }

    with torch.no_grad():
      embeddings = model(inputs)

    tags_embeddings.extend(embeddings[ModalityType.TEXT].cpu().numpy().tolist())


  #print(np.array(tags_embeddings).shape)
  np.save("./generated_data/{}.npy".format(output), np.array(tags_embeddings))
