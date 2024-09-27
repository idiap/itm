# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Alina Elena Baia <alina.baia@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)
print("loading model...")
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", load_in_8bit = True, device_map={"":0})
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
model.tie_weights()
model.config.text_config.pad_token_id = processor.tokenizer.pad_token_id 
print("done loading model")


if __name__ == '__main__':


  parser = argparse.ArgumentParser()
  parser.add_argument('-dataset', type=str, default='./data/dataset_train/')
  parser.add_argument("-csv", "--csv_name", help="csv file with the image ids", type=str, default ="./data/dataset_train_info.csv")

  parser.add_argument("-output", "--csv_name_output", help="csv file with the generated description", type=str, default ="image_descriptions")
  parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=1)

  args = parser.parse_args()

  if not os.path.exists("./generated_data"):
      os.makedirs("./generated_data")

  dataset_dir = args.dataset
  csv_name = args.csv_name
  csv_name_output = args.csv_name_output
  batch_size = args.batch_size

  dataset_urls = pd.read_csv(csv_name)
  images_name = list(dataset_urls["image_name"])
  # print(dataset_urls.head())

  descriptions_dict = {}
  prompt = "Describe this image as detailed as possible."

  nr_iterations = int(np.ceil(len(images_name)/batch_size))

  print("generating descriptions...")
  for i in tqdm(range(nr_iterations)):
      start_index = i*batch_size
      end_index = (i*batch_size) + batch_size

      images_to_describe = [Image.open(os.path.join(dataset_dir, img_path)).convert("RGB") for img_path in images_name[start_index:end_index]]

      inputs = processor(images=images_to_describe, text=[prompt]*batch_size, return_tensors="pt").to(device)

      outputs = model.generate(
          **inputs,
          do_sample=True, 
          num_beams=5,
          max_length=256,
          min_length=1,
          top_p=0.9, 
          repetition_penalty=1.5,
          length_penalty=1.0,
          temperature=1,
      )
      generated_text = processor.batch_decode(outputs, skip_special_tokens=True)  

      for idx,gt in enumerate(generated_text):

          img_name = images_name[start_index:end_index][idx]

          descriptions_dict[img_name] = {"image_name": images_name[start_index:end_index][idx],
                                      "description": gt }

  print("done generating descriptions.")
  descriptions_dict_df = pd.DataFrame.from_dict(descriptions_dict, orient="index").reset_index().drop(columns=['index'])
  descriptions_dict_df = pd.merge(descriptions_dict_df, dataset_urls, on="image_name")
  descriptions_dict_df.to_csv("./generated_data/{}.csv".format(csv_name_output), encoding = "utf-8", index=False)
