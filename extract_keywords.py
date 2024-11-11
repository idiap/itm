# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Alina Elena Baia <alina.baia@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0


import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import LlamaForCausalLM, LlamaTokenizer
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("-csv", "--csv_name", help="csv file with the image descriptions", type=str, default ="./generated_data/image_descriptions.csv")
  parser.add_argument("-output", "--csv_name_output", help="csv file with the extracted keywords", type=str, default ="extracted_keywords.csv")
  parser.add_argument("-m", "--model_dir", help="path to LLM directory", type=str, default ="../path_to_LLM/")


  args = parser.parse_args()

  if not os.path.exists("./generated_data"):
    os.makedirs("./generated_data")

  csv_name = args.csv_name
  csv_name_output = args.csv_name_output
  llm_dir = args.model_dir


  df_description = pd.read_csv(csv_name)
  df_description["description"] = df_description["description"].apply(lambda x: x.replace("\n", " "))

  images_name = df_description["image_name"].tolist()
  descriptions = df_description["description"].tolist()

  print("loading model...")
  tokenizer = LlamaTokenizer.from_pretrained(os.path.join(llm_dir, "tokenizer.model"))
  model = LlamaForCausalLM.from_pretrained(llm_dir, torch_dtype = torch.float16, low_cpu_mem_usage = True, device_map={"":0})
  model.tie_weights()
  print("done loading model.")


  print("extracting keywords...")
  dict_keywords = {}
  for  i in tqdm(range(len(descriptions))):

    prompt = "Instruction: Extract the keywords in a comma-separated list from the following text. Input: {} Keywords:".format(descriptions[i])

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(model.device)

    generate_ids = model.generate(inputs.input_ids, max_length=512)
    results = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    dict_keywords[i]={"image_name": images_name[i],
                      "description": descriptions[i],
                      "keywords": results.split("Keywords: ")[1]

    }


  keywords_df = pd.DataFrame.from_dict(dict_keywords, orient = "index").reset_index()
  keywords_df = pd.merge(keywords_df, df_description[["image_name", "gt_label"]], on="image_name")
  keywords_df.to_csv('./generated_data/{}'.format(csv_name_output), encoding='utf-8', index = False)
  print("done extracting keywords.")
