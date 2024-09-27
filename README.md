#  Image-guided topic modeling for interpretable privacy classification
This repository contains the code for the paper _Image-guided topic modeling for interpretable privacy classification_  accepted at [eXCV workshop](https://excv-workshop.github.io/) at ECCV 2024 (pdf version available [here](https://publidiap.idiap.ch/attachments/papers/2024/Baia_ECCVW_2024.pdf)).


## Installation

Clone repository:

```git clone https://github.com/idiap/itm.git```

Install [conda](https://docs.conda.io/en/latest/) and [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html). Then create a new environment:

```
mamba env create -f itm_environment.yml   # create the environment
mamba activate itm                    # activate the environment
```

Go to the project's directory and clone the [ImageBind](https://github.com/facebookresearch/ImageBind) repository. Please note that the ImageBind code and model weights are released under the  CC-BY-NC 4.0 [license](https://github.com/facebookresearch/ImageBind?tab=License-1-ov-file#readme).  The ImageBind model is used to generate image and text embeddings:

```
git clone https://github.com/facebookresearch/ImageBind.git
mv ImageBind/imagebind/ imagebind
mv ImageBind/* /imagebind
```

For keywords extraction we use Vicuna-7b model. Follow the instruction [here](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md) on how to obtain the Vicuna weights. 

Get GroundingDino to perform grounding of the keywords. Please refer to the official [page](https://github.com/IDEA-Research/GroundingDINO) for more details.

```
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e . --no-build-isolation

mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

## Usage

### Generate image descriptions
To generate image descriptions we use Huggingface [InstructBLIP](https://huggingface.co/docs/transformers/main/en/model_doc/instructblip). For more details please refer to the [documentation](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip). Please note  that "The model is intended and licensed for research use only. InstructBLIP w/ Vicuna models are restricted to uses that follow the license agreement of LLaMA and Vicuna. The models have been trained on the LLaVA dataset which is CC BY NC 4.0 (allowing only non-commercial use)" (reference [Usage and License](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)).

 The ```generate_descriptions.py``` script will generate image descriptions and save them in a .csv file inside the folder ```./generated_data```.

``` 
python generate_descriptions.py [-dataset DATASET_DIRECTORY] [-csv DATASET_INFO] [-output CVS_FILE_NAME] [-b BATCH_SIZE]

arguments:
    -dataset DATASET_DIRECTORY: image dataset folder
    -csv DATASET_INFO: path to .csv file having two columns:        
        - image_name - containing the image ids
        - label - containing the ground truth labels 
    -output CSV_FILE_NAME: name of the .csv file where the generated image descriptions are saved
    -b BATCH_SIZE: batch size to use for inference
```

### Extract keywords 
We use Vicuna-7b to extract keywords. Please note that "Vicuna is based on Llama 2 and should be used under Llama's [model license](https://github.com/meta-llama/llama/blob/main/LICENSE)" ([reference](https://github.com/lm-sys/FastChat?tab=readme-ov-file)).

The ```extract_keywords.py``` script will extract keywords from image descriptions obtained in the previous step. The keywords are saved in a .csv file inside the folder ```./generated_data```.

```
python extract_keywords.py [-csv CVS_INPUT_FILE] [-output CSV_OUTPUT_FILE] [-m MODEL_DIR]

arguments:
    -csv CVS_INPUT_FILE: path to .csv file with image descriptions
    -output CSV_OUTPUT_FILE: name of the .csv file to save the extracted keywords
    -m MODEL_DIR: path to LLM (i.e. Vicuna) directory containing the model and tokenizer.
```

### Ground keywords
We use GroundingDINO to ground keywords to the image and obtain the final image tags. The ```grounding.py``` script perform grounding and the grounded keywords are saved in a .csv file inside the folder ```./generated_data```.

```
cp grounding.py /GroundingDINO
cd GroundingDINO/
python grounding.py [-dataset DATASET_DIRECTORY] [-csv CVS_INPUT_FILE] [-output CSV_OUTPUT_FILE]
cd ..
arguments:
    -dataset DATASET_DIRECTORY: image dataset folder
    -csv CVS_INPUT_FILE: path to .csv file with the extracted keywords
    -output CSV_OUTPUT_FILE: name of the .csv file to save the grounded keywords (i.e. final image tags)
```    
### Generate embeddings

Use ```get_image_embeddings.py``` script to generate image embeddings. The embeddings are saved in the folder ```./generated_data```.

```
python get_image_embeddings.py [-dataset DATASET_DIRECTORY] [-csv DATASET_INFO] [-output OUTPUT_FILE] [-b BATCH_SIZE]

arguments:
    -dataset DATASET_DIRECTORY: image dataset folder
    -csv DATASET_INFO: path to .csv file having two columns:        
        - image_name - containing the image ids
        - gt_label - containing the ground truth labels 
    -output OUTPUT_FILE: file name to save the embeddings
    -b BATCH_SIZE: batch size to use for inference
```
Use ```get_tags_embeddings.py``` script to generate tags embeddings. The embeddings are saved in the folder ```./generated_data```.

```
python get_tags_embeddings.py [-csv CVS_INPUT_FILE] [-output OUTPUT_FILE] [-b BATCH_SIZE]

arguments:
    -csv CVS_INPUT_FILE: path to .csv file with the image tags
    -output OUTPUT_FILE: file name to save the embeddings
    -b BATCH_SIZE: batch size to use for inference
```

### Generate content descriptors and train classifier
The ```main.py``` script will generate content descriptors and uses them to train a classifier. The descriptors are saved in a .json file inside the folder ```./results```.

``` 
python main.py [-dataset_train DATASET_DIRECTORY] [-dataset_test DATASET_DIRECTORY] 
               [-dataset_val DATASET_DIRECTORY] [-csv_train CVS_INPUT_FILE] [-csv_test CVS_INPUT_FILE] 
               [-csv_val CVS_INPUT_FILE] [-embeds_train IMAGE_EMBEDDINGS_FILE] 
               [-embeds_test IMAGE_EMBEDDINGS_FILE] [-embeds_val IMAGE_EMBEDDINGS_FILE] 
               [-tags_embeds TAGS_EMBEDDINGS_FILE] [-cls] [-use_val]  [-cs CLUSTER_SIZE] 
               [-e EPOCHS] [-b BATCH_SIZE] [-lr LEARNING_RATE] [-seed SEED] [-save_dir DIRECTORY] 

arguments:
    -dataset_train DATASET_DIRECTORY: image train dataset folder
    -dataset_test DATASET_DIRECTORY: image test dataset folder
    -dataset_val DATASET_DIRECTORY: image validation dataset folder
    -csv_train CVS_INPUT_FILE: .csv file with the train image names, tags and labels
    -csv_test CVS_INPUT_FILE: .csv file with the test image names labels
    -csv_val CVS_INPUT_FILE:  .csv file with the validation image names labels
    -embeds_train IMAGE_EMBEDDINGS_FILE: file with train image embeddings
    -embeds_test IMAGE_EMBEDDINGS_FILE: file with test image embeddings
    -embeds_val IMAGE_EMBEDDINGS_FILE: file with validation image embeddings
    -tags_embeds TAGS_EMBEDDINGS_FILE: file with train tags embeddings
    -cls: flag for training classifier
    -use_val: flag for using validation
    -cs CLUSTER_SIZE: minimum cluster size
    -e EPOCHS: number of epochs
    -b BATCH_SIZE: batch size 
    -lr LEARNING_RATE: learning rate
    -seed SEED: random seed
    -save_dir DIRECTORY: folder to save the descriptors

```
