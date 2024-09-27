# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Alina Elena Baia <alina.baia@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import argparse
import os
import pandas as pd
import numpy as np
import torch
import random
import json

from PIL import Image
import hdbscan
from umap import UMAP
import pickle
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

from bertopic import BERTopic
from tqdm import tqdm


import imagebind.data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

import torch.nn as nn
import torch.optim as optim

from utils import classification_metrics



def cluster_images(image_embs, reducer, clusterer, cluster_size, save_dir, file_cluster_name = "clusterer_model.pkl"):
    reducer.fit(image_embs)
    reduced_embeddings = reducer.transform(image_embs)

    clusterer.fit(reduced_embeddings)
    cluster_ids = list(set(clusterer.labels_))
    cluster_ids = sorted(cluster_ids)

    predicted_clusters = list(clusterer.labels_)


    #save clusterer model
    list_pickle = open(os.path.join(save_dir, file_cluster_name), 'wb')
    pickle.dump(clusterer,list_pickle)
    list_pickle.close()
    return predicted_clusters, clusterer

#hdbscan documentation: https://hdbscan.readthedocs.io/en/latest/soft_clustering_explanation.html
def get_exemplars(cluster_id, condensed_tree):
    raw_tree = condensed_tree._raw_tree
    # Just the cluster elements of the tree, excluding singleton points
    cluster_tree = raw_tree[raw_tree['child_size'] > 1]
    # Get the leaf cluster nodes under the cluster we are considering

    leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, cluster_id)
    # Now collect up the last remaining points of each leaf cluster (the heart of the leaf)
    result = np.array([])
    for leaf in leaves:
        max_lambda = raw_tree['lambda_val'][raw_tree['parent'] == leaf].max()
        points = raw_tree['child'][(raw_tree['parent'] == leaf) &
                                   (raw_tree['lambda_val'] == max_lambda)]
        result = np.hstack((result, points))
    return result


def get_clusters_emebeddings(clusterer, image_embs, images_name):

    cluster_ids = list(set(clusterer.labels_))
    cluster_ids = sorted(cluster_ids)
    #print("cluster_ids: ", cluster_ids)

    cluster_embeddings = []
    
    #get exemplars
    exemplars_imgs = {}
    exemplars_embeds= {}
    clusters_embeddings = []
    condensed_tree = clusterer.condensed_tree_

    for i, c in enumerate(condensed_tree._select_clusters()):
      if i != -1:
        exemplars = get_exemplars(c, condensed_tree)
        exemplars_imgs[i] =  {"idxs": [int(index) for index in exemplars],
                              "image_names": [images_name[int(index)] for index in exemplars]}


    for cluster_id in cluster_ids:
      if cluster_id != -1:
        indices = np.array([index for index in exemplars_imgs[cluster_id]["idxs"]])

        if isinstance(image_embs, torch.Tensor):
            image_embs = image_embs.cpu().numpy()

        embeds = image_embs[indices]
        cluster_embeddings.append(np.mean(embeds, axis=0).reshape(1,-1))
        exemplars_embeds[cluster_id] = embeds
    
    return cluster_embeddings, exemplars_imgs, exemplars_embeds



def get_candidates(dataset_df, tags_reprs_embeddings, model_bertopic):

  cluster_topics_words = {}
  all_clusters = set(dataset_df["clusters"].tolist())

  for cluster_id in np.arange(0,len(all_clusters)-1):

    df_selected = dataset_df[dataset_df["clusters"]==cluster_id]

    selected_img_idxs = df_selected.index.values.tolist()

    images_name= list(df_selected["image_name"])

    selected_tags_reprs = df_selected["final_tags"].to_list()  

    selected_tags_reprs_embeddings = tags_reprs_embeddings[selected_img_idxs]

    topics, probs = model_bertopic.fit_transform(selected_tags_reprs, selected_tags_reprs_embeddings.cpu().numpy())


    tmp = []
    for topic_id in model_bertopic.topic_labels_.keys():
        tmp.append([x[0] for x in model_bertopic.topic_representations_[topic_id]])

    cluster_topics_words[int(cluster_id)] = Counter([item for sublist in tmp for item in sublist])

  clusters_candidates = [list(cluster_topics_words[i].keys()) for i in cluster_topics_words.keys()  ]

  return clusters_candidates

def get_clusters_descriptors(dataset_df, candidates, cluster_embeddings, encoder_model):
  clusters_descriptors = {}

  all_clusters = set(dataset_df["clusters"].tolist())

  for cluster_id in np.arange(0,len(all_clusters)-1):

      candidates_embeddings = []

      batch_size = 20
      nr_iterations = int(np.ceil(len(candidates[cluster_id])/batch_size))

      for i in tqdm(range(nr_iterations)):
        start_index = i * batch_size
        end_index = (i * batch_size) + batch_size

        inputs = {
          ModalityType.TEXT: imagebind.data.load_and_transform_text(candidates[cluster_id][start_index:end_index], device),
          }

        with torch.no_grad():
          embeddings = encoder_model(inputs)

        candidates_embeddings.extend(embeddings[ModalityType.TEXT].cpu().numpy().tolist())

      candidates_vs_cluster_embs = cosine_similarity(np.array(candidates_embeddings), cluster_embeddings[cluster_id])

      sim_matrix_torch = torch.from_numpy(candidates_vs_cluster_embs)
      top_k = 10
      topk_similar_vals ,topk_similar_idx = sim_matrix_torch.topk(top_k, dim=0, largest= True, sorted = True)

      tmp_cluster_desc = []
      for element in list(np.array(candidates[cluster_id])[topk_similar_idx]):
        tmp_cluster_desc.append(element[0])

      clusters_descriptors["c_{}".format(cluster_id)] = {"descriptor": ", ".join(tmp_cluster_desc)
                                                        }

  #print("clusters_descriptors: ", clusters_descriptors)

  return clusters_descriptors


def get_clusters_descriptors_embeds(encoder_model, clusters_descriptors, batch_size = 20):

  descriptors = []
  for key, value in clusters_descriptors.items():
    descriptors.append(value['descriptor'])

  nr_iterations = int(np.ceil(len(descriptors)/batch_size))
  descriptors_embeddings = []

  for i in tqdm(range(nr_iterations)):
    start_index = i * batch_size
    end_index = (i * batch_size) + batch_size

    inputs = {
      ModalityType.TEXT: imagebind.data.load_and_transform_text(descriptors[start_index:end_index], device),
      }

    with torch.no_grad():
      embeddings = encoder_model(inputs)
      descriptors_embeddings.extend(embeddings[ModalityType.TEXT].cpu().numpy().tolist())

  return np.array(descriptors_embeddings)


# Define the model
class Model(nn.Module):
    def __init__(self, input_dims):
        super(Model, self).__init__()
        self.layer = nn.Linear(input_dims, 2, bias = False)

    def forward(self, x):
        return self.layer(x)

def evaluate(model, input_features, gt_labels):
    model.eval()

    with torch.no_grad():
          outputs = model(input_features)
          predicted_classes = torch.argmax(outputs, dim=1)

    metrics = classification_metrics(gt_labels, predicted_classes.detach().numpy() )

    return metrics


def train(model, input_features, gt_labels, eval_features, eval_labels, optimizer, criterion,  device, use_eval, epochs = 100, batch_size = 8):
    best_model_state_dict= None
    best_f1_score = 0

    for epoch in tqdm(range(n_epochs)):

      indices = torch.randperm(input_features.size()[0])
      input_features=input_features[indices]
      gt_labels=gt_labels[indices]

      for i in range(0, len(input_features), batch_size):
        Xbatch = input_features[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = gt_labels.long()[i:i+batch_size]

        loss = criterion(y_pred, ybatch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      
      if use_eval:
        metrics = evaluate(model, eval_features, eval_labels)
        eval_f1_score = metrics["Overview"]["F1 Score"]
        #print("epoch: ", epoch, eval_f1_score)

        if eval_f1_score > best_f1_score:
            best_f1_score = eval_f1_score
            best_model_state_dict = model.state_dict()

    return model, best_model_state_dict

if __name__ == '__main__':

    #parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset_train', type=str, default='./data/dataset_train')
    parser.add_argument('-dataset_test', type=str, default='./data/dataset_test')
    parser.add_argument('-dataset_val', type=str, default='./data/dataset_val')

    parser.add_argument("-csv_train", "--csv_name_train", help="csv with the image names, labels, and tags", type=str, default ="./generated_data/final_image_tags.csv")
    parser.add_argument("-csv_test", "--csv_name_test", help="csv with the image names and labels", type=str, default ="./data/dataset_test_info.csv")
    parser.add_argument("-csv_val", "--csv_name_val", help="csv with the image names and labels", type=str, default ="./data/dataset_val_info.csv")

    parser.add_argument("-embeds_train", "--embeddings_train", help="embeddings file to use for clustering/training", type=str, default ="./generated_data/image_embeddings_train.npy")
    parser.add_argument("-embeds_test", "--embeddings_test", help="embeddings file to use for testing", type=str, default ="./generated_data/image_embeddings_test.npy")
    parser.add_argument("-embeds_val", "--embeddings_val", help="embeddings file to use for validation", type=str, default ="./generated_data/image_embeddings_val.npy")

    parser.add_argument("-tags_embeds", "--embeddings_tags_reprs_train", help="embeddings fo tags represetantion", type=str, default ="./generated_data/tags_embeddings_train.npy")

    parser.add_argument("-cls", action="store_true", help="flag for training classifier")
    parser.add_argument("-use_val", action="store_true", help="flag for using validation")

    parser.add_argument("-cs", "--cluster_size", help="minimum cluster seize", type=int, default = 30)    
    parser.add_argument("-e", "--epochs", help="how many epochs for training", type=int, default = 100)
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=8)
    parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default = 0.01 )
    parser.add_argument("-seed", "--seed", help="random seed", type=int, default = 19)

    parser.add_argument('--save_dir', type=str, default='./results')
    
    args = parser.parse_args()
    seed = args.seed

    random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
      device = torch.device('cuda:0')
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
      torch.backends.cudnn.enabled = False
    else:
      device = torch.device('cpu')

    encoder = imagebind_model.imagebind_huge(pretrained=True)
    encoder.eval()
    encoder.to(device)

    dataset_dir = args.dataset_train
    dataset_test = args.dataset_test
    dataset_val = args.dataset_val

    csv_train = args.csv_name_train
    csv_test = args.csv_name_test
    csv_val = args.csv_name_val


    train_urls = pd.read_csv(csv_train)
    test_urls = pd.read_csv(csv_test)
    val_urls = pd.read_csv(csv_val)
    print(train_urls.shape, test_urls.shape, val_urls.shape)

    tags_reprs_embeds_train = args.embeddings_tags_reprs_train

    cluster_size = args.cluster_size
    cls_flag = args.cls

    
    images_name = [os.path.join(dataset_dir, img_id) for img_id  in list(train_urls["image_name"])]
    images_name_test = [os.path.join(dataset_test, img_id) for img_id in  list(test_urls["image_name"])]
    images_name_val = [os.path.join(dataset_val, img_id) for img_id in  list(val_urls["image_name"])]

    ##########################################

    image_embs = np.load(args.embeddings_train)
    image_embs /= torch.from_numpy(image_embs).norm(dim =1, keepdim =True)

    image_embs_test = np.load(args.embeddings_test)
    image_embs_test /= torch.from_numpy(image_embs_test).norm(dim =1, keepdim =True)

    image_embs_val = np.load(args.embeddings_val)
    image_embs_val /= torch.from_numpy(image_embs_val).norm(dim =1, keepdim =True)

    print(image_embs.shape, image_embs_test.shape, image_embs_val.shape)

    #########################################

    tags_reprs_embeddings = np.load(tags_reprs_embeds_train)
    tags_reprs_embeddings /= torch.from_numpy(tags_reprs_embeddings).norm(dim =1, keepdim =True)
    print("tags_reprs_embeddings: ", tags_reprs_embeddings.shape)

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    #define reducer and clusterer models

    reducer = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state = seed, low_memory = False)

    #https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html
    clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_size, metric='euclidean', prediction_data=True)

    #define the topic model
    model_bertopic = BERTopic(language="english",  min_topic_size=5,
                    umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine', random_state = seed, low_memory = False))



    #perform clustering

    print("clustering...")
    predicted_clusters, clusterer = cluster_images(image_embs,reducer, clusterer, cluster_size, save_dir, file_cluster_name = "clusterer_model.pkl")
    #print("predicted_clusters: ", predicted_clusters)

    #get cluster embeddings (i.e. centroid-like embeddings)
    cluster_embeddings, exemplars_imgs, exemplars_embeds = get_clusters_emebeddings(clusterer, image_embs, images_name)

    # perform topic modeling inside each cluster

    train_urls["clusters"]= predicted_clusters

    train_urls.to_csv(os.path.join(save_dir, "final_image_tags_with_clusters.csv"))
  
    print("finished clustering.")

    print("getting descriptors...")
    candidates = get_candidates(train_urls, tags_reprs_embeddings, model_bertopic) 
    clusters_descriptors = get_clusters_descriptors(train_urls, candidates, cluster_embeddings, encoder)
    print("clusters_descriptors: \n", clusters_descriptors)

    with open(os.path.join(save_dir,'clusters_descriptors.json'), 'w') as fp:
      json.dump(clusters_descriptors, fp)

    print("done.")

    if cls_flag:      
      print("training classifier")
      use_val  = args.use_val

      print("use_val: ", use_val)
      descriptors_embeds =  get_clusters_descriptors_embeds(encoder, clusters_descriptors)
      descriptors_embeds /= torch.from_numpy(descriptors_embeds).norm(dim =1, keepdim =True)
      descriptors_embeds = descriptors_embeds.cpu().numpy()
      print(descriptors_embeds.shape)

      input_data_train = cosine_similarity(image_embs.cpu().numpy(), descriptors_embeds)
      input_data_train = input_data_train.astype("float32")
      X = torch.from_numpy(input_data_train)
      Y = torch.from_numpy(np.array(train_urls["label"].tolist()).astype("float32"))  


      input_data_test= cosine_similarity(image_embs_test.cpu().numpy(), descriptors_embeds)
      input_data_test = input_data_test.astype("float32")
      X_test = torch.from_numpy(input_data_test)
      Y_test_np = np.array(test_urls["label"].tolist())

      input_data_val= cosine_similarity(image_embs_val.cpu().numpy(), descriptors_embeds)
      input_data_val = input_data_val.astype("float32")
      X_val = torch.from_numpy(input_data_val)
      Y_val_np = np.array(val_urls["label"].tolist())

      #train privacy model
      n_epochs = args.epochs
      batch_size = args.batch_size
      lr = args.learning_rate


      input_dim = len(clusters_descriptors)
      model = Model(input_dim)

      criterion = nn.CrossEntropyLoss()
      optimizer = optim.Adam(model.parameters(), lr=lr)

      
      model, best_model_state_dict = train(model, X, Y, X_val, Y_val_np, optimizer, criterion, device, use_val, epochs = n_epochs, batch_size = batch_size)
      print("finished training classifier...")

      if use_val:
        model.load_state_dict(best_model_state_dict)

      metrics = evaluate(model, X_test, Y_test_np)

      for class_name, values in metrics.items():
            print(f"**{class_name}**")
            for metric_name, value in values.items():
                print(f"{metric_name}: {value:.6f}")
            print()

      torch.save(model, os.path.join(save_dir, "classifier_model.pth"))
