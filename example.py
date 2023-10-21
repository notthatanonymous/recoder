import os
import shutil
import sys

import numpy as np

from scipy import sparse

import pandas as pd

import multiprocessing as mp

import glog

from recoder.model import Recoder
from recoder.data import RecommendationDataset
from recoder.metrics import AveragePrecision, Recall, NDCG
from recoder.nn import DynamicAutoencoder, MatrixFactorization
from recoder.utils import dataframe_to_csr_matrix

from recoder.recommender import InferenceRecommender, SimilarityRecommender
from recoder.embedding import AnnoyEmbeddingsIndex, MemCacheEmbeddingsIndex
from recoder.metrics import RecommenderEvaluator

### change `DATA_DIR` to the location where movielens-20m dataset sits
DATA_DIR = 'data/ml-20m/'

raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'), header=0)

# binarize the data (only keep ratings >= 4)
raw_data = raw_data[raw_data['rating'] > 3.5]

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

def filter_triplets(tp, min_uc=5, min_sc=0):
    # Only keep the triplets for items which were clicked on by at least min_sc users. 
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount[itemcount['size'] >= min_sc]['movieId'].tolist())]
    
    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        # print(usercount[usercount['size'] >= min_uc]['userId'].tolist())
        tp = tp[tp['userId'].isin(usercount[usercount['size'] >= min_uc]['userId'].tolist())]
    
    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId') 
    return tp, usercount, itemcount

raw_data, user_activity, item_popularity = filter_triplets(raw_data)

sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" % 
      (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))

unique_uid = user_activity.index

np.random.seed(98765)
idx_perm = np.random.permutation(unique_uid.size)
unique_uid = unique_uid[idx_perm]

# create train/validation/test users
n_users = unique_uid.size
n_heldout_users = 100

tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
te_users = unique_uid[(n_users - n_heldout_users):]

train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]

unique_sid = pd.unique(train_plays['movieId'])

show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

pro_dir = os.path.join(DATA_DIR, 'pro_sg')


if not os.path.exists(pro_dir):
   os.makedirs(pro_dir)

def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 10 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)
    
    return data_tr, data_te

vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]

vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]

test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

def numerize(tp):
    uid = list(map(lambda x: profile2id[x], tp['userId']))
    sid = list(map(lambda x: show2id[x], tp['movieId']))
    return pd.DataFrame(data={'uid': uid, 'sid': sid, 'watched': 1}, columns=['uid', 'sid', 'watched'])

train_data = numerize(train_plays)
train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

vad_data_tr = numerize(vad_plays_tr)
vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

vad_data_te = numerize(vad_plays_te)
vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

test_data_tr = numerize(test_plays_tr)

test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

test_data_te = numerize(test_plays_te)
test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

#Training


data_dir = pro_dir + "/"
model_dir = 'models/ml-20m/'

if not os.path.exists(model_dir):
   os.makedirs(model_dir)


common_params = {
  'user_col': 'uid',
  'item_col': 'sid',
  'inter_col': 'watched',
}

glog.info('Loading Data...')

train_df = pd.read_csv(data_dir + 'train.csv')
val_tr_df = pd.read_csv(data_dir + 'validation_tr.csv')
val_te_df = pd.read_csv(data_dir + 'validation_te.csv')

# uncomment it to train with MatrixFactorization
# train_df = train_df.append(val_tr_df)

train_matrix, item_id_map, _ = dataframe_to_csr_matrix(train_df, **common_params)
val_tr_matrix, _, user_id_map = dataframe_to_csr_matrix(val_tr_df, item_id_map=item_id_map,
                                                        **common_params)
val_te_matrix, _, _ = dataframe_to_csr_matrix(val_te_df, item_id_map=item_id_map,
                                              user_id_map=user_id_map, **common_params)

train_dataset = RecommendationDataset(train_matrix)
val_tr_dataset = RecommendationDataset(val_tr_matrix, val_te_matrix)


use_cuda = False

model = DynamicAutoencoder(hidden_layers=[200], activation_type='tanh',
                           noise_prob=0.5, sparse=False)

# model = MatrixFactorization(embedding_size=200, activation_type='tanh',
#                             dropout_prob=0.5, sparse=False)

trainer = Recoder(model=model, use_cuda=use_cuda, optimizer_type='adam',
                  loss='logistic', user_based=False)

# trainer.init_from_model_file(model_dir + 'bce_ns_d_0.0_n_0.5_200_epoch_50.model')
model_checkpoint = model_dir + 'bce_ns_d_0.0_n_0.5_200'

metrics = [Recall(k=20, normalize=True), Recall(k=50, normalize=True),
           NDCG(k=100)]

try:
  trainer.train(train_dataset=train_dataset, val_dataset=val_tr_dataset,
                batch_size=500, lr=1e-3, weight_decay=2e-5,
                num_epochs=100, negative_sampling=True,
                lr_milestones=[60, 80], num_data_workers=mp.cpu_count() if use_cuda else 0,
                model_checkpoint_prefix=model_checkpoint,
                checkpoint_freq=10, eval_num_recommendations=100,
                metrics=metrics, eval_freq=10)
except (KeyboardInterrupt, SystemExit):
  trainer.save_state(model_checkpoint)
  raise

# testing

root_dir = './'
data_dir = root_dir + 'data/ml-20m/pro_sg/'
model_dir = root_dir + 'models/ml-20m/'

common_params = {
  'user_col': 'uid',
  'item_col': 'sid',
  'inter_col': 'watched',
}

method = 'inference'
model_file = model_dir + 'bce_ns_d_0.0_n_0.5_200_epoch_100.model'  # bce_ns_d_0.0_n_0.5_200_epoch_100.model
index_file = model_dir + 'bce_ns_d_0.0_n_0.5_200_epoch_100.model.index'

num_recommendations = 20

if method == 'inference':
  model = DynamicAutoencoder()
  recoder = Recoder(model)
  recoder.init_from_model_file(model_file)
  recommender = InferenceRecommender(recoder, num_recommendations)
elif method == 'similarity':
  embeddings_index = AnnoyEmbeddingsIndex()
  embeddings_index.load(index_file=index_file)
  cache_embeddings_index = MemCacheEmbeddingsIndex(embeddings_index)
  recommender = SimilarityRecommender(cache_embeddings_index, num_recommendations, scale=1, n=50)

train_df = pd.read_csv(data_dir + 'train.csv')
val_te_df = pd.read_csv(data_dir + 'test_te.csv')
val_tr_df = pd.read_csv(data_dir + 'test_tr.csv')


train_matrix, item_id_map, _ = dataframe_to_csr_matrix(train_df, **common_params)

val_tr_matrix, _, user_id_map = dataframe_to_csr_matrix(val_tr_df, item_id_map=item_id_map,
                                                        **common_params)
val_te_matrix, _, _ = dataframe_to_csr_matrix(val_te_df, item_id_map=item_id_map,
                                              user_id_map=user_id_map, **common_params)


val_tr_dataset = RecommendationDataset(val_tr_matrix, val_te_matrix)

metrics = [Recall(k=20), Recall(k=50), NDCG(k=100)]
evaluator = RecommenderEvaluator(recommender, metrics)

metrics_accumulated = evaluator.evaluate(val_tr_dataset, batch_size=500)


for metric in metrics_accumulated:
  if str(metric) == 'NDCG@100':
      print(f'Score: {np.mean(metrics_accumulated[metric])}')
