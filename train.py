# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 13:43:37 2021

@author: Wai
"""
import numpy as np
import scipy.sparse as sparse

import tensorflow as tf
import representation_graphs
import loss_graphs
from errors import (
    ModelNotBiasedException, ModelNotFitException, ModelWithoutAttentionException, BatchNonSparseInputException,
    TfVersionException
)
from input_utils import create_tensorrec_iterator, get_dimensions_from_tensorrec_dataset
from loss_graphs import AbstractLossGraph, RMSELossGraph, WMRBLossGraph, BalancedWMRBLossGraph
from prediction_graphs import AbstractPredictionGraph, DotProductPredictionGraph
from recommendation_graphs import (
    project_biases, split_sparse_tensor_indices, bias_prediction_dense, bias_prediction_serial, rank_predictions,
    densify_sampled_item_predictions, collapse_mixture_of_tastes, predict_similar_items
)
from representation_graphs import AbstractRepresentationGraph, LinearRepresentationGraph, ReLURepresentationGraph
from representation_graphs import NormalizedLinearRepresentationGraph, FeaturePassThroughRepresentationGraph
from session_management import get_session
#from util import sample_items, calculate_batched_alpha, datasets_from_raw_input
from tensorrec import TensorRec

from eval import recall_at_k
from eval import precision_at_k
from eval import ndcg_at_k

class Train(df):
    """
    A dataset of random colored graphs.
    The task is to classify each graph with the color which occurs the most in
    its nodes.
    The graphs have `n_colors` colors, of at least `n_min` and at most `n_max`
    nodes connected with probability `p`.
    """

    def __init__(self):
        pass
        
    def interaction_masking(interactions):
        
        mask_size = len(interactions.data)
        mask = np.random.choice(a=[False, True], size=mask_size, p=[.2, .8])
        not_mask = np.invert(mask)

        train_interaction_s = sparse.coo_matrix((interactions.data[mask],
                                        (interactions.row[mask],
                                         interactions.col[mask])),
                                       shape=interactions.shape)

        test_interaction_s = sparse.coo_matrix((interactions.data[not_mask],
                                       (interactions.row[not_mask],
                                        interactions.col[not_mask])),
                                      shape=interactions.shape)

        return (train_interaction_s, test_interaction_s)
    
    def create_train_df(interaction_f):
        np.random.seed(101) # 42 81/80 ; 0,0,4000
        mask_size = len(interaction_f.data)
        np.random.choice(a=[False, True], 
                 size=mask_size, 
                 p=[.2, .8])
        
        train_interaction1, test_interaction1 = interaction_masking(interaction_f)
        # Feed the user and item features
        user_features  = user_f
        item_features = item_f
        train_interactions = train_interaction1.multiply(train_interaction1 > 0.045) #0.0225
        test_interactions = test_interaction1.multiply(test_interaction1 > 0.045)
        
        return (train_interactions,test_interactions)
    
    def train_step():
        epochs = 100 #100 number of iterations ## play around with this number to achieve an optimal learning curve
        alpha = 0.01 #0.01 
        n_components =  customer_features.shape[1] - 1 #59 ## 45 play around with this number to achieve an optimal learning curve
        verbose = True
        learning_rate = 0.011 # 0.01
        n_sampled_items = int(item_features.shape[0] * .1)
        biased = False
        k_val  = 100

        train_interactions = train_interactions1
        test_interactions = test_interactions1
#train_interactions = train_interaction_inv
#test_interactions = test_interaction_inv

## Approximation of WMRB: Learning to Rank in a Scalable Batch Training Approach .
### Interactions can be any positive values, but magnitude is ignored. Negative interactions are ignored
### http://ceur-ws.org/Vol-1905/recsys2017_poster3.pdf
#  Options: BalancedWMRBLossGraph,RMSELossGraph, RMSEDenseLossGraph, SeparationDenseLossGraph

        model = TensorRec(n_components = n_components,                 
                  #user_repr_graph = representation_graphs.FeaturePassThroughRepresentationGraph(),
                  #user_repr_graph = representation_graphs.ReLURepresentationGraph(), # 0.78
                  user_repr_graph = representation_graphs.LinearRepresentationGraph(), #0.81:38, 0.83:40
                   item_repr_graph = representation_graphs.LinearRepresentationGraph(),
                   loss_graph = loss_graphs.WMRBLossGraph(), 
                  biased=biased)

        model.fit(train_interactions, 
          user_features, 
          item_features, 
          epochs=epochs, 
          verbose=True, 
          alpha=alpha, 
          n_sampled_items=n_sampled_items,
          learning_rate=learning_rate)
        
        return(model)
    
        
    def evaluate(model):
        
        predicted_ranks = model.predict_rank(user_features=user_features,
                                     item_features=item_features)
        
        r_at_k_test = recall_at_k(predicted_ranks, test_interactions, k=3) #80
        r_at_k_train = recall_at_k(predicted_ranks, train_interactions, k=3) #80
        print("Recall at @k: Train: {:.2f} Test: {:.2f}".format(r_at_k_train.mean(), r_at_k_test.mean()))
        
        ndcg_at_k_test = ndcg_at_k(predicted_ranks, test_interactions, k=3) #80
        ndcg_at_k_train = ndcg_at_k(predicted_ranks, train_interactions, k=3) #80
        print("NDCG at @k: Train: {:.3f} Test: {:.3f}".format(ndcg_at_k_train.mean(), ndcg_at_k_test.mean()))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        