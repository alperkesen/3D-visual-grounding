""" 
Modified from: https://github.com/facebookresearch/votenet/blob/master/models/proposal_module.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import pointnet2_utils

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
import lib.pointnet2.pointnet2_utils
from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes

class ClusterModule(nn.Module):
    def __init__(self, num_proposal, seed_feat_dim=256, use_brnet=False):
        super().__init__() 

        self.num_proposal = num_proposal
        self.seed_feat_dim = seed_feat_dim
        self.sample_mod = "vote" if use_brnet == False else "seed"

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
            npoint=self.num_proposal,
            radius=0.3,
            nsample=16,
            mlp=[self.seed_feat_dim, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True
        )


    def forward(self, xyz, features, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            aggregated_vote_xyz: (B, M, 3)
            aggregated_vote_features: (B, M, 128)
            aggregated_vote_inds: (B, M, )
        """

        # Farthest point sampling (FPS) on votes
        if self.sample_mod == "vote":
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
        else:
            sample_inds = pointnet2_utils.furthest_point_sample(data_dict["seed_xyz"],
                                                                self.num_proposal)
            xyz, features, fps_inds = self.vote_aggregation(xyz, features, sample_inds)
        
        sample_inds = fps_inds

        data_dict['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        data_dict['aggregated_vote_features'] = features.permute(0, 2, 1).contiguous() # (batch_size, num_proposal, 128)
        data_dict['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        return data_dict
