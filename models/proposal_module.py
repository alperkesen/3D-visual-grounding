""" 
Modified from: https://github.com/facebookresearch/votenet/blob/master/models/proposal_module.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
import lib.pointnet2.pointnet2_utils
from utils.pc_utils import rotation_3d_in_axis

class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
        super().__init__() 

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr

        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.proposal = nn.Sequential(
            nn.Conv1d(128,128,1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,128,1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,2+3+num_heading_bin*2+num_size_cluster*4+self.num_class,1)
        )

    def forward(self, data_dict):
        """
        Args:
            data_dict
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """

        features = data_dict['aggregated_vote_features'].permute(0, 2, 1).contiguous() # (batch_size, 128, num_proposal)
        # --------- PROPOSAL GENERATION ---------
        net = self.proposal(features)
        data_dict = self.decode_scores(net, data_dict, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)

        return data_dict

    def decode_scores(self, net, data_dict, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
        """
        decode the predicted parameters for the bounding boxes
        """
        net_transposed = net.transpose(2,1).contiguous() # (batch_size, 1024, ..)
        batch_size = net_transposed.shape[0]
        num_proposal = net_transposed.shape[1]

        objectness_scores = net_transposed[:,:,0:2]

        base_xyz = data_dict['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
        center = base_xyz + net_transposed[:,:,2:5] # (batch_size, num_proposal, 3)

        heading_scores = net_transposed[:,:,5:5+num_heading_bin]
        heading_residuals_normalized = net_transposed[:,:,5+num_heading_bin:5+num_heading_bin*2]
        
        size_scores = net_transposed[:,:,5+num_heading_bin*2:5+num_heading_bin*2+num_size_cluster]
        size_residuals_normalized = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster:5+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3
        
        sem_cls_scores = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster*4:] # Bxnum_proposalx10

        # store
        data_dict['objectness_scores'] = objectness_scores
        data_dict['center'] = center
        data_dict['heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
        data_dict['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
        data_dict['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin
        data_dict['size_scores'] = size_scores
        data_dict['size_residuals_normalized'] = size_residuals_normalized
        data_dict['size_residuals'] = size_residuals_normalized * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
        data_dict['sem_cls_scores'] = sem_cls_scores

        return data_dict


class ClassAgnosticProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin

        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2)
        self.proposal = nn.Sequential(
            nn.Conv1d(128,128,1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,128,1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,2+6+num_heading_bin*2,1)
        )

    def forward(self, data_dict):
        """
        Args:
            data_dict
        Returns:
            scores: (B,num_proposal,2 (sem) + 6 (dist) + num_class + num_dir_bins * 2) 
        """

        features = data_dict['aggregated_vote_features'].permute(0, 2, 1).contiguous() # (batch_size, 128, num_proposal)
        # --------- PROPOSAL GENERATION ---------
        net = self.proposal(features)
        data_dict = self.decode_scores(net, data_dict, self.num_class, self.num_heading_bin)

        return data_dict

    def decode_scores(self, net, data_dict, num_class, num_heading_bin):
        """
        decode the predicted parameters for the bounding boxes
        """
        net_transposed = net.transpose(2,1).contiguous() # (batch_size, 1024, ..)
        batch_size = net_transposed.shape[0]
        num_proposal = net_transposed.shape[1]

        objectness_scores = net_transposed[:,:,0:2] # (batch_size, num_proposal, 2)
        distance = net_transposed[:,:,2:8].exp().contiguous() # (batch_size, num_proposal, 6)

        heading_scores = net_transposed[:,:,8:8+num_heading_bin] # (batch_size, num_proposal, num_headin_bin)
        heading_residuals_normalized = net_transposed[:,:,8+num_heading_bin:8+num_heading_bin*2] # (batch_size, num_proposal, num_headin_bin)

        # store
        data_dict['objectness_scores'] = objectness_scores
        data_dict['distance'] = distance
        data_dict['heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
        data_dict['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
        data_dict['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin

        return data_dict


class RefineProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin

        # Object proposal/detection
        # Distance (6), Angle (1) + num_class

        self.proposal = nn.Sequential(
            nn.Conv1d(256,128,1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,128,1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,6+1+self.num_class,1)
        )

    def forward(self, data_dict):
        """
        Args:
            data_dict
        Returns:
            scores: (B,num_proposal,6 (distance) + 1 (heading) + num_class) 
        """

        features = data_dict['fused_feats'] # (batch_size, 256, num_proposal)
        # --------- PROPOSAL GENERATION ---------
        net = self.proposal(features)
        data_dict = self.decode_scores(net, data_dict, self.num_class, self.num_heading_bin)

        return data_dict

    def decode_scores(self, net, data_dict, num_class, num_heading_bin):
        """
        decode the predicted parameters for the bounding boxes
        """
        net_transposed = net.transpose(2,1).contiguous() # (batch_size, 1024, ..)
        batch_size = net_transposed.shape[0]
        num_proposal = net_transposed.shape[1]

        distance = data_dict['distance'] # (batch_size, num_proposal, 6)
        angle = data_dict['angle'][..., -1] # (B, N)

        angle_delta = net_transposed[..., :1].squeeze(-1) # (B, N)
        distance_delta = net_transposed[..., 1:7] # (B, N, 6)
        sem_cls_scores = net_transposed[:,:,7:] # Bxnum_proposalxnum_class

        refined_distance = distance + distance_delta # (B, N, 6)

        refined_angle = angle + angle_delta # (B, N)
        refined_angle[refined_angle > np.pi] -= 2 * np.pi

        data_dict['refined_distance'] = refined_distance  # (B, N, 6)
        data_dict['refined_angle'] = refined_angle  # (B, N)
        data_dict['sem_cls_scores'] = sem_cls_scores # (B, N, num_class)

        return data_dict
