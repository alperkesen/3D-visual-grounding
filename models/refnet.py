import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.cluster_module import ClusterModule
from models.proposal_module import ProposalModule, ClassAgnosticProposalModule
from models.proposal_module import RefineProposalModule
from models.rpg_module import RPGModule
from models.lang_module import LangModule
from models.match_module import MatchModule

class RefNet(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, 
    input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
                 use_lang_classifier=True, use_bidir=False, use_brnet=False,
                 use_self_attn=False, use_cross_attn=False, use_dgcnn=False, no_reference=False,
                 emb_size=300, hidden_size=256):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir
        self.use_brnet = use_brnet
        self.use_self_attn = use_self_attn
        self.use_cross_attn = use_cross_attn
        self.use_dgcnn = use_dgcnn
        self.no_reference = no_reference


        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote clustering
        self.cluster = ClusterModule(num_proposal, use_brnet=use_brnet)
        self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr)

        if use_brnet:
            print("Using BRNet...") # to-do
            self.proposal = ClassAgnosticProposalModule(num_class, num_heading_bin)
            self.rpg_module = RPGModule()
            self.refine = RefineProposalModule(num_class, num_heading_bin)

        if not no_reference:
            # --------- LANGUAGE ENCODING ---------
            # Encode the input descriptions into vectors
            # (including attention and language classification)
            self.lang = LangModule(num_class, use_lang_classifier, use_bidir, emb_size, hidden_size, use_self_attn=use_self_attn)

            # --------- PROPOSAL MATCHING ---------
            # Match the generated proposals and select the most confident ones
            self.match = MatchModule(num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * hidden_size, use_cross_attn=use_cross_attn, use_dgcnn=use_dgcnn)

    def forward(self, data_dict):
        """ Forward pass of the network
        Args:
            data_dict: dict
                {
                    point_clouds, 
                    lang_feat
                }
                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        # --------- HOUGH VOTING ---------
        data_dict = self.backbone_net(data_dict)
                
        # --------- HOUGH VOTING ---------
        xyz = data_dict["fp2_xyz"]
        features = data_dict["fp2_features"]
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz
        data_dict["seed_features"] = features
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        data_dict["vote_xyz"] = xyz
        data_dict["vote_features"] = features

        # --------- VOTE CLUSTER ----------------

        data_dict = self.cluster(xyz, features, data_dict)

        # --------- PROPOSAL GENERATION ---------

        data_dict = self.proposal(data_dict)

        if self.use_brnet:
            data_dict = self.rpg_module(data_dict)

            # (B, 128+128, num_proposal)
            fused_feats = torch.cat(
                (data_dict['aggregated_vote_features'].permute(0, 2, 1).contiguous(),
                 data_dict["rpg_features"]), dim=1)

            data_dict["fused_feats"] = fused_feats

            data_dict = self.refine(data_dict)


        if not self.no_reference:
            #######################################
            #                                     #
            #           LANGUAGE BRANCH           #
            #                                     #
            #######################################

            # --------- LANGUAGE ENCODING ---------
            data_dict = self.lang(data_dict)

            #######################################
            #                                     #
            #          PROPOSAL MATCHING          #
            #                                     #
            #######################################

            # --------- PROPOSAL MATCHING ---------
            data_dict = self.match(data_dict)

        return data_dict
