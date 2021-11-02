"""
PointGroup/One-thing-one-click
Written by Li Jiang
Modified by Zhengzhe Liu
"""

import torch
import torch.nn as nn
import spconv
from spconv.modules import SparseModule
import functools
from collections import OrderedDict
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
print(os.path.join(os.path.dirname(__file__), "../.."))

from pointgroup_ops.functions import pointgroup_ops
# from util import utils


class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            # norm_fn(in_channels),
            nn.BatchNorm1d(in_channels, eps=1e-4, momentum=0.1),
            nn.ReLU(),
            spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
            # norm_fn(out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-4, momentum=0.1),
            nn.ReLU(),
            spconv.SubMConv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
        )

        self.indice_key = indice_key

    def forward(self, input):
        identity = spconv.SparseConvTensor(
            input.features, input.indices, input.spatial_shape, input.batch_size
        )
        # print(input)
        # print(self.indice_key)
        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            # nn.BatchNorm1d(in_channels, eps=1e-4, momentum=0.1),
            nn.ReLU(),
            spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
        )

    def forward(self, input):
        return self.conv_layers(input)


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {
            "block{}".format(i): block(
                nPlanes[0],
                nPlanes[0],
                norm_fn,
                indice_key="subm{}".format(indice_key_id),
            )
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                # nn.BatchNorm1d(nPlanes[0], eps=1e-4, momentum=0.1),
                nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[0],
                    nPlanes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key="spconv{}".format(indice_key_id),
                ),
            )

            self.u = UBlock(
                nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id + 1
            )

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                # nn.BatchNorm1d(nPlanes[1], eps=1e-4, momentum=0.1),
                nn.ReLU(),
                spconv.SparseInverseConv3d(
                    nPlanes[1],
                    nPlanes[0],
                    kernel_size=2,
                    bias=False,
                    indice_key="spconv{}".format(indice_key_id),
                ),
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail["block{}".format(i)] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    indice_key="subm{}".format(indice_key_id),
                )
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(
            output.features, output.indices, output.spatial_shape, output.batch_size
        )

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output.features = torch.cat(
                (identity.features, output_decoder.features), dim=1
            )

            output = self.blocks_tail(output)

        return output


class PointGroup(nn.Module):
    def __init__(
        self, cfg
    ):
        super().__init__()

        input_c = cfg["DATA"]["input_channel"]
        m = cfg["STRUCTURE"]["m"]
        classes = cfg["DATA"]["classes"]
        block_reps = cfg["STRUCTURE"]["block_reps"]
        block_residual = cfg["STRUCTURE"]["block_residual"]

        self.cluster_radius = cfg["GROUP"]["cluster_radius"]
        self.cluster_meanActive = cfg["GROUP"]["cluster_meanActive"]
        self.cluster_shift_meanActive = cfg["GROUP"]["cluster_shift_meanActive"]
        self.cluster_npoint_thre = cfg["GROUP"]["cluster_npoint_thre"]

        self.score_scale = cfg["TRAIN"]["score_scale"]
        self.score_fullscale = cfg["TRAIN"]["score_fullscale"]
        self.mode = cfg["TRAIN"]["score_mode"]

        self.prepare_epochs = cfg["GROUP"]["prepare_epochs"]

        self.pretrain_path = cfg["TRAIN"]["pretrain_path"]
        self.pretrain_module = cfg["TRAIN"]["pretrain_module"]
        self.fix_module = cfg["TRAIN"]["fix_module"]

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        if cfg["STRUCTURE"]["use_coords"]:
            input_c += 3

        #### backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_c, m, kernel_size=3, padding=1, bias=False, indice_key="subm1"
            )
        )

        self.unet = UBlock(
            [m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m],
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
        )

        self.output_layer = spconv.SparseSequential(
            norm_fn(m),
            # nn.BatchNorm1d(m, eps=1e-4, momentum=0.1),
            nn.ReLU(),
        )

        #### semantic segmentation
        self.linear = nn.Linear(m, classes)  # bias(default): True

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def clusters_voxelization(
        self, clusters_idx, clusters_offset, feats, coords, fullscale, scale, mode
    ):
        """
        :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param clusters_offset: (nCluster + 1), int, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :return:
        """
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()]
        clusters_coords = coords[c_idxs.long()]

        clusters_coords_mean = pointgroup_ops.sec_mean(
            clusters_coords, clusters_offset.cuda()
        )  # (nCluster, 3), float
        clusters_coords_mean = torch.index_select(
            clusters_coords_mean, 0, clusters_idx[:, 0].cuda().long()
        )  # (sumNPoint, 3), float
        clusters_coords -= clusters_coords_mean

        clusters_coords_min = pointgroup_ops.sec_min(
            clusters_coords, clusters_offset.cuda()
        )  # (nCluster, 3), float
        clusters_coords_max = pointgroup_ops.sec_max(
            clusters_coords, clusters_offset.cuda()
        )  # (nCluster, 3), float

        clusters_scale = (
            1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[0]
            - 0.01
        )  # (nCluster), float
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(
            -1
        )  # (nCluster, 3), float
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(
            clusters_scale, 0, clusters_idx[:, 0].cuda().long()
        )

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        range = max_xyz - min_xyz
        offset = (
            -min_xyz
            + torch.clamp(fullscale - range - 0.001, min=0) * torch.rand(3).cuda()
            + torch.clamp(fullscale - range + 0.001, max=0) * torch.rand(3).cuda()
        )
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert (
            clusters_coords.shape.numel()
            == ((clusters_coords >= 0) * (clusters_coords < fullscale)).sum()
        )

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat(
            [clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()], 1
        )  # (sumNPoint, 1 + 3)

        out_coords, inp_map, out_map = pointgroup_ops.voxelization_idx(
            clusters_coords, int(clusters_idx[-1, 0]) + 1, mode
        )
        # output_coords: M * (1 + 3) long
        # input_map: sumNPoint int
        # output_map: M * (maxActive + 1) int

        out_feats = pointgroup_ops.voxelization(
            clusters_feats, out_map.cuda(), mode
        )  # (M, C), float, cuda

        spatial_shape = [fullscale] * 3
        voxelization_feats = spconv.SparseConvTensor(
            out_feats,
            out_coords.int().cuda(),
            spatial_shape,
            int(clusters_idx[-1, 0]) + 1,
        )

        return voxelization_feats, inp_map

    def forward(self, input, input_map, coords, batch_idxs, batch_offsets, epoch):
        """
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        """
        ret = {}

        output = self.input_conv(input)
        output = self.unet(output)

        output = self.output_layer(output)
        output_feats = output.features[input_map.long()]

        #### semantic segmentation
        semantic_scores = self.linear(output_feats)  # (N, nClass), float

        semantic_preds = semantic_scores.max(1)[1]  # (N), long

        ret["semantic_scores"] = semantic_scores
        ret["feats"] = output_feats

        return ret


def model_fn_decorator(cfg, test=False):
    #### config
    # from util.config import cfg

    #### criterion
    semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg["DATA"]['ignore_label']).cuda()
    score_criterion = nn.BCELoss(reduction="none").cuda()

    def test_model_fn(batch, model, epoch):
        # print ('test model fn')
        coords = batch[
            "locs"
        ].cuda()  # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch["voxel_locs"].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch["p2v_map"].cuda()  # (N), int, cuda
        v2p_map = batch["v2p_map"].cuda()  # (M, 1 + maxActive), int, cuda

        coords_float = batch["locs_float"].cuda()  # (N, 3), float32, cuda
        feats = batch["feats"].cuda()  # (N, C), float32, cuda

        batch_offsets = batch["offsets"].cuda()  # (B + 1), int, cuda

        spatial_shape = batch["spatial_shape"]

        if cfg['STRUCTURE']['use_coords']:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = pointgroup_ops.voxelization(
            feats, v2p_map, cfg['DATA']['mode']
        )  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(
            voxel_feats, voxel_coords.int(), spatial_shape, cfg['DATA']['batch_size']
        )

        ret = model(
            input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch
        )
        semantic_scores = ret["semantic_scores"]  # (N, nClass) float32, cuda
        """pt_offsets = ret['pt_offsets']            # (N, 3), float32, cuda
        if (epoch > cfg.prepare_epochs):
            scores, proposals_idx, proposals_offset = ret['proposal_scores']"""

        ##### preds
        with torch.no_grad():
            preds = {}
            preds["semantic"] = semantic_scores
            preds["feats"] = ret["feats"]

        return preds

    def model_fn(batch, model, epoch):
        # print ('model fn')
        ##### prepare input and forward
        # batch {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        # 'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
        # 'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
        # 'id': tbl, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}
        coords = batch[
            "locs"
        ].cuda()  # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch["voxel_locs"].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch["p2v_map"].cuda()  # (N), int, cuda
        v2p_map = batch["v2p_map"].cuda()  # (M, 1 + maxActive), int, cuda

        coords_float = batch["locs_float"].cuda()  # (N, 3), float32, cuda
        feats = batch["feats"].cuda()  # (N, C), float32, cuda
        labels = batch["labels"].cuda()  # (N), long, cuda
        """instance_labels = batch['instance_labels'].cuda()      # (N), long, cuda, 0~total_nInst, -100

        instance_info = batch['instance_info'].cuda()          # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), int, cuda"""

        batch_offsets = batch["offsets"].cuda()  # (B + 1), int, cuda

        spatial_shape = batch["spatial_shape"]

        if cfg['STRUCTURE']['use_coords']:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = pointgroup_ops.voxelization(
            feats, v2p_map, cfg['DATA']['mode']
        )  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(
            voxel_feats, voxel_coords.int(), spatial_shape, cfg['DATA']['batch_size']
        )

        ret = model(
            input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch
        )
        semantic_scores = ret["semantic_scores"]  # (N, nClass) float32, cuda

        loss_inp = {}
        loss_inp["semantic_scores"] = (semantic_scores, labels)

        loss, loss_out, infos = loss_fn(loss_inp, epoch)

        ##### accuracy / visual_dict / meter_dict
        with torch.no_grad():
            preds = {}
            preds["semantic"] = semantic_scores

            visual_dict = {}
            visual_dict["loss"] = loss
            for k, v in loss_out.items():
                visual_dict[k] = v[0]

            meter_dict = {}
            meter_dict["loss"] = (loss.detach().item(), coords.shape[0])
            for k, v in loss_out.items():
                meter_dict[k] = (float(v[0]), v[1])
        # print (meter_dict.keys())
        return loss, preds, visual_dict, meter_dict

    def loss_fn(loss_inp, epoch):
        # print ('loss fn')

        loss_out = {}
        infos = {}

        """semantic loss"""
        semantic_scores, semantic_labels = loss_inp["semantic_scores"]
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda

        semantic_loss = semantic_criterion(semantic_scores, semantic_labels)
        loss_out["semantic_loss"] = (semantic_loss, semantic_scores.shape[0])

        """total loss"""
        loss = (
            cfg['TRAIN']['loss_weight'][0] * semantic_loss
        )

        return loss, loss_out, infos

    def get_segmented_scores(scores, fg_thresh=1.0, bg_thresh=0.0):
        """
        :param scores: (N), float, 0~1
        :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
        """
        fg_mask = scores > fg_thresh
        bg_mask = scores < bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)

        segmented_scores = (fg_mask > 0).float()
        k = 1 / (fg_thresh - bg_thresh)
        b = bg_thresh / (bg_thresh - fg_thresh)
        segmented_scores[interval_mask] = scores[interval_mask] * k + b

        return segmented_scores

    if test:
        fn = test_model_fn
    else:
        fn = model_fn
    return fn
