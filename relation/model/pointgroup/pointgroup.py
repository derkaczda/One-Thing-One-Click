'''
PointGroup
Written by Li Jiang
'''

import torch
import torch.nn as nn
import spconv.pytorch as spconv
import numpy as np
import random
from spconv.pytorch.modules import SparseModule
import functools
from collections import OrderedDict
import sys
sys.path.append('../../')
sys.path.append('lib/spconv/build/lib.linux-x86_64-3.7/spconv')
sys.path.append('lib/spconv')
from lib.pointgroup_ops.functions import pointgroup_ops
from util import utils

class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        # output.features += self.i_branch(identity).features
        output = output.replace_feature(
            output.features + self.i_branch(identity).features
        )

        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id)) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id+1)

            if nPlanes[0]==32:
              self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
              )
            else:
              self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
              )

            blocks_tail = {}

            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            # output.features = torch.cat((identity.features, output_decoder.features), dim=1)
            output = output.replace_feature(
                torch.cat((identity.features, output_decoder.features), dim=1)
            )

            output = self.blocks_tail(output)

        return output


class PointGroup(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        input_c = cfg["DATA"]["input_channel"]
        m = cfg["STRUCTURE"]["m"]
        self.m=m
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
            spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )

        self.unet = UBlock([m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], norm_fn, block_reps, block, indice_key_id=1)

        self.output_layer = spconv.SparseSequential(
            norm_fn(m) #,
            #nn.ReLU()
        )

        #### semantic segmentation
        #self.linear = nn.Linear(m, classes) # bias(default): True

        self.feat=torch.Tensor(10,m).cuda()
        self.start=1


        '''#### offset
        self.offset = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU()
        )
        self.offset_linear = nn.Linear(m, 3, bias=True)

        #### score branch
        self.score_unet = UBlock([m, 2*m], norm_fn, 2, block, indice_key_id=1)
        self.score_outputlayer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )
        self.score_linear = nn.Linear(m, 1)

        self.apply(self.set_bn_init)

        #### fix parameter
        module_map = {'input_conv': self.input_conv, 'unet': self.unet, 'output_layer': self.output_layer,
                      'linear': self.linear, 'offset': self.offset, 'offset_linear': self.offset_linear,
                      'score_unet': self.score_unet, 'score_outputlayer': self.score_outputlayer, 'score_linear': self.score_linear}

        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False

        #### load pretrain weights
        if self.pretrain_path is not None:
            pretrain_dict = torch.load(self.pretrain_path)
            for m in self.pretrain_module:
                print("Load pretrained " + m + ": %d/%d" % utils.load_model_param(module_map[m], pretrain_dict, prefix=m))'''


    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)


    def clusters_voxelization(self, clusters_idx, clusters_offset, feats, coords, fullscale, scale, mode):
        '''
        :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param clusters_offset: (nCluster + 1), int, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :return:
        '''
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()]
        clusters_coords = coords[c_idxs.long()]

        clusters_coords_mean = pointgroup_ops.sec_mean(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_mean = torch.index_select(clusters_coords_mean, 0, clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float
        clusters_coords -= clusters_coords_mean

        clusters_coords_min = pointgroup_ops.sec_min(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_max = pointgroup_ops.sec_max(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float

        clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[0] - 0.01  # (nCluster), float
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)  # (nCluster, 3), float
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        range = max_xyz - min_xyz
        offset = - min_xyz + torch.clamp(fullscale - range - 0.001, min=0) * torch.rand(3).cuda() + torch.clamp(fullscale - range + 0.001, max=0) * torch.rand(3).cuda()
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert clusters_coords.shape.numel() == ((clusters_coords >= 0) * (clusters_coords < fullscale)).sum()

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()], 1)  # (sumNPoint, 1 + 3)

        out_coords, inp_map, out_map = pointgroup_ops.voxelization_idx(clusters_coords, int(clusters_idx[-1, 0]) + 1, mode)
        # output_coords: M * (1 + 3) long
        # input_map: sumNPoint int
        # output_map: M * (maxActive + 1) int

        out_feats = pointgroup_ops.voxelization(clusters_feats, out_map.cuda(), mode)  # (M, C), float, cuda

        spatial_shape = [fullscale] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats, out_coords.int().cuda(), spatial_shape, int(clusters_idx[-1, 0]) + 1)

        return voxelization_feats, inp_map


    def forward(self, input, input_map):
        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        '''
        ret = {}

        output = self.input_conv(input)
        #print ('0',output.features)
        output = self.unet(output)
        #print ('1',output.features)
        output = self.output_layer(output)
        #print ('2',output.features)
        output_feats = output.features[input_map.long()]

        #### semantic segmentation
        return ret, output_feats


def model_fn_decorator(test=False):
    #### config
    from util.config import cfg

    #### criterion
    semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()
    score_criterion = nn.BCELoss(reduction='none').cuda()

    def test_model_fn(batch, model, epoch):
        #print ('test model fn')

        feats=torch.zeros((20,16))

        coords = batch['locs'].cuda()              # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()          # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()          # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()  # (N, 3), float32, cuda
        feats = batch['feats'].cuda()              # (N, C), float32, cuda

        batch_offsets = batch['offsets'].cuda()    # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']
        labels = batch['labels']
        groups=batch['groups']
        group2points=batch['group2points']

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        ret,output_feats = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch)
        products=torch.zeros((output_feats.shape[0], 20))
        result=torch.zeros((output_feats.shape[0]))
        result_feat=torch.zeros((output_feats.shape[0],32))
        #for group in groups:

        #print ('len',len(groups),groups)

        feat_voxel=[]
        gt_voxel=[]
        rgb_voxel=[]

        group=groups[0]
        for i in range(20):
          for s in range(len(group[i])):

            idxs=group2points[0][group[i][s]]
            feats=output_feats[idxs]
            #rgbs=feats[idxs]
            feat=torch.mean(feats,0)
            #rgb=torch.mean(rgbs,0)
            #feat_voxel.append(feat.detach().cpu().numpy())
            #gt_voxel.append(i)
            #rgb_voxel.append(rgb.detach().cpu().numpy())


            #stdfeat=torch.Tensor(torch.load('exp/scannetv2/pointgroup/pointgroup_run1_scannet/pointgroup_run1_scannet-000000074_feat.pth')).cuda()

            product=torch.matmul(feat,torch.transpose(model.feat,0,1))
            #print (i,int(torch.argmax(product)),product)
            #result[idxs]=int(torch.argmax(product))
            for idx in idxs:
              result[idx]=int(torch.argmax(product))
              result_feat[idx,:]=feat
              products[idx,:]=product
            #print (int(torch.argmax(product)))
            #print ('idxs',len(idxs),idxs)

        #print ('unique',torch.unique(result))
        #semantic_scores=torch.argmax(product,1)



        '''feats=np.zeros((20,16))
        labels=np.zeros((20))
        for i in range(feat.shape[0]):
          if label[i]==-100:
            continue
          feats[label[i],:]+=feat[i,:].cpu().numpy()
          labels[label[i]]+=1
        for i in range(20):
          feats[i,:]/=labels[i]'''

        #print (feat.shape,label.shape)


        #semantic_scores = ret['semantic_scores']  # (N, nClass) float32, cuda
        '''pt_offsets = ret['pt_offsets']            # (N, 3), float32, cuda
        if (epoch > cfg.prepare_epochs):
            scores, proposals_idx, proposals_offset = ret['proposal_scores']'''



        '''product=torch.matmul(feat,torch.transpose(model.feat,0,1))


        print ('product',product.shape)

        result=torch.argmax(product,1)

        print ('result',result.shape)'''



        #loss_inp = {}
        #loss_inp['semantic_scores'] = (product, label)

        ##### preds
        with torch.no_grad():
            preds = {}
            preds['semantic'] = result
            preds['products']=products
            print ('feat')
            preds['feat'] = result_feat #np.asarray(feat_voxel)
            #preds['gt'] = np.asarray(gt_voxel)
            #preds['rgb'] = np.asarray(rgb_voxel)
            '''preds['pt_offsets'] = pt_offsets
            if (epoch > cfg.prepare_epochs):
                preds['score'] = scores
                preds['proposals'] = (proposals_idx, proposals_offset)'''

        return preds


    def model_fn(batch, model, epoch):
        #print ('model fn')
        ##### prepare input and forward
        # batch {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        # 'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
        # 'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
        # 'id': tbl, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}
        coords = batch['locs'].cuda()                          # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()              # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()                      # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()                      # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()              # (N, 3), float32, cuda
        feats = batch['feats'].cuda()                          # (N, C), float32, cuda
        labels = batch['labels'].cuda()                        # (N), long, cuda

        groups=batch['groups']
        group2points=batch['group2points']

        #for i in range(4):
        #    print (len(groups[i][0]))

        classes=[]
        poss=[]
        negs=[]
        for i in range(20):
            classes.append([])

        for g in range(len(groups)):
            group=groups[g]
            for i in range(20):
                for s in range(len(group[i])):
                    classes[i].append((i,g,group[i][s]))

        ignore=[]
        mini=10 #min(min(map(len, classes)),30)
        for i in range(20):
            random.shuffle(classes[i])
            if len(classes[i])==0:
              ignore.append(i)
              continue
            if len(classes[i])>=mini:
              classes[i]=classes[i][:mini]
            else:
              while(len(classes[i])<10):
                classes[i].append(random.choice(classes[i]))



        '''for i in range(20):
            if len(classes[i])==0:
                continue
            for times in range(10):
                poss.append((random.choice(classes[i]),random.choice(classes[i])))
            for j in range(20):
                if j==i or len(classes[j])==0:
                  continue
                negs.append((random.choice(classes[i]),random.choice(classes[j])))
            while(len(negs)<20):
                j=random.randint(0,19)
                if j==i or len(classes[j])==0:
                  continue
                negs.append((random.choice(classes[i]),random.choice(classes[j])))


        posidxs=[]
        for pos in poss:
            c0 = pos[0][0]
            b0 = pos[0][1]
            idx0 = pos[0][2]
            #print ('fffff',group2points[b0].keys())
            idx_off0 = torch.tensor(np.asarray(group2points[b0][idx0])) + batch['offsets'][b0]
            #print ('sssssss',group2points[b0])
            c1 = pos[1][0]
            b1 = pos[1][1]
            idx1 = pos[1][2]
            idx_off1 = torch.tensor(np.asarray(group2points[b1][idx1])) + batch['offsets'][b1]
            posidxs.append((idx_off0,idx_off1))

        negidxs=[]
        for neg in negs:
            c0 = neg[0][0]
            b0 = neg[0][1]
            idx0 = neg[0][2]
            idx_off0 = torch.tensor(np.asarray(group2points[b0][idx0])) + batch['offsets'][b0]
            #print (idx_off0)
            c1 = neg[1][0]
            b1 = neg[1][1]
            idx1 = neg[1][2]
            idx_off1 = torch.tensor(np.asarray(group2points[b1][idx1])) + batch['offsets'][b1]
            negidxs.append((idx_off0,idx_off1))'''

        batch_offsets = batch['offsets'].cuda()                # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        ret, output_feats = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch)

        #print (torch.unique(voxel_feats),output_feats)


        tmpfeat=torch.Tensor(20,mini,model.m).cuda()


        label=torch.zeros(20*mini).long().cuda()

        for i in range(20):
          if i in ignore:
            tmpfeat[i,:,:]=0
            label[i*mini:i*mini+mini]=-100
            continue
          for j in range(mini):
            sample=classes[i][j]
            c0 = sample[0]
            b0 = sample[1]
            idx0 = sample[2]
            #print ('fffff',group2points[b0].keys())
            idx_off = torch.tensor(np.asarray(group2points[b0][idx0])) + batch['offsets'][b0]
            feat=output_feats[idx_off]
            feat=torch.mean(feat,0)
            tmpfeat[i,j,:]=feat

            label[i*mini+j]=i



        if model.start==1:
          model.start=0

          model.feat=torch.mean(tmpfeat.detach(),1)
        else:
          model.feat=0.9*model.feat+0.1*torch.mean(tmpfeat.detach(),1)




        #model.feat=nn.functional.normalize(model.feat,1)
        #tmpfeat=nn.functional.normalize(tmpfeat,2)

        tmpfeat=torch.reshape(tmpfeat,(20*mini,model.m))

        product=torch.matmul(tmpfeat,torch.transpose(model.feat,0,1))/0.07



        loss_inp = {}
        loss_inp['semantic_scores'] = (product, label)
        '''posfeat=torch.zeros((10,128*2))
        for j in range(len(posidxs)):
          posidx=posidxs[j]
          feat0=output_feats[posidx[0]]
          feat0=torch.mean(feat0,0)
          feat1=output_feats[posidx[1]]
          feat1=torch.mean(feat0,0)
          feat=torch.cat((feat0,feat1),1)
          posfeat[j][:128]=feat0
          posfeat[j][128:]=feat1


        negfeat=torch.zeros((20,128*2))
        for j in range(len(negidxs)):
          negidx=negidxs[j]
          feat0=output_feats[negidx[0]]
          feat0=torch.mean(feat0,0)
          feat1=output_feats[negidx[1]]
          feat1=torch.mean(feat0,0)
          feat=torch.cat((feat0,feat1),1)
          negfeat[j][:128]=feat0
          negfeat[j][128:]=feat1'''
        '''semantic_scores = ret['semantic_scores'] # (N, nClass) float32, cuda


        print ('s',semantic_scores.shape)
        #print ('groups',groups)


        loss_inp = {}
        loss_inp['semantic_scores'] = (semantic_scores, labels)
        print ('l',labels.shape,'\n')'''

        loss, loss_out, infos = loss_fn(loss_inp, epoch)


        ##### accuracy / visual_dict / meter_dict
        with torch.no_grad():

            preds = {}
            #preds['semantic'] = semantic_scores


            visual_dict = {}
            visual_dict['loss'] = loss
            for k, v in loss_out.items():
                visual_dict[k] = v[0]

            meter_dict = {}
            meter_dict['loss'] = (loss.item(), coords.shape[0])
            for k, v in loss_out.items():
                meter_dict[k] = (float(v[0]), v[1])

        return loss, preds, visual_dict, meter_dict


    def loss_fn(loss_inp, epoch):
        #print ('loss fn')

        loss_out = {}
        infos = {}

        '''semantic loss'''
        semantic_scores, semantic_labels = loss_inp['semantic_scores']
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda

        semantic_loss = semantic_criterion(semantic_scores, semantic_labels)
        loss_out['semantic_scores'] = (semantic_loss, semantic_scores.shape[0])

        '''offset loss'''
        '''pt_offsets, coords, instance_info, instance_labels = loss_inp['pt_offsets']
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 9), float32 tensor (meanxyz, minxyz, maxxyz)
        # instance_labels: (N), long

        gt_offsets = instance_info[:, 0:3] - coords   # (N, 3)
        pt_diff = pt_offsets - gt_offsets   # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
        valid = (instance_labels != cfg.ignore_label).float()
        offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)   # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
        pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)   # (N)
        offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

        loss_out['offset_norm_loss'] = (offset_norm_loss, valid.sum())
        loss_out['offset_dir_loss'] = (offset_dir_loss, valid.sum())

        if (epoch > cfg.prepare_epochs):
            #score loss
            scores, proposals_idx, proposals_offset, instance_pointnum = loss_inp['proposal_scores']
            # scores: (nProposal, 1), float32
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # instance_pointnum: (total_nInst), int

            ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(), instance_labels, instance_pointnum) # (nProposal, nInstance), float
            gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
            gt_scores = get_segmented_scores(gt_ious, cfg.fg_thresh, cfg.bg_thresh)

            score_loss = score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
            score_loss = score_loss.mean()

            loss_out['score_loss'] = (score_loss, gt_ious.shape[0])'''

        '''total loss'''
        loss = cfg.loss_weight[0] * semantic_loss #+ cfg.loss_weight[1] * offset_norm_loss + cfg.loss_weight[2] * offset_dir_loss
        #if(epoch > cfg.prepare_epochs):
        #    loss += (cfg.loss_weight[3] * score_loss)

        return loss, loss_out, infos


    def get_segmented_scores(scores, fg_thresh=1.0, bg_thresh=0.0):
        '''
        :param scores: (N), float, 0~1
        :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
        '''
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
