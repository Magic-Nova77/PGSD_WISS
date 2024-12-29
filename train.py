 import torch
import numpy as np
from torch import distributed
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import tqdm
from utils.loss import KnowledgeDistillationLoss, BCEWithLogitsLossWithIgnoreIndex, \
    UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropy, IcarlLoss
from torch.cuda import amp
from segmentation_module import make_model, TestAugmentation
import tasks
from torch.nn.parallel import DistributedDataParallel
import os.path as osp
from wss.modules import PAMR, ASPP
from utils.utils import denorm, label_to_one_hot
from wss.single_stage import pseudo_gtmask, balanced_mask_loss_ce, balanced_mask_loss_unce
from utils.wss_loss import bce_loss, ngwp_focal, binarize
from segmentation_module import get_norm
from utils.scheduler import get_scheduler


class Trainer:
    def __init__(self, logger, device, opts):
        self.logger = logger
        self.device = device
        self.opts = opts
        self.scaler = amp.GradScaler()
        self.step = opts.step

        self.bkg_alpha = nn.Parameter(torch.ones(1), requires_grad=True).cuda()
        self.old_cls_alpha = nn.Parameter(torch.ones(1), requires_grad=True).cuda()
        self.new_cls_alpha = nn.Parameter(torch.ones(1), requires_grad=True).cuda()

        self.classes = classes = tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)

        if classes is not None:
            new_classes = classes[-1]
            self.tot_classes = reduce(lambda a, b: a + b, classes)
            self.old_classes = self.tot_classes - new_classes
            self.new_classes = new_classes
        else:
            self.old_classes = 0

        old_class_num = int(opts.task[:opts.task.find("-")])
        pth_name = "./sim/clip_name_synonym_description.pt"
        self.my_sim = torch.load(pth_name)
        # _, indices = torch.topk(self.my_sim, 3, -1, True, sorted=False)
        self.my_sim *= 10.
        for i in range(self.tot_classes):
            self.my_sim[i][:old_class_num + 1] = torch.softmax(self.my_sim[i][:old_class_num + 1], dim=-1)
        self.task = opts.task

        self.super_cls = K-means(pth_name)

        self.sim_alpha = 0.1

        self.alpha_lde = 150.

        self.model = make_model(opts, classes=classes, sim=self.my_sim)

        if opts.step == 0:  # if step 0, we don't need to instance the model_old
            self.model_old = None
        else:  # instance model_old
            self.model_old = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step - 1))
            self.model_old.to(self.device)
            # freeze old model and set eval mode
            for par in self.model_old.parameters():
                par.requires_grad = False
            self.model_old.eval()

        self.weakly = opts.weakly and opts.step > 0
        self.pos_w = opts.pos_w
        self.use_aff = opts.affinity
        self.weak_single_stage_dist = opts.ss_dist
        self.pseudo_epoch = opts.pseudo_ep
        cls_classes = self.tot_classes
        self.pseudolabeler = None

        if self.weakly:
            self.affinity = PAMR(num_iter=10, dilations=[1, 2, 4, 8, 12]).to(device)
            for p in self.affinity.parameters():
                p.requires_grad = False
            norm = get_norm(opts)
            channels = 4096 if "wide" in opts.backbone else 2048
            self.pseudolabeler = nn.Sequential(nn.Conv2d(channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                               norm(256),
                                               nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                               norm(256),
                                               nn.Conv2d(256, cls_classes, kernel_size=1, stride=1))

            self.icarl = opts.icarl

        self.optimizer, self.scheduler = self.get_optimizer(opts)

        self.distribute(opts)

        # Select the Loss Type
        reduction = 'none'

        self.bce = opts.bce or opts.icarl
        if self.bce:
            self.criterion = BCEWithLogitsLossWithIgnoreIndex(reduction=reduction)
        elif opts.unce and self.old_classes != 0:
            self.criterion = UnbiasedCrossEntropy(old_cl=self.old_classes, ignore_index=255, reduction=reduction)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

        # ILTSS
        self.lde = opts.loss_de
        self.lde_flag = self.lde > 0. and self.model_old is not None
        self.lde_loss = nn.MSELoss()

        self.lkd = opts.loss_kd
        self.lkd_flag = self.lkd > 0. and self.model_old is not None
        if opts.unkd:
            self.lkd_loss = UnbiasedKnowledgeDistillationLoss(alpha=opts.alpha)
        else:
            self.lkd_loss = KnowledgeDistillationLoss(alpha=opts.alpha)

        # ICARL
        self.icarl_combined = False
        self.icarl_only_dist = False
        if opts.icarl:
            self.icarl_combined = not opts.icarl_disjoint and self.model_old is not None
            self.icarl_only_dist = opts.icarl_disjoint and self.model_old is not None
            if self.icarl_combined:
                self.licarl = nn.BCEWithLogitsLoss(reduction='mean')
                self.icarl = opts.icarl_importance
            elif self.icarl_only_dist:
                self.licarl = IcarlLoss(reduction='mean', bkg=opts.icarl_bkg)
        self.icarl_dist_flag = self.icarl_only_dist or self.icarl_combined

    def get_optimizer(self, opts):
        params = []
        if not opts.freeze:
            params.append({"params": filter(lambda p: p.requires_grad, self.model.body.parameters()),
                           'weight_decay': opts.weight_decay})

        params.append({"params": filter(lambda p: p.requires_grad, self.model.head.parameters()),
                       'weight_decay': opts.weight_decay, 'lr': opts.lr*opts.lr_head})
        params.append({"params": filter(lambda p: p.requires_grad, self.model.cls.parameters()),
                       'weight_decay': opts.weight_decay, 'lr': opts.lr*opts.lr_head})
        if self.weakly:
            params.append({"params": filter(lambda p: p.requires_grad, self.pseudolabeler.parameters()),
                           'weight_decay': opts.weight_decay, 'lr': opts.lr_pseudo})

        optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=True)
        scheduler = get_scheduler(opts, optimizer)

        return optimizer, scheduler

    def distribute(self, opts):
        self.model = DistributedDataParallel(self.model.to(self.device), device_ids=[opts.device_id],
                                             output_device=opts.device_id, find_unused_parameters=False)
        if self.weakly:
            self.pseudolabeler = DistributedDataParallel(self.pseudolabeler.to(self.device), device_ids=[opts.device_id],
                                                         output_device=opts.device_id, find_unused_parameters=False)

    def train(self, cur_epoch, train_loader, print_int=10):
        """Train and return epoch loss"""
        optim = self.optimizer
        scheduler = self.scheduler
        device = self.device
        model = self.model
        criterion = self.criterion
        logger = self.logger

        nb_new_classes = self.new_classes
        nb_old_classes = self.old_classes
        nb_tot_classes = self.tot_classes
        my_sim = self.my_sim

        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

        epoch_loss = 0.0
        reg_loss = 0.0
        l_cam_out = 0.0
        l_cam_int = 0.0
        l_seg = 0.0
        l_cls = 0.0
        interval_loss = 0.0

        l_seg_base = 0.0
        l_seg_new = 0.0

        l_mask = 0.0
        l_super_dis = 0.0

        l_loc = 0.0

        l_sim = 0.0
        l_sim_add = 0.0

        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)

        train_loader.sampler.set_epoch(cur_epoch)

        if distributed.get_rank() == 0:
            tq = tqdm.tqdm(total=len(train_loader))
            tq.set_description("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))
        else:
            tq = None

        model.train()

        for cur_step, (images, labels, l1h) in enumerate(train_loader):

            images = images.to(device, dtype=torch.float)
            # one-hot的image-level标签
            l1h = l1h.to(device, dtype=torch.float)  # this are one_hot
            # pixel-level标签
            labels = labels.to(device, dtype=torch.long)

            with amp.autocast():
                if (self.lde_flag or self.lkd_flag or self.icarl_dist_flag or self.weakly) and self.model_old is not None:
                    with torch.no_grad():
                        outputs_old, features_old = self.model_old(images, interpolate=False)

                optim.zero_grad()

                # xxx BCE / Cross Entropy Loss
                if not self.weakly:
                    outputs = F.interpolate(outputs, size=images.shape[-2:], mode="bilinear", align_corners=False)
                    if not self.icarl_only_dist:
                        loss = criterion(outputs, labels)  # B x H x W
                    else:
                        # ICaRL loss -- unique CE+KD
                        outputs_old = F.interpolate(outputs_old, size=images.shape[-2:], mode="bilinear",
                                                    align_corners=False)
                        loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old))

                    loss = loss.mean()  # scalar

                    # xxx ICARL DISTILLATION
                    if self.icarl_combined:
                        # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                        n_cl_old = outputs_old.shape[1]
                        outputs_old = F.interpolate(outputs_old, size=images.shape[-2:], mode="bilinear",
                                                    align_corners=False)
                        # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                        l_icarl = self.icarl * n_cl_old * self.licarl(outputs.narrow(1, 0, n_cl_old),
                                                                      torch.sigmoid(outputs_old))

                    # xxx ILTSS (distillation on features or logits)
                    if self.lde_flag:
                        lde = self.lde * torch.mean(self.lde_loss(features['body'], features_old['body']))

                    if self.lkd_flag:
                        outputs_old = F.interpolate(outputs_old, size=images.shape[-2:], mode="bilinear",
                                                    align_corners=False)
                        # resize new output to remove new logits and keep only the old ones
                        lkd = self.lkd * self.lkd_loss(outputs, outputs_old)

                else:
                    bs = images.shape[0]

                    self.pseudolabeler.eval()
                    int_masks = self.pseudolabeler(features['body']).detach()

                    self.pseudolabeler.train()
                    # localizer的输出
                    int_masks_raw = self.pseudolabeler(features['body'])

                    if self.opts.no_mask:
                        l_cam_new = bce_loss(int_masks_raw, l1h, mode=self.opts.cam, reduction='mean')
                    else:
                        l_cam_new = bce_loss(int_masks_raw, l1h[:, self.old_classes - 1:],
                                             mode=self.opts.cam, reduction='mean')

                    l_loc = F.binary_cross_entropy_with_logits(int_masks_raw[:, :self.old_classes],
                                                               torch.sigmoid(outputs_old.detach()),
                                                               reduction='mean')

                    _outputs_old = torch.zeros_like(int_masks_raw[:, self.old_classes:]).to(outputs_old.dtype)

                    for n in range(nb_new_classes):
                        current_class = nb_old_classes + n
                        current_sim = my_sim[current_class][:nb_old_classes]
                        current_sim = current_sim.reshape(current_sim.shape[0], 1, 1).to(int_masks_raw.device)
                        _outputs_old[:, n] += torch.sum(current_sim * outputs_old, dim=1)

                    l_sim = torch.abs(int_masks_raw[:, self.old_classes:] - _outputs_old)
                    l_sim = torch.mean(l_sim)

                    l_sim_add = torch.abs(outputs[:, self.old_classes:] - _outputs_old)
                    l_sim_add = torch.mean(l_sim_add)

                    b, _, h, w = outputs.shape
                    super_class_num = torch.max(self.super_cls) + 1
                    now_classes = self.old_classes + self.new_classes
                    super_class_mask = torch.zeros([super_class_num, now_classes])
                    for i in range(super_class_num):
                        super_class_mask[i][self.super_cls == i] = 1
                    super_class_mask = super_class_mask.transpose(1, 0).to(outputs.device)

                    super_outputs = torch.softmax(outputs, dim=1).permute(0, 2, 3, 1).reshape(-1, now_classes)
                    super_outputs = super_outputs.matmul(super_class_mask).reshape(b, h, w, super_class_num). \
                        permute(0, 3, 1, 2)
                    super_outputs_old = torch.softmax(outputs_old, dim=1).permute(0, 2, 3, 1). \
                        reshape(-1, self.old_classes)
                    super_outputs_old = super_outputs_old.matmul(super_class_mask[:self.old_classes, :]). \
                        reshape(b, h, w, super_class_num).permute(0, 3, 1, 2)

                    l_super_dis = torch.abs(super_outputs - super_outputs_old)
                    l_super_dis = torch.mean(l_super_dis)

                    l_cam_int = l_cam_new + l_loc + (l_sim + l_sim_add) * self.sim_alpha + l_super_dis

                    if self.lde_flag:
                        lde = self.lde * self.lde_loss(features['body'], features_old['body']) * self.alpha_lde

                    l_cam_out = 0 * outputs[0, 0].mean()  # avoid errors due to DDP

                    if cur_epoch >= self.pseudo_epoch:
                        int_masks_orig = int_masks.softmax(dim=1)
                        int_masks_soft = int_masks.softmax(dim=1)
                        masks_spec = torch.zeros(12, 21)

                        if self.use_aff:
                            image_raw = denorm(images)
                            im = F.interpolate(image_raw, int_masks.shape[-2:], mode="bilinear",
                                               align_corners=True)
                            int_masks_soft = self.affinity(im, int_masks_soft.detach())

                        int_masks_orig[:, 1:] *= l1h[:, :, None, None]
                        int_masks_soft[:, 1:] *= l1h[:, :, None, None]

                        pseudo_gt_seg = pseudo_gtmask(int_masks_soft, ambiguous=True, cutoff_top=0.6,
                                                      cutoff_bkg=0.7, cutoff_low=0.2).detach()  # B x C x HW
                        pseudo_gt_seg_lx = binarize(int_masks_orig)
                        pseudo_gt_seg_lx = (self.opts.alpha * pseudo_gt_seg_lx) + \
                                           ((1-self.opts.alpha) * int_masks_orig)

                        # ignore_mask = (pseudo_gt_seg.sum(1) > 0)
                        px_cls_per_image = pseudo_gt_seg_lx.view(bs, self.tot_classes, -1).sum(dim=-1)
                        batch_weight = torch.eq((px_cls_per_image[:, self.old_classes:] > 0),
                                                l1h[:, self.old_classes - 1:].bool())
                        batch_weight = (
                                    batch_weight.sum(dim=1) == (self.tot_classes - self.old_classes)).float()

                        # origin
                        target_old = torch.sigmoid(outputs_old.detach())

                        target = torch.cat((target_old, pseudo_gt_seg_lx[:, self.old_classes:]), dim=1)
                        if self.opts.icarl_bkg == -1:
                            # wilson进入该分支
                            target[:, 0] = torch.min(target[:, 0], pseudo_gt_seg_lx[:, 0])
                        else:
                            target[:, 0] = (1-self.opts.icarl_bkg) * target[:, 0] + \
                                           self.opts.icarl_bkg * pseudo_gt_seg_lx[:, 0]

                        l_seg = F.binary_cross_entropy_with_logits(outputs, target, reduction='none').sum(dim=1)

                        l_seg = l_seg.view(bs, -1).mean(dim=-1)
                        l_seg = self.opts.l_seg * (batch_weight * l_seg).sum() / (batch_weight.sum() + 1e-5)

                        l_cls = balanced_mask_loss_ce(int_masks_raw, pseudo_gt_seg, l1h)

                    loss = l_seg + l_cam_out
                    l_reg = l_cls + l_cam_int
                # xxx first backprop of previous loss (compute the gradients for regularization methods)
                loss_tot = loss + lkd + lde + l_icarl + l_reg

            self.scaler.scale(loss_tot).backward()
            self.scaler.step(optim)
            if scheduler is not None:
                scheduler.step()
            self.scaler.update()

            epoch_loss += loss.item()
            reg_loss += l_reg.item() if l_reg != 0. else 0.
            reg_loss += lkd.item() + lde.item() + l_icarl.item()
            interval_loss += loss.item() + lkd.item() + lde.item() + l_icarl.item()
            interval_loss += l_reg.item() if l_reg != 0. else 0.

            if tq is not None:
                tq.update(1)
                tq.set_postfix(loss='%.6f' % loss)

            if (cur_step + 1) % print_int == 0:
                interval_loss = interval_loss / print_int
                logger.debug(f"Epoch {cur_epoch}, Batch {cur_step + 1}/{len(train_loader)},"
                             f" Loss={interval_loss}")
                logger.debug(f"Loss made of: CE {loss}, LKD {lkd}, LDE {lde}, LReg {l_reg}")
                # visualization
                if logger is not None:
                    x = cur_epoch * len(train_loader) + cur_step + 1
                    logger.add_scalar('Loss/tot', interval_loss, x, intermediate=True)
                    logger.add_scalar('Loss/CAM_int', l_cam_int, x, intermediate=True)
                    logger.add_scalar('Loss/CAM_out', l_cam_out, x, intermediate=True)
                    logger.add_scalar('Loss/SEG_int', l_cls, x, intermediate=True)
                    logger.add_scalar('Loss/SEG_out', l_seg, x, intermediate=True)
                    logger.commit(intermediate=True)
                interval_loss = 0.0

        if tq is not None:
            tq.close()

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        reg_loss = torch.tensor(reg_loss).to(self.device)

        torch.distributed.reduce(epoch_loss, dst=0)
        torch.distributed.reduce(reg_loss, dst=0)

        if distributed.get_rank() == 0:
            epoch_loss = epoch_loss / distributed.get_world_size() / len(train_loader)
            reg_loss = reg_loss / distributed.get_world_size() / len(train_loader)

        logger.info(f"Epoch {cur_epoch}, Class Loss={epoch_loss}, Reg Loss={reg_loss}")

        return (epoch_loss, reg_loss)

    def validate(self, loader, metrics):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device

        model.eval()

        with torch.no_grad():
            for i, x in enumerate(loader):
                images = x[0].to(device, dtype=torch.float32)
                labels = x[1].to(device, dtype=torch.long)

                with amp.autocast():
                    outputs, features = model(images)
                _, prediction = outputs.max(dim=1)

                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

            # collect statistics from multiple processes
            metrics.synch(device)
            score = metrics.get_results()

        return score

    def validate_CAM(self, loader, metrics):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device

        self.pseudolabeler.eval()
        model.eval()

        def classify(images):
            masks = self.pseudolabeler(model(images, as_feature_extractor=True)['body'])
            masks = F.interpolate(masks, size=images.shape[-2:], mode="bilinear", align_corners=False)
            masks = masks.softmax(dim=1)
            return masks

        i = -1
        with torch.no_grad():
            for x in tqdm.tqdm(loader):
                i = i+1
                images = x[0].to(device, dtype=torch.float32)
                labels = x[1].to(device, dtype=torch.long)
                l1h = x[2].to(device, dtype=torch.bool)

                with amp.autocast():
                    masks = classify(images)

                _, prediction = masks.max(dim=1)

                labels[labels < self.old_classes] = 0
                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

            # collect statistics from multiple processes
            metrics.synch(device)
            score = metrics.get_results()

        return score

    def load_step_ckpt(self, path):
        # generate model from path
        if osp.exists(path):
            step_checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(step_checkpoint['model_state'], strict=False)  # False for incr. classifiers
            if self.opts.init_balanced:
                # implement the balanced initialization (new cls has weight of background and bias = bias_bkg - log(N+1)
                self.model.module.init_new_classifier(self.device)
            # Load state dict from the model state dict, that contains the old model parameters
            new_state = {}
            for k, v in step_checkpoint['model_state'].items():
                new_state[k[7:]] = v
            self.model_old.load_state_dict(new_state, strict=True)  # Load also here old parameters

            self.logger.info(f"[!] Previous model loaded from {path}")
            # clean memory
            del step_checkpoint['model_state']
        elif self.opts.debug:
            self.logger.info(f"[!] WARNING: Unable to find of step {self.opts.step - 1}! "
                             f"Do you really want to do from scratch?")
        else:
            raise FileNotFoundError(path)

    def load_ckpt(self, path):
        opts = self.opts
        assert osp.isfile(path), f"Error, ckpt not found in {path}"

        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        if self.weakly:
            self.pseudolabeler.load_state_dict(checkpoint["pseudolabeler"])

        cur_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint['best_score']
        self.logger.info("[!] Model restored from %s" % opts.ckpt)
        # if we want to resume training, resume trainer from checkpoint
        del checkpoint

        return cur_epoch, best_score


def custom_loss_function(output):
    loss = torch.mean(output ** 2)
    return loss