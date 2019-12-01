import torch
import torch.nn as nn
from torch.nn import functional as F


from vgg import vgg_face, VGG_Activations
from torchvision.models import vgg19



class Vgg_face_loss(nn.Module):
    def __init__(self, gpu=None):
        super(Vgg_face_loss, self).__init__()

        self.VGG_FACE_AC = VGG_Activations(vgg_face(pretrained=True), [1, 6, 11, 18, 25])
        # self.VGG19_AC = VGG_Activations(vgg19(pretrained=True), [1, 6, 11, 20, 29])

        self.gpu = gpu
        if gpu is not None:
            self.cuda(gpu)

    def forward(self, gt, input, output):
        IMG_NET_MEAN = torch.Tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).to(gt.device)
        IMG_NET_STD = torch.Tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).to(gt.device)

        gt = (gt - IMG_NET_MEAN) / IMG_NET_STD
        output = (output - IMG_NET_MEAN) / IMG_NET_STD


        # # VGG19 Loss
        # vgg19_output = self.VGG19_AC(output)
        # vgg19_gt = self.VGG19_AC(gt)
        #
        # vgg19_loss = 0
        # for i in range(0, len(vgg19_gt)):
        #     vgg19_loss += F.l1_loss(vgg19_output[i], vgg19_gt[i])

        # VGG Face Loss
        vgg_face_output = self.VGG_FACE_AC(output)
        vgg_face_gt = self.VGG_FACE_AC(gt)

        vgg_face_loss = 0
        for i in range(0, len(vgg_face_gt)):
            vgg_face_loss += F.l1_loss(vgg_face_output[i], vgg_face_gt[i])

        return 5e-3*vgg_face_loss



class L1_input(nn.Module):
    def __init__(self, gpu = None):
        super(L1_input, self).__init__()
        self.VGG_FACE_AC = VGG_Activations(vgg_face(pretrained=True), [1, 6, 11, 18, 25])
        # self.VGG19_AC = VGG_Activations(vgg19(pretrained=True), [1, 6, 11, 20, 29])

        self.gpu = gpu
        if gpu is not None:
            self.cuda(gpu)

    def forward(self, gt, input, output):

        input = input[:, :3]
        IMG_NET_MEAN = torch.Tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).to(gt.device)
        IMG_NET_STD = torch.Tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).to(gt.device)


        l1_loss = F.l1_loss(input, output)

        gt = (gt - IMG_NET_MEAN) / IMG_NET_STD
        output = (output - IMG_NET_MEAN) / IMG_NET_STD

        vgg_face_output = self.VGG_FACE_AC(output)
        vgg_face_gt = self.VGG_FACE_AC(gt)

        vgg_face_loss = 0
        for i in range(0, len(vgg_face_gt)):
            vgg_face_loss += F.l1_loss(vgg_face_output[i], vgg_face_gt[i])

        return l1_loss + 0.002*vgg_face_loss



class L1_gt(nn.Module):
    def __init__(self, gpu=None):
        super(L1_gt, self).__init__()
        self.VGG_FACE_AC = VGG_Activations(vgg_face(pretrained=True), [1, 6, 11, 18, 25])

        self.gpu = gpu
        if gpu is not None:
            self.cuda(gpu)

    def forward(self, gt, input, output):
        IMG_NET_MEAN = torch.Tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).to(gt.device)
        IMG_NET_STD = torch.Tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).to(gt.device)

        l1_loss = F.l1_loss(gt, output)

        gt = (gt - IMG_NET_MEAN) / IMG_NET_STD
        output = (output - IMG_NET_MEAN) / IMG_NET_STD

        vgg_face_output = self.VGG_FACE_AC(output)
        vgg_face_gt = self.VGG_FACE_AC(gt)

        vgg_face_loss = 0
        for i in range(0, len(vgg_face_gt)):
            vgg_face_loss += F.l1_loss(vgg_face_output[i], vgg_face_gt[i])

        return l1_loss + 0.002 * vgg_face_loss


class L1(nn.Module):
    def __init__(self, gpu = None):
        super(L1, self).__init__()

    def forward(self, gt, input, output):

        l1_loss = F.l1_loss(gt, output)


        return l1_loss




class L1_Percep_gt(nn.Module):
    def __init__(self, gpu=None):
        super(L1_Percep_gt, self).__init__()
        self.VGG_FACE_AC = VGG_Activations(vgg_face(pretrained=True), [1, 6, 11, 18, 25])
        self.VGG19_AC = VGG_Activations(vgg19(pretrained=True), [1, 6, 11, 20, 29])


        self.gpu = gpu
        if gpu is not None:
            self.cuda(gpu)

    def forward(self, gt, input, output):
        IMG_NET_MEAN = torch.Tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).to(gt.device)
        IMG_NET_STD = torch.Tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).to(gt.device)

        l1_loss = F.l1_loss(gt, output)

        gt = (gt - IMG_NET_MEAN) / IMG_NET_STD
        output = (output - IMG_NET_MEAN) / IMG_NET_STD

        vgg_face_output = self.VGG_FACE_AC(output)
        vgg_face_gt = self.VGG_FACE_AC(gt)

        # VGG19 Loss
        vgg19_output = self.VGG19_AC(output)
        vgg19_gt = self.VGG19_AC(gt)

        vgg19_loss = 0
        for i in range(0, len(vgg19_gt)):
            vgg19_loss += F.l1_loss(vgg19_output[i], vgg19_gt[i])


        vgg_face_loss = 0
        for i in range(0, len(vgg_face_gt)):
            vgg_face_loss += F.l1_loss(vgg_face_output[i], vgg_face_gt[i])

        return l1_loss + 0.01 * vgg_face_loss + 0.1*vgg19_loss