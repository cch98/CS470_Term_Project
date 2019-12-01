import argparse
import logging
import os
import os.path
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
from PIL import Image

from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader

from face_alignment import FaceAlignment, LandmarksType
import matplotlib.pyplot as plt
import numpy as np

from loss import Vgg_face_loss, L1, L1_gt, L1_input, L1_Percep_gt
from dataloader import NFNDataset

from RAdam.radam import RAdam

from model import Pix2Pix, DRA


from tensorboardX import SummaryWriter


def train(gt_path, noisy_path,landmark_path, gpu, tb_dir, checkpoint_dir, loss_type):
    writer = SummaryWriter(os.path.join("tb_logs", tb_dir))

    # region DATASET----------------------------------------------------------------------------------------------------
    train_dataset = NFNDataset(
        gt_root=os.path.join(gt_path, "train"),
        noisy_root=os.path.join(noisy_path, "train"),
        landmark_root=os.path.join(landmark_path, "train"),
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
    )

    test_dataset = NFNDataset(
        gt_root=os.path.join(gt_path, "test"),
        noisy_root=os.path.join(noisy_path, "test"),
        landmark_root=os.path.join(landmark_path, "test"),
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
    )

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=40)
    test_dataloder = DataLoader(test_dataset, batch_size=8, num_workers=10)

    # endregion



    # region NETWORK

    NFNet = DRA()
    # NFNet = Pix2Pix()
    NFNet.to(gpu)

    optimizer = RAdam(params=list(NFNet.parameters()),lr=1e-4)



    # endregion

    init_epoch = -1
    NFNet.train()


    loss_func = Vgg_face_loss(gpu)
    if loss_type == "L1":
        loss_func = L1(gpu)
        print(f"loss type: {loss_type}")
    elif loss_type == "L1_gt":
        loss_func = L1_gt(gpu)
        print(f"loss type: {loss_type}")
    elif loss_type == "L1_input":
        loss_func = L1_input(gpu)
        print(f"loss type: {loss_type}")

    elif loss_type == "L1_Percep_gt":
        loss_func = L1_Percep_gt(gpu)
        print(f"loss type: {loss_type}")


    train_loss = 0
    test_loss = 0

    for epoch in range(init_epoch + 1, 300):
        train_loss = 0
        for num, (gt, x) in enumerate(train_dataloader):
            x = x.to(gpu)
            gt = gt.to(gpu)

            output = NFNet(x)
            loss = loss_func(gt, x, output)

            train_loss+=loss.item()
            loss.backward()
            optimizer.step()


            if num>1 and num%1000 == 0:
                NFNet.eval()
                test_loss = 0
                gts = []
                lms = []
                noisys = []
                outputs=[]

                for test_num, (gt, x) in enumerate(test_dataloder):

                    x = x.to(gpu)
                    gt = gt.to(gpu)
                    output = NFNet(x)
                    loss = loss_func(gt, x, output)

                    test_loss+=loss.item()
                    if test_num == 1:
                        gts = gt.cpu()
                        lms = x[:,:3].cpu()
                        noisys = x[:,3:].cpu()
                        outputs = output.cpu()

                writer.add_scalar("train_loss", train_loss/1000, len(train_dataloader) * epoch + num)
                writer.add_scalar("test_loss", test_loss/len(test_dataloder), len(train_dataloader) * epoch + num)

                print(f"epoch:{epoch}, train loss:{train_loss}, test_loss:{test_loss}")

                img = [gts[0], lms[0], noisys[0], outputs[0], gts[1], lms[1], noisys[1], outputs[1], gts[2], lms[2], noisys[2], outputs[2], gts[3], lms[3], noisys[3], outputs[3], gts[4], lms[4], noisys[4], outputs[4]]
                img = torchvision.utils.make_grid(img, 4)
                writer.add_image("test_img", img, len(train_dataloader) * epoch + num)

                NFNet.train()

                ckpt = {'my_classifier': NFNet.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': train_loss}
                torch.save(ckpt, os.path.join(checkpoint_dir, f'{tb_dir}_{epoch:04}.pt'))

                train_loss = test_loss = 0



def plot_landmarks(frame, landmarks):
    """
    Creates an RGB image with the landmarks. The generated image will be of the same size as the frame where the face
    matching the landmarks.

    The image is created by plotting the coordinates of the landmarks using matplotlib, and then converting the
    plot to an image.

    Things to watch out for:
    * The figure where the landmarks will be plotted must have the same size as the image to create, but matplotlib
    only accepts the size in inches, so it must be converted to pixels using the DPI of the screen.
    * A white background is printed on the image (an array of ones) in order to keep the figure from being flipped.
    * The axis must be turned off and the subplot must be adjusted to remove the space where the axis would normally be.

    :param frame: Image with a face matching the landmarks.
    :param landmarks: Landmarks of the provided frame,
    :return: RGB image with the landmarks as a Pillow Image.
    """
    dpi = 100
    fig = plt.figure(figsize=(frame.shape[0] / dpi, frame.shape[1] / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis('off')
    plt.imshow(np.ones(frame.shape))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Head
    ax.plot(landmarks[0:17, 0], landmarks[0:17, 1], linestyle='-', color='green', lw=2)
    # Eyebrows
    ax.plot(landmarks[17:22, 0], landmarks[17:22, 1], linestyle='-', color='orange', lw=2)
    ax.plot(landmarks[22:27, 0], landmarks[22:27, 1], linestyle='-', color='orange', lw=2)
    # Nose
    ax.plot(landmarks[27:31, 0], landmarks[27:31, 1], linestyle='-', color='blue', lw=2)
    ax.plot(landmarks[31:36, 0], landmarks[31:36, 1], linestyle='-', color='blue', lw=2)
    # Eyes
    ax.plot(landmarks[36:42, 0], landmarks[36:42, 1], linestyle='-', color='red', lw=2)
    ax.plot(landmarks[42:48, 0], landmarks[42:48, 1], linestyle='-', color='red', lw=2)
    # Mouth
    ax.plot(landmarks[48:60, 0], landmarks[48:60, 1], linestyle='-', color='purple', lw=2)

    fig.canvas.draw()
    data = Image.frombuffer('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb(), 'raw', 'RGB', 0, 1)
    plt.close(fig)
    return data

def eval(input_path, output_path, checkpoint_path, model, gpu):
    input = Image.open(input_path)
    input = input.convert("RGB")

    w, h = input.size
    w_, h_ = 128*(w//128), 128*(h//128)

    fa = FaceAlignment(LandmarksType._2D, device="cuda:" + str(gpu))
    landmarks = fa.get_landmarks_from_image(input_path)[0]
    landmark_img = plot_landmarks(np.array(input), landmarks)

    transform_forward = transforms.Compose([
        transforms.Resize((w_, h_)),
        transforms.CenterCrop((w_, h_)),
        transforms.ToTensor()
    ])
    transform_backward = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((w, h)),
        transforms.CenterCrop((w, h)),

    ])

    input = transform_forward(input)
    landmark_img = transform_forward(landmark_img)

    if model == "Pix2Pix":
        NFNet = Pix2Pix()
    else:
        NFNet = DRA()

    checkpoint = torch.load(checkpoint_path)
    NFNet.load_state_dict(checkpoint['my_classifier'])
    NFNet.to(gpu)


    x = torch.cat((input, landmark_img), 0)
    x = x.unsqueeze(0)
    x = x.to(gpu)
    output = NFNet(x)
    output = output.to("cpu")
    output = transform_backward(output[0])
    output.save(output_path)



def main():
    # ARGUMENTS --------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Talking Heads')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    # ARGUMENTS: META_TRAINING  ----------------------------------------------------------------------------------------
    train_parser = subparsers.add_parser("train", help="Starts the training process.")
    train_parser.add_argument("--gt_dataset", type=str, required=True,
                              help="Path to the pre-processed ground-truth image dataset.")
    train_parser.add_argument("--noisy_dataset", type=str, required=True,
                              help="Path to the pre-processed noisy images dataset.")
    train_parser.add_argument("--landmark_dataset", type=str, required=True,
                              help="Path to the pre-processed landmark images dataset.")
    train_parser.add_argument("--gpu", type=int, required=True,
                              help="Run the model on GPU.")
    train_parser.add_argument("--tb_dir", type=str, required=True,
                              help="directory to save the tensorboardx log data")
    train_parser.add_argument("--checkpoint_dir", type=str, required=True,
                              help="directory path to saving pre-trained models")
    train_parser.add_argument("--loss_type", type=str, default=None,
                              help="loss type")
    train_parser.add_argument("--continue_id", type=str, default=None,
                              help="Id of the models to continue training.")

    # ARGUMENTS: FINE_TUNING  ------------------------------------------------------------------------------------------
    tuning_parser = subparsers.add_parser("eval", help="evaluation of NFNet")
    tuning_parser.add_argument("--input_path", type=str, required=True,
                               help="Path to the fine tuning dataset")
    tuning_parser.add_argument("--output_path", type=str, required=True,
                               help="Path to the fine validation dataset")
    tuning_parser.add_argument("--gpu", type=int, required=True,
                               help="Run the model on GPU")
    tuning_parser.add_argument("--checkpoint_path", type=str, required=True,)
    tuning_parser.add_argument("--model", type=str, required=True,
                               help="Id of the models to continue training")

    args = parser.parse_args()

    # EXECUTE ----------------------------------------------------------------------------------------------------------
    try:
        if args.subcommand == "train":
            train(
                gt_path=args.gt_dataset,
                noisy_path=args.noisy_dataset,
                landmark_path=args.landmark_dataset,
                gpu=(torch.cuda.is_available() and args.gpu),
                tb_dir=args.tb_dir,
                checkpoint_dir=args.checkpoint_dir,
                loss_type = args.loss_type
                # continue_id=args.continue_id,
            )
        elif args.subcommand == "eval":
            eval(
                input_path= args.input_path,
                 output_path= args.output_path,
                 checkpoint_path= args.checkpoint_path,
                 gpu= args.gpu,
                 model = args.model,
            )

        else:
            print("invalid command")
    except Exception as e:
        logging.error('Something went wrong: {}'.format(e))
        raise e


if __name__ == '__main__':
    main()
