from PIL import Image
import PIL
from PIL import ImageFilter
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse
import os
from time import time
import logging
import random
from face_alignment import FaceAlignment, LandmarksType


def file_list(folder_path, endwith):
    filelist = [
        os.path.join(path, filename)
        for path, dirs, files in os.walk(folder_path)
        for filename in files
        if filename.endswith(endwith)
    ]

    return filelist


def gausaian_noise(img_path):
    img = cv2.imread(img_path) / 255.0
    noise = np.random.normal(loc=0, scale=1, size=img.shape)

    # noise overlaid over image
    # noisy = np.clip((img + noise * 0.2), 0, 1)
    # noisy2 = np.clip((img + noise * 0.4), 0, 1)

    # noise multiplied by image:
    # whites can go to black but blacks cannot go to white
    # noisy2mul = np.clip((img * (1 + noise * 0.2)), 0, 1)
    # noisy4mul = np.clip((img * (1 + noise * 0.4)), 0, 1)


    noise_coefficient = random.randrange(1, 6)
    noisy2mul = np.clip((img * (1 + noise * 0.01*noise_coefficient )), 0, 1)
    # noisy4mul = np.clip((img * (1 + noise * 0.4)), 0, 1)

    # noise multiplied by bottom and top half images,
    # whites stay white blacks black, noise is added to center
    # img2 = img * 2
    # n2 = np.clip(np.where(img2 <= 1, (img2 * (1 + noise * 0.2)), (1 - img2 + 1) * (1 + noise * 0.2) * -1 + 2) / 2, 0, 1)
    # n4 = np.clip(np.where(img2 <= 1, (img2 * (1 + noise * 0.4)), (1 - img2 + 1) * (1 + noise * 0.4) * -1 + 2) / 2, 0, 1)

    return noisy2mul


def plot_landmarks(size, landmarks):
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
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    ax = fig.add_subplot('111')

    ax.axis('off')
    plt.imshow(np.ones(size))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.set_facecolor('white')

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
    data = PIL.Image.frombuffer('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb(), 'raw', 'RGB', 0, 1)
    plt.close(fig)
    return data



def gt_preprocessing(raw_path, dataset_path):
    Image_files = [
            os.path.join(path, filename)
            for path, dirs, files in os.walk(raw_path)
            for filename in files
            if filename.endswith(".png")
        ]

    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)


    avg = lambda x, n, x0: (x * n/(n+1) + x0 / (n+1), n+1)
    t0 = time()
    n0 = 0
    fpx = 0
    l = len(Image_files)
    for i in range(l):
        img = Image.open(Image_files[i])
        img = img.resize((300, 300), Image.BICUBIC)
        img.save(os.path.join(args.dataset_path, "gt"+Image_files[i].split("/")[-1]))
        # print(os.path.join(args.dataset_path, "gt"+Image_files[i].split("/")[-1]))
        if(i%100 == 0):
            fpx, n0 = avg(fpx, n0, i+1 / (time() - t0))
            done = int(100 * i / l)
            eta = (l - i) / fpx
            print("\r Done: {:03d}%, ETA: {:.2f}s".format(done, eta))


def checkpath(path):
    if not os.path.exists(path):
        os.mkdir(path)


def noisy_preprocess(gt_path, dataset_path):
    gt_train_path = os.path.join(gt_path, "train")
    gt_test_path = os.path.join(gt_path, "test")
    gt_val_path = os.path.join(gt_path, "validation")

    train_path = os.path.join(dataset_path, "train")
    test_path = os.path.join(dataset_path, "test")
    val_path = os.path.join(dataset_path, "validation")

    checkpath(dataset_path)
    checkpath(train_path)
    checkpath(test_path)
    checkpath(val_path)

    train_files = file_list(gt_train_path, ".png")
    for i in train_files:
        img = Image.open(i)

        size = random.randrange(1, 6)
        img2 = img.resize((size*50, size*50), Image.BICUBIC)
        img = img2.resize((300, 300), Image.BICUBIC)
        img.save(os.path.join(train_path, "temp.png"))

        img = gausaian_noise(os.path.join(train_path, "temp.png"))
        img = img * 255
        cv2.imwrite( os.path.join(train_path, "noisy"+i.split("t")[-1]), img)

    os.remove(os.path.join(train_path, "temp.png"))



    test_files = file_list(gt_test_path, ".png")
    for i in test_files:
        img = Image.open(i)

        size = random.randrange(1, 6)
        img2 = img.resize((size * 50, size * 50), Image.BICUBIC)
        img = img2.resize((300, 300), Image.BICUBIC)
        img.save(os.path.join(test_path, "temp.png"))

        img = gausaian_noise(os.path.join(test_path, "temp.png"))
        img = img * 255
        cv2.imwrite(os.path.join(test_path, "noisy"+i.split("t")[-1]), img)

    os.remove(os.path.join(test_path, "temp.png"))



    val_files = file_list(gt_val_path, ".png")
    for i in val_files:
        img = Image.open(i)

        size = random.randrange(1, 6)
        img2 = img.resize((size * 50, size * 50), Image.BICUBIC)
        img = img2.resize((300, 300), Image.BICUBIC)
        img.save(os.path.join(val_path, "temp.png"))

        img = gausaian_noise(os.path.join(val_path, "temp.png"))
        img = img * 255
        cv2.imwrite(os.path.join(val_path, "noisy"+i.split("t")[-1]), img)

    os.remove(os.path.join(val_path, "temp.png"))


def landmark_preprocess(noisy_path, dataset_path):
    noisy_train_path = os.path.join(noisy_path, "train")
    noisy_test_path = os.path.join(noisy_path, "test")
    noisy_val_path = os.path.join(noisy_path, "validation")

    train_path = os.path.join(dataset_path, "train")
    test_path = os.path.join(dataset_path, "test")
    val_path = os.path.join(dataset_path, "validation")

    checkpath(dataset_path)
    checkpath(train_path)
    checkpath(test_path)
    checkpath(val_path)

    fa = FaceAlignment(LandmarksType._2D, device="cuda:1")



    train_files = file_list(noisy_train_path, ".png")
    for i in train_files:
        landmarks = fa.get_landmarks_from_image(i)[0]
        img = plot_landmarks((300, 300, 3), landmarks)
        img.save(os.path.join(train_path, "lm"+i.split("y")[-1]))

    test_files = file_list(noisy_test_path, ".png")
    for i in test_files:
        landmarks = fa.get_landmarks_from_image(i)[0]
        img = plot_landmarks((300, 300, 3), landmarks)
        img.save(os.path.join(test_path, "lm"+i.split("y")[-1]))

    val_files = file_list(noisy_val_path, ".png")
    print(noisy_val_path)
    for i in val_files:
        landmarks = fa.get_landmarks_from_image(i)[0]
        img = plot_landmarks((300, 300, 3), landmarks)
        img.save(os.path.join(val_path, "lm"+ i.split("y")[-1]))


parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

# ARGUMENTS: GROUND TRUTH DATASET PRE-PROCESSING  -------------------------------------------------------------------------------
gt_dataset = subparsers.add_parser("gt_dataset")
gt_dataset.add_argument("--raw_path", type=str, help="path to raw image folder")
gt_dataset.add_argument("--dataset_path", type=str, help="path to ground truth dataset folder")

# ARGUMENTS: NOISY DATASET PRE-PROCESSING  -------------------------------------------------------------------------------
noisy_dataset = subparsers.add_parser("noisy_dataset", help="Pre-process the dataset for its use.")
noisy_dataset.add_argument("--gt_path", type=str, help="path to ground truth dataset folder")
noisy_dataset.add_argument("--dataset_path", type=str, help="path to noisy dataset folder")

# ARGUMENTS: NOISY DATASET PRE-PROCESSING  -------------------------------------------------------------------------------
landmark_dataset = subparsers.add_parser("landmark_dataset", help="Pre-process the dataset for its use.")
landmark_dataset.add_argument("--noisy_path", type=str, help="path to ground truth dataset folder")
landmark_dataset.add_argument("--dataset_path", type=str, help="path to noisy dataset folder")

args = parser.parse_args()


try:
    if args.subcommand == "gt_dataset":
        gt_preprocessing(
            raw_path= args.raw_path,
            dataset_path= args.dataset_path,
        )
    elif args.subcommand == "noisy_dataset":
        noisy_preprocess(
            gt_path= args.gt_path,
            dataset_path= args.dataset_path,
        )

    elif args.subcommand == "landmark_dataset":
        landmark_preprocess(
            noisy_path=args.noisy_path,
            dataset_path=args.dataset_path,
        )

    else:
        print("invalid command")
except Exception as e:
    # logging.error(f'Something went wrong: {e}')
    logging.error('Something went wrong: {}'.format(e))
    raise e









# for i in range(10):
#     blur = i
#     img = Image.open("/Users/choichangho/Desktop/00041.png")
#     img = img.filter(ImageFilter.GaussianBlur(blur))
#
#     img.save(f"/Users/choichangho/Desktop/00041_blur{blur}.png")





#
#
#
# for i in range(10):
#     img = gausaian_noise(f"/Users/choichangho/Desktop/00041_blur{i}.png")
#     img = img*255
#     cv2.imwrite( f"/Users/choichangho/Desktop/00041_noise{i}.png", img)



# for i in range(2, 7):
#     size = i
#     img = Image.open("/Users/choichangho/Desktop/00041.png")
#     resize_image = img.resize((1024//(1<<i), 1024//(1<<i)), Image.ANTIALIAS)
#     resize_image = resize_image.resize((1024, 1024), Image.ANTIALIAS)
#     resize_image.save(f"/Users/choichangho/Desktop/00041_resize_anti{size}.png")

# for i in range(2, 7):
#     size = i
#     img = Image.open("/Users/choichangho/Desktop/00041.png")
#     resize_image = img.resize((1024//(1<<i), 1024//(1<<i)), Image.BICUBIC)
#     resize_image.save(f"/Users/choichangho/Desktop/00041_bicubic{size}.png")
#     output_1 = gausaian_noise(f"/Users/choichangho/Desktop/00041_bicubic{size}.png")
#     output_1 = output_1*255
#     cv2.imwrite(f"/Users/choichangho/Desktop/00041_bignoise{i}.png", output_1)
#     output_1 = Image.open(f"/Users/choichangho/Desktop/00041_bignoise{i}.png")
#     output_1 = output_1.resize((1024, 1024), Image.BICUBIC)
#     output_1.save(f"/Users/choichangho/Desktop/00041_bignoise{i}.png")
#
#
#     resize_image = resize_image.resize((1024, 1024), Image.BICUBIC)
#     resize_image.save(f"/Users/choichangho/Desktop/00041_bicubic{size}.png")
#     output_2 = gausaian_noise(f"/Users/choichangho/Desktop/00041_bicubic{size}.png")
#     output_2 = output_2 *255
#
#
#     # cv2.imwrite(f"/Users/choichangho/Desktop/00041_bignoise{i}.png", output_1)
#     cv2.imwrite(f"/Users/choichangho/Desktop/00041_smallnoise{i}.png", output_2)

#     img = gausaian_noise(f"/Users/choichangho/Desktop/00041_blur{i}.png")
#     img = img*255
#     cv2.imwrite( f"/Users/choichangho/Desktop/00041_noise{i}.png", img)
    