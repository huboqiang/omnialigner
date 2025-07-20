from PIL import Image
import torch
import cv2
from romatch import roma_outdoor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/sacre_coeur_A.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/sacre_coeur_B.jpg", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path

    # Create model
    roma_model = roma_outdoor(device=device)


    W_A, H_A = Image.open(im1_path).size
    W_B, H_B = Image.open(im2_path).size

    # Match
    warp, certainty = roma_model.match(im1_path, im2_path, device=device)
    # Sample matches for estimation
    matches, certainty = roma_model.sample(warp, certainty)
    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)    
    F, mask = cv2.findFundamentalMat(
        kpts1.cpu().numpy(), kpts2.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
    )
    import numpy as np
    import matplotlib.pyplot as plt
    im_A = Image.open(im1_path)
    im_A = np.array(im_A.convert("RGB"))
    im_B = Image.open(im2_path)
    im_B = np.array(im_B.convert("RGB"))
    torch.save({
        "im_A": im_A,
        "im_B": im_B,
        "kpts1": kpts1,
        "kpts2": kpts2,
        "F": F,
        "mask": mask
    }, "./matches.pt")

    kpts1 = kpts1.cpu().numpy()
    kpts2 = kpts2.cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(im_A)
    ax.scatter(kpts1[:, 0], kpts1[:, 1], c='r', s=0.2)
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(im_B)
    ax.scatter(kpts2[:, 0], kpts2[:, 1], c='r', s=0.2)
    plt.savefig("./matches.png")