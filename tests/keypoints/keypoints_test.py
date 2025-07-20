import unittest
import cv2
import omnialigner as om

class TestDeeperHistRegNonrigid(unittest.TestCase):
    def test_affine(self):
        image_F = cv2.imread("./F.png")
        image_M = cv2.imread("./M.png")
        tensor_F = om.tl.im2tensor(image_F)
        tensor_M = om.tl.im2tensor(image_M)

        kd = om.kp.detect_AngleFlipScale(tensor_F, tensor_M, detector=None)
        kd.dataset['image_label'] = tensor_F
        kd.dataset['image_input'] = tensor_M
        fig = kd.plot_dataset()
        fig.savefig("./tmp.png")

    def test_move_with_tfrs(self):
        image_F = cv2.imread("./F.png")
        image_M = cv2.imread("./M.png")
        tensor_F = om.tl.im2tensor(image_F)
        tensor_M = om.tl.im2tensor(image_M)

        kd = om.kp.detect_AngleFlipScale(tensor_F, tensor_M, detector=None)
        kd.dataset['image_label'] = tensor_F
        kd.dataset['image_input'] = tensor_M

        tensor_tfrs = kd.calculate_tfrs()

        img_M_moved = kd.move_img_M(tensor_tfrs=tensor_tfrs)
        kpt_F = kd.dataset["train_label"]
        kpt_M = kd.move_kpt_M(tensor_tfrs=tensor_tfrs)
        plt = om.pl.plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img_M_moved)
        ax.scatter(kpt_M[:, 0], kpt_M[:, 1], c='r', label='F')
        fig.savefig("./tmp_moved.png")

if __name__ == "__main__":
    unittest.main()
