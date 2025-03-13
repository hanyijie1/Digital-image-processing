import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


class Module2:
    """
    This class implements the module 2: process images in matlab.

    methods:
    section1_part1: display and save the case images.
    section2_part2: display red tier and find its max, min value and shape.
    section2_part3: transform of bgr to gray scale and return the shape of gray scale.
    section3_part1: contrast adjustment and display via histogram.
    section3_part2: 交互式Python没有，不写了。
    """
    def __init__(self, ):
        # path
        self.raw_graph_folder_path = "./graph/raw"  # raw
        self.case1_png_path = os.path.join(
            self.raw_graph_folder_path,
            "Screenshot_20250310_223427.png",
        )
        self.case2_png_path = os.path.join(
            self.raw_graph_folder_path,
            "Screenshot_20250310_223929.png",
        )
        self.processed_graph_path = "./graph/processed"  # processed
        self.module2_section1_part1_graph_path = os.path.join(
            self.processed_graph_path,
            "module2_section1_part1.png",
        )
        self.module2_section2_part2_graph_path = os.path.join(
            self.processed_graph_path,
            "module2_section2_part2.png",
        )
        self.module2_section2_part3_graph_path = os.path.join(
            self.processed_graph_path,
            "module2_section2_part3.png",
        )
        self.module2_section3_part1_graph_path = os.path.join(
            self.processed_graph_path,
            "module2_section3_part1.png",
        )
        # target
        self.case1_bgr = cv2.imread(self.case1_png_path, 1)  # raw
        self.case2_bgr = cv2.imread(self.case2_png_path, 1)
        self.case1_rgb = cv2.cvtColor(self.case1_bgr, cv2.COLOR_BGR2RGB)
        self.case2_rgb = cv2.cvtColor(self.case2_bgr, cv2.COLOR_BGR2RGB)
        self.case1_gray = cv2.cvtColor(
            self.case1_bgr,
            cv2.COLOR_BGR2GRAY,
        )
        self.case2_gray = cv2.cvtColor(
            self.case2_bgr,
            cv2.COLOR_BGR2GRAY,
        )
        self.case1_contrast_gray = np.zeros_like(self.case1_bgr)  # processed
        self.case2_contrast_gray = np.zeros_like(self.case2_bgr)

    def module2_section1_part1(self, ):
        # display
        ax = [0, 0, ]
        fig = plt.figure(figsize=(8, 4))
        ax[0] = fig.add_subplot(1, 2, 1)
        ax[1] = fig.add_subplot(1, 2, 2)
        ax[0].axis("off")
        ax[1].axis("off")
        ax[0].imshow(
            self.case1_rgb,
        )
        ax[1].imshow(
            self.case2_rgb,
        )
        ax[0].set_title("section1_part1: display and save")
        # save
        plt.savefig(self.module2_section1_part1_graph_path)
        plt.show()

    def module2_section2_part2(self, ):
        # compute
        case_1_red_tier_matrix = self.case1_rgb[:, :, 0]
        case_1_red_tier_arr = np.zeros_like(self.case1_rgb)
        case_1_red_tier_arr[:,:,0] = case_1_red_tier_matrix  # use to plot rgb
        row_count, column_count = np.shape(case_1_red_tier_matrix)
        max_bin = np.max(case_1_red_tier_matrix)
        min_bin = np.min(case_1_red_tier_matrix)
        # graph
        ax = [0, 0, ]
        fig = plt.figure(figsize=(8, 4))
        ax[0] = fig.add_subplot(1, 2, 1)
        ax[1] = fig.add_subplot(1, 2, 2)
        ax[0].axis("off")
        ax[1].axis("off")
        ax[0].imshow(
            self.case1_rgb,
        )
        ax[1].imshow(
            case_1_red_tier_arr,
        )
        ax[0].set_title("section2_part2")
        plt.savefig(self.module2_section2_part2_graph_path)
        plt.show()
        # print compute info
        print("The shape of red matrix is {}*{}, and max bin is {}, min bin is {}"
              .format(row_count, column_count, max_bin, min_bin))

    def module2_section2_part3(self, ):
        # transform
        row_count, column_count = np.shape(self.case2_gray)
        # graph
        ax = [0, 0, ]
        fig = plt.figure(figsize=(8, 4))
        ax[0] = fig.add_subplot(1, 2, 1)
        ax[1] = fig.add_subplot(1, 2, 2)
        ax[0].axis("off")
        ax[1].axis("off")
        ax[0].imshow(
            self.case2_rgb,
        )
        ax[1].imshow(
            self.case2_gray,
            cmap="gray",
        )
        ax[0].set_title("section2_part3")
        plt.savefig(self.module2_section2_part3_graph_path)
        plt.show()
        # print compute info
        print("The shape of red matrix is {}*{}.".format(row_count, column_count))

    def module2_section3_part1(self, ):
        # primal histogram
        case_1_hist_vec = cv2.calcHist(
            [self.case1_gray],
            [0],
            None,
            [256],
            [0, 256],
        )
        case_2_hist_vec = cv2.calcHist(
            [self.case2_gray],
            [0],
            None,
            [256],
            [0, 256],
        )
        # adjust hist
        self.case1_contrast_gray = cv2.equalizeHist(self.case1_gray)
        self.case2_contrast_gray = cv2.equalizeHist(self.case2_gray)
        case_1_contrast_adjust_hist_vec = cv2.calcHist(
            [self.case1_contrast_gray],
            [0],
            None,
            [256],
            [0, 256],
        )
        case_2_contrast_adjust_hist_vec = cv2.calcHist(
            [self.case2_contrast_gray],
            [0],
            None,
            [256],
            [0, 256],
        )
        # graph
        fig = plt.figure(figsize=(10, 16))
        ax=[fig.add_subplot(4, 2, i + 1) for i in range(8)]
        ax[0].set_title("primal case 1")
        ax[1].set_title("primal hist of case 1")
        ax[2].set_title("primal case 2")
        ax[3].set_title("primal hist of case 2")
        ax[4].set_title("processed case 1")
        ax[5].set_title("processed hist of case 1")
        ax[6].set_title("processed case 2")
        ax[7].set_title("processed hist of case 2")
        for i in range(1, 8 + 1, 2):
            ax[i].set_ylabel("# of Pixels")
        ax[-1].set_xlabel("Bins")
        for i in range(0, 8, 2):
            ax[i].axis('off')
        ax[0].imshow(
            self.case1_gray,
            cmap="gray",
        )
        ax[1].plot(
            case_1_hist_vec,
            label="primal case 1",
        )
        ax[2].imshow(
            self.case2_gray,
            cmap="gray",
        )
        ax[3].plot(
            case_2_hist_vec,
            label="primal case 2",
        )
        ax[4].imshow(
            self.case1_contrast_gray,
            cmap="gray",
        )
        ax[5].plot(
            case_1_contrast_adjust_hist_vec,
            label="processed case 1",
        )
        ax[6].imshow(
            self.case2_contrast_gray,
            cmap="gray",
        )
        ax[7].plot(
            case_2_contrast_adjust_hist_vec,
            label="processed case 2",
        )
        for i in range(1, 8, 2):
            ax[i].legend()
        # save
        plt.savefig(self.module2_section3_part1_graph_path)
        plt.show()