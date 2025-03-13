from .module_2 import *
from scipy.signal import argrelmin
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

class Module5(Module2):
    """
    This class implements the module 5: sorting and batch processing.

    methods:
    get_image_info(img_bgr: np.ndarray, ) -> np.ndarray, np.ndarray, np.ndarray: get vec_smoothed, local_min_indices and local_min_values of img_bgr.
    get_local_min(vec: np.ndarray, target_count: np.int32) -> np.int32: get local minimum of vec through method self-defined.
    get_local_min_count(vec: np.ndarray) -> np.int32: get local minimum counts ergodic params.
    judgment_local_min_count(image_bgr: np.ndarray) -> np.int32: sum up get_image_info(img_bgr) and get_local_min_count(img_bgr)
    module5_section1_part1(self, ): find local minimum of case1 and case2.
    module5_section1_part32(self, ): judgment of is_receipt in images list.
    """
    def __init__(self):
        super().__init__()
        # path
        self.case3_png_path = os.path.join(
            self.raw_graph_folder_path,
            "Screenshot_20250313_105013.png"
        )
        self.case4_png_path = os.path.join(
            self.raw_graph_folder_path,
            "Screenshot_20250313_105235.png"
        )
        self.case5_png_path = os.path.join(
            self.raw_graph_folder_path,
            "下载.jpeg"
        )
        self.module5_section1_part1_path = os.path.join(
            self.processed_graph_path,
            "module5_section1_part1.png"
        )
        self.module5_section1_part34_path = os.path.join(
            self.processed_graph_path,
            "module5_section1_part34.png"
        )
        # raw
        self.case3_bgr = cv2.imread(self.case3_png_path, 1)
        self.case4_bgr = cv2.imread(self.case4_png_path, 1)
        self.case5_bgr = cv2.imread(self.case5_png_path, 1)
        self.case3_rgb = cv2.cvtColor(self.case3_bgr, cv2.COLOR_BGR2RGB)
        self.case4_rgb = cv2.cvtColor(self.case4_bgr, cv2.COLOR_BGR2RGB)
        self.case5_rgb = cv2.cvtColor(self.case5_bgr, cv2.COLOR_BGR2RGB)
        self.images_rgb_list = [
            self.case1_rgb,
            self.case2_rgb,
            self.case3_rgb,
            self.case4_rgb,
            self.case5_rgb,
        ]
        self.images_rgb_sorted_list = self.images_rgb_list

    @staticmethod
    def get_image_info(img_bgr):  # please adjust param in this function.
        img_gra = cv2.cvtColor(  # binary
            img_bgr,
            cv2.COLOR_BGR2GRAY,
        )
        img_binary = cv2.adaptiveThreshold(
            src=img_gra,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=101,
            C=20
        )
        img_filtered = cv2.blur(  # filter
            img_binary,
            (3, 3),
        )
        imag_burred = cv2.adaptiveThreshold(  # binary again
            src=img_filtered,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=5,
            C=10
        )
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))  # close
        closed_tool = cv2.morphologyEx(
            imag_burred,
            cv2.MORPH_CLOSE,
            kernel_rect,
        )
        imag_closed_inv = closed_tool - imag_burred
        imag_closed = cv2.bitwise_not(imag_closed_inv)
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))  # open
        opened_tool = cv2.morphologyEx(
            imag_closed,
            cv2.MORPH_OPEN,
            kernel_rect,
        )
        opened_tool_row_sum_vec = np.sum(opened_tool, 1)
        return imag_closed, opened_tool, opened_tool_row_sum_vec

    @staticmethod
    def get_local_min(vec, target_count, ):
        vec_smoothed = gaussian_filter1d(vec, sigma=3)
        local_min_indices, local_min_values = None, None
        for i in range(10, 100):
            local_min_indices = argrelmin(vec_smoothed, order=i)[0]
            if np.size(local_min_indices) <= target_count:
                local_min_values = vec_smoothed[local_min_indices]
                break
        return vec_smoothed, local_min_indices, local_min_values

    @staticmethod
    def get_local_min_count(vec, ):
        local_min_count = 0
        for i in range(29900, 30000):
            for j in range(40, 50):
                min_indices, _ = find_peaks(
                    -vec,
                    prominence=(i, None),
                    wlen=j,
                )
                local_min_count += np.count_nonzero(min_indices)
        return local_min_count

    def judgment_local_min_count(self, img_bgr):
        _, _, opened_tool_row_sum_vec = self.get_image_info(img_bgr)
        receipt_possibility = self.get_local_min_count(opened_tool_row_sum_vec)
        return receipt_possibility

    def module5_section1_part1(self, ):
        # compute
        case3_closed_matrix, case3_opened_tool_matrix, case3_opened_tool_row_sum_vec \
            = self.get_image_info(self.case3_bgr)
        case4_closed_matrix, case4_opened_tool_matrix, case4_opened_tool_row_sum_vec \
            = self.get_image_info(self.case4_bgr)
        case3_opened_tool_row_sum_smoothed_vec, case3_local_min_indices, case3_local_min_values \
            = self.get_local_min(case3_opened_tool_row_sum_vec, 9)
        case4_opened_tool_row_sum_smoothed_vec, case4_local_min_indices, case4_local_min_values \
            = self.get_local_min(case4_opened_tool_row_sum_vec, 12)
        # graph
        fig = plt.figure(figsize=(12, 9))
        ax = [fig.add_subplot(2, 3, i) for i in range(1, 2 * 3 + 1)]
        ax[0].imshow(
            self.case3_rgb,
        )
        ax[1].imshow(
            case3_opened_tool_matrix,
            cmap="gray",
        )
        ax[2].plot(
            case3_opened_tool_row_sum_vec,
            label="opened tool row sum",
        )
        ax[2].plot(
            case3_opened_tool_row_sum_smoothed_vec,
            label="smoothed opened tool row sum",
        )
        ax[2].plot(
            case3_local_min_indices,
            case3_local_min_values,
            label="local min",
            linewidth=0,
            marker='o',
        )
        ax[3].imshow(
            self.case4_rgb,
        )
        ax[4].imshow(
            case4_opened_tool_matrix,
            cmap="gray",
        )
        ax[5].plot(
            case4_opened_tool_row_sum_vec,
            label="opened tool row sum",
        )
        ax[5].plot(
            case4_opened_tool_row_sum_smoothed_vec,
            label="smoothed opened tool row sum",
        )
        ax[5].plot(
            case4_local_min_indices,
            case4_local_min_values,
            label="local min",
            linewidth=0,
            marker='o',
        )
        ax[2].legend(fontsize=8)
        ax[5].legend(fontsize=8)
        ax[0].axis("off")
        ax[1].axis("off")
        ax[3].axis("off")
        ax[4].axis("off")
        plt.savefig(self.module5_section1_part1_path)
        plt.show()

    def module5_section1_part34(self, ):
        # compute
        receipt_possibility_list = []
        for i in self.images_rgb_list:
            receipt_possibility_list.append(self.judgment_local_min_count(i))
        receipt_possibility_vec = np.array(receipt_possibility_list)
        receipt_possibility_vec = ((receipt_possibility_vec - np.min(receipt_possibility_vec))
                           / (np.max(receipt_possibility_vec) - np.min(receipt_possibility_vec)))  # normal
        print("To sum up, the compared possibilities images are receipts: ")
        print(receipt_possibility_vec)
        # display
        print("the possibility sequence of is_receipt images display: ")
        receipt_possibility_vec_indices = np.argsort(- receipt_possibility_vec)
        self.images_rgb_sorted_list = [self.images_rgb_list[i] for i in receipt_possibility_vec_indices]
        fig = plt.figure(figsize=(6 * np.size(receipt_possibility_vec), 6))
        ax = [fig.add_subplot(1, np.size(receipt_possibility_vec), i) for i in range(1, np.size(receipt_possibility_vec) + 1)]
        for i in range(np.size(receipt_possibility_vec_indices)):
            ax[i].set_title("possibility rank {}".format(i))
            ax[i].imshow(
                self.images_rgb_sorted_list[i],
            )
            ax[i].axis('off')
        plt.savefig(self.module5_section1_part34_path)
        plt.show()





