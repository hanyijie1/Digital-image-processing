from .module_3 import *
import seaborn as sns

class Module4(Module3):
    """
    This class implements the module 4: Pretreatment and post-treatment methods.

    methods:
    section1_part2: filter to binary case 1. there is no param such as replicate in python.
    section2_part3: base on filtered binary case 1 and binary case 1, cut the background.
    section2_part4: pre-process and row sum compare between case_2_close_typeface and case2_opened_mask.
    """
    def __init__(self, ):
        super().__init__()
        # path
        self.module_4_section1_part2_path = os.path.join(
            self.processed_graph_path,
            "module4_section1_part2.png",
        )
        self.module_4_section2_part3_path = os.path.join(
            self.processed_graph_path,
            "module4_section2_part3.png",
        )
        self.module_4_section2_part4_path = os.path.join(
            self.processed_graph_path,
            "module4_section2_part4.png",
        )
        # target
        self.case1_re_binary_gray =  np.zeros_like(self.case1_gray)
        self.case2_rm_noise_gray = np.zeros_like(self.case2_gray)
        self.case2_opened_mask = np.zeros_like(self.case2_gray)

    def module4_section1_part2(self, ):  # notice: you must run .module3_section1_part2(self, ) firstly.
        case1_filtered_gray = cv2.blur(  # filter
             self.case1_binary_gray,
            (3, 3),
        )
        self.case1_re_binary_gray = cv2.adaptiveThreshold(  # binary again
            src=case1_filtered_gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=5,
            C=1
        )
        # graph
        fig = plt.figure(figsize=(12, 6))
        ax = [fig.add_subplot(1, 3, i) for i in range(1, 3 + 1)]
        ax[0].imshow(
            self.case1_binary_gray,
            cmap="gray",
        )
        ax[1].imshow(
            case1_filtered_gray,
            cmap="gray",
        )
        ax[2].imshow(
            self.case1_re_binary_gray,
            cmap="gray",
        )
        ax[0].set_title("primal binary case 1")
        ax[1].set_title("filter binary case 1")
        ax[2].set_title("binary filter binary case 1")
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        plt.savefig(self.module_4_section1_part2_path)
        plt.show()
        print("这到底是清楚还是模糊了就很难说了。")

    def module4_section2_part3(self, ):
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        case1_background_h_gray = cv2.morphologyEx(
            self.case1_re_binary_gray,
            cv2.MORPH_CLOSE,
            kernel_h,
        )
        case1_background_v_gray = cv2.morphologyEx(
            self.case1_re_binary_gray,
            cv2.MORPH_CLOSE,
            kernel_v,
        )
        case1_background_gray = cv2.bitwise_and(
            case1_background_h_gray,
            case1_background_v_gray
        )
        case1_background_unfiltered_gray = cv2.morphologyEx(
            self.case1_binary_gray,
            cv2.MORPH_CLOSE,
            kernel_rect,
        )
        case1_rm_background_inv_gray = case1_background_gray - self.case1_re_binary_gray
        case1_rm_noise_gray = cv2.bitwise_not(case1_rm_background_inv_gray)
        case1_unfiltered_rm_background_inv = case1_background_unfiltered_gray - self.case1_binary_gray
        case1_unfiltered_rm_noise_gray = cv2.bitwise_not(case1_unfiltered_rm_background_inv)
        # graph
        fig = plt.figure(figsize=(12, 12))
        ax = [fig.add_subplot(2, 3, i) for i in range(1, 6 + 1)]
        ax[0].imshow(
            self.case1_re_binary_gray,
            cmap="gray",
        )
        ax[1].imshow(
            case1_background_gray,
            cmap="gray",
        )
        ax[2].imshow(
            case1_rm_noise_gray,
            cmap="gray",
        )
        ax[3].imshow(
            self.case1_binary_gray,
            cmap="gray",
        )
        ax[4].imshow(
            case1_background_unfiltered_gray,
            cmap="gray",
        )
        ax[5].imshow(
            case1_unfiltered_rm_noise_gray,
            cmap="gray",
        )
        ax[0].set_title("binary blurred binary case 1")
        ax[1].set_title("background")
        ax[2].set_title("typeface")
        ax[3].set_title("binary case 1")
        ax[4].set_title("background")
        ax[5].set_title("typeface")
        for i in range(6):
            ax[i].axis('off')
        plt.savefig(self.module_4_section2_part3_path)
        plt.show()

    def module4_section2_part4(self, ):
        # closed operation
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        case2_background_gray = cv2.morphologyEx(
            self.case2_binary_gray,
            cv2.MORPH_CLOSE,
            kernel_rect,
        )
        case2_rm_noise_inv_gray = case2_background_gray - self.case2_binary_gray
        self.case2_rm_noise_gray = cv2.bitwise_not(case2_rm_noise_inv_gray)
        # opened operation
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        self.case2_opened_mask = cv2.morphologyEx(
            self.case2_rm_noise_gray,
            cv2.MORPH_OPEN,
            kernel_rect,
        )
        # row sum compare
        case2_rm_noise_row_sum_pd ={
            "column number": np.arange(0, np.shape(self.case2_rm_noise_gray)[0], 1),
            "sum values": np.sum(self.case2_rm_noise_gray, 1),
        }
        case2_open_mask_row_sum_pd ={
            "column number": np.arange(0, np.shape(self.case2_opened_mask)[0], 1),
            "sum values": np.sum(self.case2_opened_mask, 1),
        }
        # graph
        fig = plt.figure(figsize=(12, 8))
        ax = [fig.add_subplot(2, 3, i) for i in range(1, 6 + 1)]
        ax[0].imshow(
            self.case2_binary_gray,
            cmap="gray",
        )
        ax[1].imshow(
            case2_background_gray,
            cmap="gray",
        )
        ax[2].imshow(
            self.case2_rm_noise_gray,
            cmap="gray",
        )
        ax[3].imshow(
            self.case2_opened_mask,
            cmap="gray",
        )
        sns.lineplot(
            data=case2_rm_noise_row_sum_pd,
            x="column number",
            y="sum values",
            label="row sum of close_typeface",
            ax=ax[4],
        )
        sns.lineplot(
            data=case2_open_mask_row_sum_pd,
            x="column number",
            y="sum values",
            label="row sum of open_mask",
            ax=ax[4],
        )
        ax[0].set_title("binary blurred binary case 1")
        ax[1].set_title("background")
        ax[2].set_title("typeface")
        ax[3].set_title("opened mask")
        ax[4].set_title(
            "row sum compare: \n case_2_close_typeface and case2_opened_mask",
            fontsize=10,
        )
        for i in range(4):
            ax[i].axis('off')
        ax[5].axis('off')
        plt.savefig(self.module_4_section2_part4_path)
        plt.show()