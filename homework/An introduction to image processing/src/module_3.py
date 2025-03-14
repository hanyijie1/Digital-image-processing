from .module_2 import *

class Module3(Module2):
    """
    This class implements the module 3: split images.

    methods:
    module3_section1_part1()
        Binary, the threshold is 200.
    module3_section1_part2()
        auto-binary, local auto-binary and foreground reversal.
    module3_section2_part1()
        sum to column(row sum) to case 1 and case 2.
    """
    def __init__(self):
        super().__init__()
        """
        Parameters
        ----------
        case{}_binary_gray : np.ndarray
        """
        # path
        self.module3_section1_part1_graph_path = os.path.join(
            self.processed_graph_path,
            "module3_section1_part1.png",
        )
        self.module3_section1_part2_graph_path = os.path.join(
            self.processed_graph_path,
            "module3_section1_part2.png",
        )
        self.module3_section2_part1_graph_path = os.path.join(
            self.processed_graph_path,
            "module3_section2_part1.png",
        )
        # target
        self.case1_binary_gray = np.zeros_like(self.case1_gray)
        self.case2_binary_gray = np.zeros_like(self.case2_gray)

    def module3_section1_part1(self, ):
        print("We have plot the hist image of case 1 on module 2 section 3 part 1, we won't go into details here.")
        # binary
        threshold = 200
        case1_gray_binary_matrix = self.case1_gray
        case1_gray_binary_matrix[case1_gray_binary_matrix > threshold] = 255
        case1_gray_binary_matrix[case1_gray_binary_matrix < threshold] = 0
        # display
        plt.figure(figsize=(8, 4))
        plt.axis('off')
        plt.imshow(
            case1_gray_binary_matrix,
            cmap='gray',
        )
        plt.title('Binary of case 1, the threshold is {}.'.format(threshold))
        plt.savefig(self.module3_section1_part1_graph_path)
        plt.show()

    def module3_section1_part2(self, ):
        # auto-binary
        _, auto_binary_case_1_matrix = cv2.threshold(
            src=self.case1_gray,
            thresh=0,
            maxval=255,
            type=cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )
        # local binary
        self.case1_binary_gray = cv2.adaptiveThreshold(  # case 1
            src=self.case1_gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=101,
            C=8
        )
        self.case2_binary_gray = cv2.adaptiveThreshold(  # case 1
            src=self.case2_gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=101,
            C=30
        )
        local_binary_inv_case_1_matrix =cv2.adaptiveThreshold(
            src=self.case1_gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=101,
            C=8
        )
        # graph
        fig = plt.figure(figsize=(12, 4))
        ax = [fig.add_subplot(1, 3, i) for i in range(1, 3 + 1)]
        ax[0].imshow(
            auto_binary_case_1_matrix,
            cmap="gray",
        )
        ax[1].imshow(
            self.case1_binary_gray,
            cmap="gray",
        )
        ax[2].imshow(
            local_binary_inv_case_1_matrix,
            cmap="gray",
        )
        ax[0].set_title("auto-binary")
        ax[1].set_title("local-binary")
        ax[2].set_title("local-binary-INV")
        for i in range(3):
            ax[i].axis('off')
        plt.savefig(self.module3_section1_part2_graph_path)
        plt.show()
        print("python好像不需要INV")

    def module3_section2_part1(self, ):
        row_sum_case_1_vec = np.sum(self.case1_binary_gray, axis=1)
        row_sum_case_2_vec = np.sum(self.case2_binary_gray, axis=1)
        # graph
        fig = plt.figure(figsize=(8, 3))
        ax = [fig.add_subplot(1, 2, i) for i in range(1, 2 + 1)]
        ax[0].plot(
            row_sum_case_1_vec,
            label="row sum case 1 local-binary-INV",
        )
        ax[1].plot(
            row_sum_case_2_vec,
            label="row sum case 2 local-binary-INV",
        )
        ax[0].set_title("row sum case 1 local-binary-INV")
        ax[1].set_title("row sum case 2 local-binary-INV")
        plt.savefig(self.module3_section2_part1_graph_path)
        plt.show()