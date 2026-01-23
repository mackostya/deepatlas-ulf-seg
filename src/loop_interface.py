import matplotlib.pyplot as plt
import gc
import src.visualizations as vis

from abc import ABC


class ModelInterface(ABC):
    # TODO: Remove this interface if not needed anymore
    def __init__(self):
        super().__init__()
        self.tb_logger = None

    def log_images(self, img, labels, preds, seg_type):
        fig = vis.vis_segmentation_volume_per_type(
            img,
            labels,
            preds,
            seg_type=seg_type,
        )
        self.tb_logger.add_figure(f"Image/Segmentation_{seg_type}", fig, global_step=self.current_epoch)

        plt.close(fig)
        del fig
        gc.collect()

    def log_tb_histograms(self) -> None:
        """
        Interface for logging histograms to tensorboard.
        :return:
        """
        for block in self.model.image_encoder.trunk.blocks:
            for name, weight in block.attn.qkv.named_parameters():
                self.tb_logger.add_histogram(name, weight, self.current_epoch)
                if weight.grad is not None:
                    self.tb_logger.add_histogram(name + "_grad", weight.grad, self.current_epoch)
            for name, weight in block.attn.qkv.named_parameters():
                self.tb_logger.add_histogram(name, weight, self.current_epoch)
                if weight.grad is not None:
                    self.tb_logger.add_histogram(name + "_grad", weight.grad, self.current_epoch)
