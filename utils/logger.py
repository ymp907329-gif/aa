import os
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def image(self, tag, img, step, nrow=4):
        self.writer.add_images(tag, img, step)

    def close(self):
        self.writer.close()