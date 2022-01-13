import tensorflow as tf
from stable_baselines3.common.callbacks import BaseCallback
import os
from glob import glob

class DummyLogger:
    def __init__(self, log_dir):
        pass

    def write_metadata(self, epoch, key, value):
        print("Epoch ", epoch, " ", key, ":", value)

class TensorboardLoggerSimple:
    def __init__(self, log_dir, run_name="run", run_id=None):
        folders = [folder_name for folder_name in os.listdir(log_dir) if folder_name.startswith(run_name)]
        total_runs_so_far = len(folders)

        if run_id is None:
            self.run_id = f"{run_name}_{total_runs_so_far + 1}"
        else:
            self.run_id = run_id

        print("Logdir: ", self.run_id)

        self.tb_logger = tf.summary.create_file_writer(f"{log_dir}/{self.run_id}")

    def write_metadata(self, epoch, key, value):
        with self.tb_logger.as_default():
            tf.summary.scalar(key, value, step=epoch)

if __name__ == "__main__":
    logger = TensorboardLoggerSimple(log_dir="test_log")

    for i in range(10 + 1):
        logger.write_metadata(epoch=i, key="accuracy", value=i*10)