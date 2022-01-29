import tensorflow as tf

class gannLogger:
    def __init__(self,logDir):
        self.losses = {}
        self.writer = tf.summary.create_file_writer(logDir)

    def updateLosses(self,losses):
        for k,v in losses.items():
            if not k in self.losses:
                self.losses[k] = tf.keras.metrics.Mean(k, dtype=tf.float32)
            self.losses[k](v)

    def dumpLosses(self,epoch):
        with self.writer.as_default():
            for l,v in self.losses.items():
                tf.summary.scalar(l, v.result(), step=epoch)
                v.reset_states()