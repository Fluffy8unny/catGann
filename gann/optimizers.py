import tensorflow as tf

crossEntropy           = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generatorOptimizer     = tf.keras.optimizers.Adam(1e-4)
discriminatorOptimizer = tf.keras.optimizers.Adam(1e-4)
