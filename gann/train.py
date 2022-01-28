import os
from   time import time

import tensorflow as tf

from gann.generator     import makeGeneratorModel, generatorLoss
from gann.discriminator import makeDiscriminatorModel, discriminatorLoss
from gann.optimizers    import generatorOptimizer, discriminatorOptimizer
from gann.vis           import getVisualizer

from settings import settings

from util.util import Timer

generator     = makeGeneratorModel()
discriminator = makeDiscriminatorModel()

def trainStep(images, meanData, inputData):
    with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
      generatedImage = generator( inputData, training = True )

      discReal = discriminator( images,         training = True)
      discFake = discriminator( generatedImage, training = True)

      discLoss = discriminatorLoss( discReal, discFake )
      genLoss  = generatorLoss( discFake, generatedImage, meanData )

      gradientsGenerator     = genTape.gradient(  genLoss,  generator.trainable_variables )
      gradientsDiscriminator = discTape.gradient( discLoss, discriminator.trainable_variables )

      generatorOptimizer.apply_gradients(     zip( gradientsGenerator,     generator.trainable_variables))
      discriminatorOptimizer.apply_gradients( zip( gradientsDiscriminator, discriminator.trainable_variables))

def getCheckpoint():
  checkpointDir    = settings.train["output"]
  checkpointPrefix = os.path.join( checkpointDir, "ckpt")

  return tf.train.Checkpoint( generatorOptimizer      = generatorOptimizer,
                              discriminatorOptimizer  = discriminatorOptimizer,
                              generator               = generator,
                              discriminator           = discriminator),checkpointPrefix

def train(dataset, epochs = settings.train["epochs"]):
  visualizer         = getVisualizer()
  checkpoint, prefix = getCheckpoint()

  for epoch in range(epochs):
    with Timer('Time for epoch {epoch} is {t} sec',{"epoch":epoch}):
      start = time()
      for images, meanData, inputData in dataset:
        trainStep(images, meanData, inputData)

    if epoch % settings.train["checkpointFrequency"] == 0:
      visualizer(generator,epoch)
      checkpoint.save(file_prefix = prefix)
