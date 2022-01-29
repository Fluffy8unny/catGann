import os
from   time import time

import tensorflow as tf
from gann.discriminator import makeDiscriminatorModel, discriminatorLoss
from gann.generator     import makeGeneratorModel, generatorLoss
from gann.logging       import gannLogger
from gann.optimizers    import generatorOptimizer, discriminatorOptimizer
from gann.vis           import getImageWriter

from settings import settings

from util.util         import Timer
from util.checkFolders import checkFolders

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
      return {"discriminatorLoss": discLoss, "generatorLoss":genLoss}


def getCheckpoint():
  checkpointDir    = settings.train["output"]
  checkpointPrefix = os.path.join( checkpointDir, "ckpt")

  return tf.train.Checkpoint( generatorOptimizer      = generatorOptimizer,
                              discriminatorOptimizer  = discriminatorOptimizer,
                              generator               = generator,
                              discriminator           = discriminator),checkpointPrefix

def train(dataset, epochs = settings.train["epochs"]):
  logFolder = "".join([ settings.train["logDir"],
                        os.path.sep,
                        settings.train["networkName"]])
  
  checkFolders([  settings.train["output"]
                 ,logFolder  
               ])
  checkpoint, prefix = getCheckpoint()
  logger             = gannLogger( logFolder )
  imageLogger        = getImageWriter(logFolder )

  for epoch in range(epochs):
    dataset.shuffle(1000)
    with Timer('Time for epoch {epoch} is {t} sec',{"epoch":epoch}):
      start = time()
      for images, meanData, inputData in dataset:
        losses = trainStep(images, meanData, inputData)
        logger.updateLosses(losses)

    if epoch % settings.train["checkpointFrequency"] == 0:
      logger.dumpLosses( epoch )
      imageLogger(generator,epoch)
      checkpoint.save( file_prefix = prefix) 