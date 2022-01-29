import os
from   time import time

import tensorflow as tf

from image.augmentation import augmentRandomly

from gann.discriminator import makeDiscriminatorModel, discriminatorLoss
from gann.generator     import makeGeneratorModel, generatorLoss
from gann.logging       import gannLogger
from gann.optimizers    import generatorOptimizer, discriminatorOptimizer
from gann.vis           import getImageWriter

from settings import settings

from util.util         import Timer,addFolderToPath,checkFolders

generator     = makeGeneratorModel()
discriminator = makeDiscriminatorModel()

def trainStep(images, meanData, inputData):
    with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
      generatedImage = generator( inputData, training = True )

      discReal = discriminator( augmentRandomly(images), training = True)
      discFake = discriminator( generatedImage,          training = True)

      discLoss = discriminatorLoss( discReal, discFake )
      genLoss  = generatorLoss( discFake, generatedImage, meanData )

      gradientsGenerator     = genTape.gradient(  genLoss,  generator.trainable_variables )
      gradientsDiscriminator = discTape.gradient( discLoss, discriminator.trainable_variables )

      generatorOptimizer.apply_gradients(     zip( gradientsGenerator,     generator.trainable_variables))
      discriminatorOptimizer.apply_gradients( zip( gradientsDiscriminator, discriminator.trainable_variables))
      return {"discriminatorLoss": discLoss, "generatorLoss":genLoss}


def getCheckpoint(checkpointDir):
  checkpointPrefix = os.path.join( checkpointDir, "ckpt")

  return tf.train.Checkpoint( generatorOptimizer      = generatorOptimizer,
                              discriminatorOptimizer  = discriminatorOptimizer,
                              generator               = generator,
                              discriminator           = discriminator,
                              step                    = tf.Variable(1)),checkpointPrefix

def train(dataset, epochs = settings.train["epochs"], continueTraining = True ):
  logFolder,checkpointFolder  = [addFolderToPath(p,settings.train["networkName"]) for p in [ settings.train["output"],
                                                                                             settings.train["logDir"]]]   
  checkFolders( [logFolder, checkpointFolder] )
  checkpoint, prefix = getCheckpoint(checkpointFolder)
  logger             = gannLogger( logFolder )
  imageLogger        = getImageWriter(logFolder )

  if continueTraining:
       checkpoint.restore(tf.train.latest_checkpoint(checkpointFolder))
 
  for _ in range(epochs):
    epoch = int(checkpoint.step)

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

    checkpoint.step.assign_add(1)