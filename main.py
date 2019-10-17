# smeenapadnekar
# 28/07/2019

from __future__ import division
import numpy as np
import os 
import argparse
import cv2

import editdistance
from imagePreprocessing import preprocess
from model import Model, DecoderType
from DataProcessing import DataLoader, Batch

# File path 
class FilePath:
    input = 'data/test.png'
    charList = 'model/charList.txt'
    accuracy = 'model/accuracy.txt'
    train = 'data/'
    corpus = 'data/corpus.txt'

def train(model, loader):
    epoc = 0
    bestCharErrorRate = float('inf')
    noImprovement = 0
    earlyStopping = 50

    while True:
        epoc += 1
        print('Epoc',epoc)
        loader.trainSet()
        print('Training Neural Network')
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            Batch = loader.getNext()
            loss = model.trainBatch(Batch)
            print('Batch : ',iterInfo[0],'/',iterInfo[1],' Loss =',loss)
        
        print('Validate')
        charErrorRate = validate(model,loader)

        if charErrorRate < bestCharErrorRate:
            print('Increase in accuracy. Saving Model')
            bestCharErrorRate = charErrorRate
            noImprovement = 0
            model.save()
            open(FilePath.accuracy,'w').write('Validation Character error rate of the saved model%f%%'%(bestCharErrorRate*100))
        else:
            print('No increase in Accuracy')
            noImprovement +=1
        
        # stopping if no improving in acc after 5 epoc
        if noImprovement>=earlyStopping:
            break




def validate(model, loader):
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0],'/', iterInfo[1])
        batch = loader.getNext()
        (recognized,_) = model.inferBatch(batch)

        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0 
            numWordTotal +=1
            dist = editdistance.eval(recognized[i],batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')

    # print validation result
    charErrorRate = numCharErr/numCharTotal if numCharTotal !=0 else 0
    wordAccuracy = numWordOK/numWordTotal if numWordTotal !=0 else 0
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
    return charErrorRate

def recognize(model,InImage):
    img = preprocess(cv2.imread(InImage,cv2.IMREAD_GRAYSCALE),Model.imageSize)
    batch = Batch(None,[img])
    (recognized,probability) = model.inferBatch(batch, True)
    print('Recognized:', '"' + recognized[0] + '"')
    print('Probability:', probability[0])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', help='train the Neural network', action='store_true')
    parser.add_argument('--validate', help='test the Neural network', action='store_true')
    parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
    parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding', action='store_true')
    parser.add_argument('--dump', help='store the NN weights', action='store_true')

    args = parser.parse_args()
    
    decoderType = DecoderType.BestPath
    if args.beamsearch:
	    decoderType = DecoderType.BeamSearch
    elif args.wordbeamsearch:
        decoderType = DecoderType.WordBeamSearch

    if args.train or args.validate :
        # load training data
        # execute training and validation
        loader = DataLoader(FilePath.train,Model.batchSize,Model.imageSize,Model.maxTextLen)
        open(FilePath.charList, 'w').write(str().join(loader.charList))
        open(FilePath.corpus,'w').write(str(' ').join(loader.trainWords + loader.validationWords))

        if args.train:
            # training
            model = Model(loader.charList,decoderType)
            train(model, loader)
        elif args.validate:
            # validate
            model = Model(loader,charList,decoderType,mustRestore=True)
            validate(model, loader)
    else:
        # print accuracy
        print(open(FilePath.accuracy).read())
        model = Model(open(FilePath.charList).read(), decoderType, mustRestore=True, dump=args.dump)
        recognize(model,FilePath.input)
            
if __name__ == '__main__':
    main()
