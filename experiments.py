#!/usr/bin/env python

from ece import *
from ksskml import *
import itertools
import copy


def binarizeDataset(dataset, falseLabel):
    original_source_samples = copy.deepcopy(dataset.source_samples)
    for index, sample in enumerate(dataset.source_samples):
        if sample.label == falseLabel:
            dataset.source_samples[index].label = 0
        else:
            dataset.source_samples[index].label = 1
    dataset.classes = {'0': 0, '1': 1}
    return original_source_samples

def run(dataset, configuration, selection):
    acc = 0
    bac = 0
    for fold in xrange(0, 5):
        dataset.setCV(fold)

        ensemble = ECE(dataset, configuration, selection)
        ensemble.learn()
        dataset.clearSupports()

        ensemble.predict()
        score = dataset.score()

        acc += score['accuracy']
        bac += score['bac']
        print '[%i] ACC: %.3f\t BAC: %.3f' % (
            fold, score['accuracy'], score['bac'])

    print '[-] ---: %.3f\t ---: %.3f' % (acc / 5, bac / 5)

def bgMask(dataset, predictionToMask):
    bg_mask = []
    for index, sample in enumerate(dataset_2.source_samples):
        if sample.prediction == predictionToMask:
            bg_mask += [index]
    return bg_mask

# Static configuration
configuration = {
    'radius': .5,
    'grain': 40,
    'dimensions': [2],
    'eceApproach': ECEApproach.brutal,
    'exposerVotingMethod': ExposerVotingMethod.lone
}

selection = [1, 2, 6, 8, 15, 17]

# Experiment 1
print "\n# Experiment 1\nECE for multilabel problem"
dataset_1 = Dataset("hyper.csv")

run(dataset_1,configuration,selection)

# Experiment 2
print "\n# Experiment 2\nECE for binarized problem"
dataset_2 = Dataset("hyper.csv")
binarizeDataset(dataset_2,2)
run(dataset_2,configuration,selection)

# Experiment 3
print "\n# Experiment 3\nECE for two-staged problem, only sickness"

dataset_3 = Dataset("hyper.csv")
bg_mask = bgMask(dataset_2, 0)
bg_samples = [dataset_3.source_samples[i] for i in bg_mask]
dataset_3.source_samples = [i for j, i in enumerate(dataset_3.source_samples) if j not in bg_mask]

dataset_3.prepareCV()
run(dataset_3,configuration,selection)

# Experiment 4
print "\n# Experiment 4\nECE for two-staged problem, connected"
i = 0
for index, sample in enumerate(bg_samples):
    sample.prediction = 2
    dataset_3.source_samples.insert(bg_mask[index], sample)

dataset_3.prepareCV()
run(dataset_3,configuration,selection)
