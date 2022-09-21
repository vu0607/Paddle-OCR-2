#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
from shapely.geometry import Polygon
"""
reference from :
https://github.com/MhLiao/DB/blob/3c32b808d4412680310d3d28eeb6a2d5bf1566c5/concern/icdar2015_eval/detection/iou.py#L8
"""


class DetectionIoUEvaluator(object):
    def __init__(self, iou_constraint=0.6, area_precision_constraint=0.5, iou_values=None):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint
        if iou_values is not None:
            self.iou_values = iou_values
            self.is_multi_iou = True
        else:
            self.iou_values = [iou_constraint]
            self.is_multi_iou = False

    def get_union(self, pD, pG):
        return Polygon(pD).union(Polygon(pG)).area

    def get_intersection_over_union(self, pD, pG):
        return self.get_intersection(pD, pG) / self.get_union(pD, pG)

    def get_intersection(self, pD, pG):
        return Polygon(pD).intersection(Polygon(pG)).area

    def compute_ap(self, confList, matchList, numGtCare):
        correct = 0
        AP = 0
        if len(confList) > 0:
            confList = np.array(confList)
            matchList = np.array(matchList)
            sorted_ind = np.argsort(-confList)
            confList = confList[sorted_ind]
            matchList = matchList[sorted_ind]
            for n in range(len(confList)):
                match = matchList[n]
                if match:
                    correct += 1
                    AP += float(correct) / (n + 1)

            if numGtCare > 0:
                AP /= numGtCare

        return AP

    def init_variable(self):
        self.perSampleMetrics = {}
        self.matchedSum = 0

        self.numGlobalCareGt = 0
        self.numGlobalCareDet = 0

        self.gtPols = []
        self.detPols = []

        self.gtPolPoints = []
        self.detPolPoints = []

        # Array of Ground Truth Polygons' keys marked as don't Care
        self.gtDontCarePolsNum = []
        # Array of Detected Polygons' matched with a don't Care GT
        self.detDontCarePolsNum = []

        self.evaluationLog = ""

    def valid_gt(self, gt):
        for n in range(len(gt)):
            points = gt[n]['points']
            dontCare = gt[n]['ignore']

            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            gtPol = points
            self.gtPols.append(gtPol)
            self.gtPolPoints.append(points)
            if dontCare:
                self.gtDontCarePolsNum.append(len(self.gtPols) - 1)

        self.evaluationLog += "GT polygons: " + str(len(self.gtPols)) + (
            " (" + str(len(self.gtDontCarePolsNum)) + " don't care)\n"
            if len(self.gtDontCarePolsNum) > 0 else "\n")

    def valid_pred(self, pred):
        for n in range(len(pred)):
            points = pred[n]['points']

            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            detPol = points
            self.detPols.append(detPol)
            self.detPolPoints.append(points)
            if len(self.gtDontCarePolsNum) > 0:
                for dontCarePol in self.gtDontCarePolsNum:
                    dontCarePol = self.gtPols[dontCarePol]
                    intersected_area = self.get_intersection(dontCarePol, detPol)
                    pdDimensions = Polygon(detPol).area
                    precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                    if (precision > self.area_precision_constraint):
                        self.detDontCarePolsNum.append(len(self.detPols) - 1)
                        break

        self.evaluationLog += "DET polygons: " + str(len(self.detPols)) + (
            " (" + str(len(self.detDontCarePolsNum)) + " don't care)\n"
            if len(self.detDontCarePolsNum) > 0 else "\n")

    def reset_variable(self):
        self.gtRectMat = np.zeros(len(self.gtPols), np.int8)
        self.detRectMat = np.zeros(len(self.detPols), np.int8)
        self.detMatched = 0
        self.pairs = []
        self.detMatchedNums = []

    def evaluate_image(self, gt, pred):
        self.init_variable()
        self.valid_gt(gt)
        self.valid_pred(pred)

        numGtCare = (len(self.gtPols) - len(self.gtDontCarePolsNum))
        numDetCare = (len(self.detPols) - len(self.detDontCarePolsNum))

        # Calculate IoU and precision matrixs
        outputShape = [len(self.gtPols), len(self.detPols)]
        iouMat = np.empty(outputShape)
        for gtNum in range(len(self.gtPols)):
            for detNum in range(len(self.detPols)):
                pG = self.gtPols[gtNum]
                pD = self.detPols[detNum]
                iouMat[gtNum, detNum] = self.get_intersection_over_union(pD, pG)

        for iou_value in self.iou_values:
            self.reset_variable()
            for gtNum in range(len(self.gtPols)):
                for detNum in range(len(self.detPols)):
                    if self.gtRectMat[gtNum] == 0 and self.detRectMat[
                            detNum] == 0 and gtNum not in self.gtDontCarePolsNum and detNum not in self.detDontCarePolsNum:
                        if iouMat[gtNum, detNum] > iou_value:
                            self.gtRectMat[gtNum] = 1
                            self.detRectMat[detNum] = 1
                            self.detMatched += 1
                            self.pairs.append({'gt': gtNum, 'det': detNum})
                            self.detMatchedNums.append(detNum)
                            self.evaluationLog += "Match GT #" + \
                                             str(gtNum) + " with Det #" + str(detNum) + "\n"

            if numGtCare == 0:
                recall = float(1)
                precision = float(0) if numDetCare > 0 else float(1)
            else:
                recall = float(self.detMatched) / numGtCare
                precision = 0 if numDetCare == 0 else float(self.detMatched) / numDetCare

            hmean = 0 if (precision + recall) == 0 else 2.0 * \
                                                        precision * recall / (precision + recall)

            self.matchedSum += self.detMatched
            self.numGlobalCareGt += numGtCare
            self.numGlobalCareDet += numDetCare

            self.perSampleMetrics[str(iou_value)] = {
                'precision': precision,
                'recall': precision,
                'hmean': hmean,
                'pairs': self.pairs,
                'iouMat': [] if len(self.detPols) > 100 else iouMat.tolist(),
                'gtPolPoints': self.gtPolPoints,
                'detPolPoints': self.detPolPoints,
                'gtCare': numGtCare,
                'detCare': numDetCare,
                'gtDontCare': self.gtDontCarePolsNum,
                'detDontCare': self.detDontCarePolsNum,
                'detMatched': self.detMatched,
                'evaluationLog': self.evaluationLog
            }

        if not self.is_multi_iou:
            return self.perSampleMetrics[str(self.iou_constraint)]
        else:
            return self.perSampleMetrics

    def combine_results_multi_iou(self, results, iou_values):
        methodMetrics = {}

        for iou_value in iou_values:
            iou_str = str(iou_value)
            numGlobalCareGt = 0
            numGlobalCareDet = 0
            matchedSum = 0
            for result in results:
                numGlobalCareGt += result[iou_str]['gtCare']
                numGlobalCareDet += result[iou_str]['detCare']
                matchedSum += result[iou_str]['detMatched']

            methodRecall = 0 if numGlobalCareGt == 0 else float(
                matchedSum) / numGlobalCareGt
            methodPrecision = 0 if numGlobalCareDet == 0 else float(
                matchedSum) / numGlobalCareDet
            methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * \
                                                                    methodRecall * methodPrecision / (
                                                                            methodRecall + methodPrecision)
            # print(methodRecall, methodPrecision, methodHmean)
            # sys.exit(-1)
            methodMetrics.update({
                f'precision {iou_value}': methodPrecision,
                f'recall {iou_value}': methodRecall,
                f'hmean {iou_value}': methodHmean
            })

        return methodMetrics

    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        for result in results:
            numGlobalCareGt += result['gtCare']
            numGlobalCareDet += result['detCare']
            matchedSum += result['detMatched']

        methodRecall = 0 if numGlobalCareGt == 0 else float(
            matchedSum) / numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(
            matchedSum) / numGlobalCareDet
        methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * \
                                                                    methodRecall * methodPrecision / (
                                                                            methodRecall + methodPrecision)
        # print(methodRecall, methodPrecision, methodHmean)
        # sys.exit(-1)
        methodMetrics = {
            'precision': methodPrecision,
            'recall': methodRecall,
            'hmean': methodHmean
        }

        return methodMetrics


if __name__ == '__main__':
    evaluator = DetectionIoUEvaluator()
    gts = [[{
        'points': [(0, 0), (1, 0), (1, 1), (0, 1)],
        'text': 1234,
        'ignore': False,
    }, {
        'points': [(2, 2), (3, 2), (3, 3), (2, 3)],
        'text': 5678,
        'ignore': False,
    }]]
    preds = [[{
        'points': [(0.1, 0.1), (1, 0), (1, 1), (0, 1)],
        'text': 123,
        'ignore': False,
    }]]
    results = []
    for gt, pred in zip(gts, preds):
        results.append(evaluator.evaluate_image(gt, pred))
    metrics = evaluator.combine_results(results)
    print(metrics)
