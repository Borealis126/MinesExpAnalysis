import subprocess
import math
import numpy as np
from datetime import datetime


def initStudy(studyPath):
    subprocess.call("if not exist " + str(studyPath) + " mkdir " + str(studyPath), shell=True)

    dateStr = datetime.today().strftime('%Y-%m-%d')+'_0'
    resultsFolder = studyPath / 'results' / 'gate' / dateStr
    initialStatesResultsFolder = studyPath / 'results' / 'initialStates' / dateStr
    gatePulseFolder = studyPath / 'pulses' / 'gate'
    initialStatesPulseFolder = studyPath / 'pulses' / 'initialStates'

    for folder in [resultsFolder, initialStatesResultsFolder, gatePulseFolder, initialStatesPulseFolder]:
        subprocess.call("if not exist " + str(folder) + " mkdir " + str(folder), shell=True)


class Paths:
    def __init__(self, studyPath):
        self.studyPath = studyPath
        initStudy(studyPath)
        self.gateFolderPath = studyPath / 'pulses' / 'gate'
        self.gateFilePath = self.gateFolderPath / 'gates.json'
        self.resultsFolderPath = studyPath / 'results' / 'gate'

    def resultsPath(self, resultsName):
        return self.resultsFolderPath / resultsName


def expIndex(numInitialStates, numPostRotations, gateIndex, initialStateIndex, postRotationIndex):
    return gateIndex*(numInitialStates*numPostRotations)+initialStateIndex*numPostRotations+postRotationIndex

