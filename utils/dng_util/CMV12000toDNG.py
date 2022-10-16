import struct, collections
import numpy as np
import sys
import shutil
from math import *
from DNGTags import *
from DNG import *

def addTag(dict, ifd, tagName):
    dict[DNG_TAGS_STR_ID[tagName]] = ifd.getTag(tagName)
def delTag(ifd, tagName):
    del ifd.tags[DNG_TAGS_STR_ID[tagName]]
    
organization = "apertus°"
cameraModel = "Axiom Alpha"
emptyDNG = 'empty.dng'
outputDNG = 'img.dng'
rawFile = sys.stdin

dng = DNG(emptyDNG)
dng.openDNG()
dng.readHeader()
dng.readIFDs()
dng.readSubIFD('SubIFDs')

makeTag = dng.ifd.getTag('Make')
makeTag.value = organization
makeTag.count = len(makeTag.value)
modelTag = dng.ifd.getTag('Model')
modelTag.value = organization + " " + cameraModel
modelTag.count = len(modelTag.value)
uniqueCameraModelTag = dng.ifd.getTag('UniqueCameraModel')
uniqueCameraModelTag.value = modelTag.value
uniqueCameraModelTag.count = modelTag.count
origRawFileNameTag = dng.ifd.getTag('OriginalRawFileName')
origRawFileNameTag.value = emptyDNG
origRawFileNameTag.count = len(origRawFileNameTag.value)

camCalTag = dng.ifd.getTag('CameraCalibrationSignature')
camCalTag.value = organization
camCalTag.count = len(camCalTag.value)
profCalTag = dng.ifd.getTag('ProfileCalibrationSignature')
profCalTag.value = organization
profCalTag.count = len(profCalTag.value)

# Camera calibration
matrix1Tag = dng.ifd.getTag('ColorMatrix1')
matrix1Tag.value[0].num = 19049 
matrix1Tag.value[1].num = -7877
matrix1Tag.value[2].num = -3582
matrix1Tag.value[3].num = -5724
matrix1Tag.value[4].num = 10121
matrix1Tag.value[5].num = 1917
matrix1Tag.value[6].num = -1267
matrix1Tag.value[7].num = 210
matrix1Tag.value[8].num = 5121

matrix1Tag.value[0].denom = 10000 
matrix1Tag.value[1].denom = 10000
matrix1Tag.value[2].denom = 10000 
matrix1Tag.value[3].denom = 10000 
matrix1Tag.value[4].denom = 10000 
matrix1Tag.value[5].denom = 10000 
matrix1Tag.value[6].denom = 10000 
matrix1Tag.value[7].denom = 10000 
matrix1Tag.value[8].denom = 10000 

delTag(dng.ifd,'ColorMatrix2')

calIllum1Tag = dng.ifd.getTag('CalibrationIlluminant1')
calIllum1Tag.value[0] = 3

delTag(dng.ifd,'CalibrationIlluminant2')

delTag(dng.ifd,'CameraCalibration1')
delTag(dng.ifd,'CameraCalibration2')

# Replace Image
width = 4096
height = 3072
numPixels = width*height
#rf = open(rawFile, mode='rb')
rf = rawFile
rawData = struct.unpack("H"*numPixels,rf.read(2*numPixels))
regData = struct.unpack("B"*256,rf.read(256))

rawFlatImage = np.zeros(numPixels,dtype=np.uint16)
rawFlatImage[:] = rawData[:] 
rawImage = np.reshape(rawFlatImage,(height,width))
rawImage = rawImage >> 4

rawIFD = dng.ifd.subIFDs[0]
rawIFD.image = rawImage
rawIFD.imageWidth = width
rawIFD.imageHeight = height
rawIFD.getTag('ImageWidth').value[0] = width
rawIFD.getTag('ImageLength').value[0] = height
rawIFD.getTag('StripByteCounts').value[0] = width*height*2
rawIFD.getTag('RowsPerStrip').value[0] = height
rawIFD.getTag('ActiveArea').value[0] = 0
rawIFD.getTag('ActiveArea').value[1] = 0
rawIFD.getTag('ActiveArea').value[2] = height
rawIFD.getTag('ActiveArea').value[3] = width
rawIFD.getTag('DefaultCropSize').value[0].num = width
rawIFD.getTag('DefaultCropSize').value[1].num = height
rawIFD.getTag('DefaultCropOrigin').value[0].num = 0
rawIFD.getTag('DefaultCropOrigin').value[1].num = 0

# Change Order to GBRG when y-flipped
rawIFD.getTag('CFAPattern').value[0] = 1
rawIFD.getTag('CFAPattern').value[1] = 2
rawIFD.getTag('CFAPattern').value[2] = 0
rawIFD.getTag('CFAPattern').value[3] = 1

# Add private data (registers)
privTag = TAG()
privTag.tag = DNG_TAGS_STR_ID['DNGPrivateData']
privTag.type = 1
privTag.count = 256
privTag.value = regData
dng.ifd.tags[privTag.tag] = privTag

dng.writeDNG(outputDNG)

