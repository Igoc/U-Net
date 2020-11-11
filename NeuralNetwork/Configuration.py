import configparser

COLORS = {
    'RED': [255, 0, 0],
    'ORANGE': [255, 128, 0],
    'YELLOW': [255, 255, 0],
    'GREEN': [0, 255, 0],
    'BLUE': [0, 0, 255],
    'NAVY': [0, 0, 128],
    'VIOLET': [128, 0, 128],
    'WHITE': [255, 255, 255],
    'BLACK': [0, 0, 0]
}

patchSizes   = []
patchStrides = []
labelColors  = []
classColors  = []
classNumber  = -1

layerNumber  = -1
featuresRoot = -1

epoch             = -1
batchSize         = -1
learningRate      = -1
learningRateDecay = -1
classWeights      = []
weightRestoration = False

def LoadConfiguration(path):
    global patchSizes
    global patchStrides
    global labelColors
    global classColors
    global classNumber
    
    global layerNumber
    global featuresRoot
    
    global epoch
    global batchSize
    global learningRate
    global learningRateDecay
    global classWeights
    global weightRestoration
    
    parser = configparser.ConfigParser()
    parser.read(f'{path}/config.ini')
    
    patchSizes   = [int(patchSize.strip()) for patchSize in parser.get('Input', 'patch-sizes').split(',')]
    patchStrides = [int(patchStride.strip()) for patchStride in parser.get('Input', 'patch-strides').split(',')]
    labelColors  = [labelColor.strip() for labelColor in parser.get('Input', 'label-colors').split(',')]
    classColors  = [COLORS[labelColor.upper()] for labelColor in labelColors]
    classNumber  = len(labelColors)
    
    layerNumber  = parser.getint('Network', 'layer-number')
    featuresRoot = parser.getint('Network', 'features-root')
    
    epoch             = parser.getint('Training', 'epoch')
    batchSize         = parser.getint('Training', 'batch-size')
    learningRate      = parser.getfloat('Training', 'learning-rate')
    learningRateDecay = parser.getfloat('Training', 'learning-rate-decay')
    classWeights      = [float(classWeight.strip()) for classWeight in parser.get('Training', 'class-weights').split(',')]
    weightRestoration = parser.getboolean('Training', 'weight-restoration')