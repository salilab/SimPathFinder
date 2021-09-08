from dnn import *

D = DNNModel()
D.cleanTextChar()
D.createEmbMatrix()
D.runParameterizeModel()
D.plotAccLoss()
