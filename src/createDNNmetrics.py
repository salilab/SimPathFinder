from metrics import *

D = EvaluateMetricsDNN()
D.cleanTextChar()
D.confusionMatrix()
D.confusionMatrixTrain()


D = EvaluateMetricsAnnot3DNN()
D.cleanTextChar()
D.prCurvePlotAll()
D.rocValidate()
D.allMultilabelMetrics(annot=1)
