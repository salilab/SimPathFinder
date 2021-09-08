from models import ModelVectors, BuildClassicalModel, BuildControlModels
from metrics import Metrics, EvaluateMetrics, EvaluateControl, combinedEvaluations

E1 = EvaluateMetrics(data_dir='../labeldata/',
                     model_dir='../models/')
E1.confusionMatrixTrain()
E1.confusionMatrix()
E1.rocValidate()
E1.prValidate()

E = combinedEvaluations(data_dir='../labeldata/',
                        model_dir='../models/')
E.prCurvePlotAll()
E.rocCurvePlotAll()
