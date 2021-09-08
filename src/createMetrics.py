from metrics import *

E100 = EvaluateMetrics(final_model='RMmodel100.pkl',
                       modelName='tierT12_10_100')
E100.confusionMatrixTrain()
E100.confusionMatrix()
E100.rocValidate()
E100.prValidate()

E100_all = combinedEvaluations(
    final_model_name='RMmodel100.pkl', modelName='tierT12_10_100')
E100_all.prCurvePlotAll()
E100_all.rocCurvePlotAll()

E50 = EvaluateMetrics(final_model='RMmodel50.pkl', modelName='tierT12_10_50')
E50.confusionMatrixTrain()
E50.confusionMatrix()
E50.rocValidate()
E50.rocCurvePlotAll()

E50_all = combinedEvaluations(
    final_model_name='RMmodel50.pkl', modelName='tierT12_10_50', control=False)
E50_all.prCurvePlotAll()
E50_all.rocCurvePlotAll()


Eannot3 = EvaluateAnnot3()
Eannot3.allMultilabelMetrics()

Eannot2 = EvaluateAnnot2()
Eannot2.allMultilabelMetrics()

Eannot1 = EvaluateAnnot1()
Eannot1.allMultilabelMetrics()

Econt = EvaluateControl()
Econt.allMultilabelMetrics()
