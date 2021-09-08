from __init__ import BalanceLabelData

B=BalanceLabelData()
B.loadAllData()
B.createAllPartialAnnot()
B.splitDataLSTM()
