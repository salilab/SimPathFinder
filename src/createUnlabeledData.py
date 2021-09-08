from __init__ import ExtractUnlabeledData, SampleUnlabeledData

S=SampleUnlabeledData(data_dir='../../')
for ind in range(0,2000,500):
    S.saveSamples(sample_nos=ind)