from keggAnalysis import ExtractKEGGDataControl, ExtractKEGGData


E = ExtractKEGGDataControl(final_model='LRCmodel.pkl')
E.multiConfusionMatrix(tagY='glycan', tagP='glycan',
                       target1='Label Name', target2='Pos')
