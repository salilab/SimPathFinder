from embeddings import CreateEmbedding, ParameterizeEmbedding, ClusterEmbedding, ClusterPWYEmbedding
import pickle

EC_l = pickle.load(open('../ecdata/sampledECT12_1500.pkl', 'rb'))
C = CreateEmbedding(EC=EC_l)
model = C.FT_sg(size=50)
P = ParameterizeEmbedding()
P.parameterize_FT_complete()
count, mm = P.measure_results_all(model=model)
C1 = ClusterEmbedding()
C1.FT_tsne(model)
C = ClusterPWYEmbedding()
C.plot_umap()
