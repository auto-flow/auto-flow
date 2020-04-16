from scipy.stats import pearsonr

from autoflow.feature_engineer.compress.similarity_base import SimilarityBase

class Pearson(SimilarityBase):
    name = "pearson and f1_score"

    def core_func(self, s, e, L):
        # X_是全局变量
        to_del = []
        for i in range(s, e):
            for j in range(i + 1, L):
                r = pearsonr(self.X_[:, i], self.X_[:, j])[0]
                if r > self.threshold:
                    to_del.append([r, i])
                    break
        return to_del