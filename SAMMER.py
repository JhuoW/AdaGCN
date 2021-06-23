import numpy as np
import torch
import torch.nn.functional as F


class SAMMER:
    def __init__(self, n_nodes, n_classes) -> None:
        self.n_nodes = n_nodes
        self.n_classes = n_classes


    def boost_real(self, model, X, y, idx,  sample_weight):

        model.eval()
        out = model(X)


        self.y_predict_proba =  F.softmax(out, -1)

        y_predict = torch.argmax(self.y_predict_proba, dim = -1)[idx]

        incorrect = y_predict != y
        self.classes = np.arange(self.n_classes)
        y_codes = np.array([-1. / (self.n_classes - 1), 1.])
        y = y.cpu().detach().numpy()
        y_coding = torch.from_numpy(y_codes.take(self.classes == y[:, np.newaxis])).cuda() 


        estimator_weight = (-1. * ((self.n_classes - 1.) / self.n_classes) * torch.xlogy(y_coding, self.y_predict_proba[idx]).sum(axis=1))



        # update sample weight
        sample_weight[idx] *= torch.exp(estimator_weight * ((sample_weight[idx] > 0) | (estimator_weight < 0)))

        return sample_weight


    def _samme_proba(self, n_classes):
        """
        Algo line 5
        """
        proba = self.y_predict_proba

        log_proba = torch.log(proba)

        return (n_classes - 1) * (log_proba - (1. / n_classes)
                                * log_proba.sum(axis=1)[:, None])