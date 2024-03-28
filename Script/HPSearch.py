import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd
from src.trainer import Trainer
from src.trainer import CaseDataSet
from src.model import DLModels
from src.utils import TorchUtils

import torch
from torch import nn

import optuna
import importlib.util
import joblib
import pickle


if __name__ == '__main__':
    args = sys.argv[1:]
    trainer_hyper_para = {"max_epoch": 100,
                          "max_ob_iter": 20,
                          "score_margin": 1,
                          "num_class": 1,
                          "num_layers": 1,
                          "learning_rate": 1e-3,
                          "training_loss_func": "CCM",
                          "eval_loss_func": "CCM"}

    torch_device, device_package = TorchUtils.get_torch_device()
    train_set = CaseDataSet.CaseDataset(split_pattern=args[0], input_data=args[1],
                                        data_version="_train", embedding_version="_" + args[2],
                                        earliness_requirement=True)

    val_set = CaseDataSet.CaseDataset(split_pattern=args[0], input_data=args[1],
                                      data_version="_val", embedding_version="_" + args[2],
                                      earliness_requirement=True)

    def objective(trial):
        # Hyperparameters
        input_size = 51  # The number of expected features in the input x
        hidden_size = trial.suggest_int("n_hidden", 4, 512)  # The number of features in the hidden state h
        num_layers = trial.suggest_int("n_layer", 1, 4)  # Number of recurrent layers
        batch_size = trial.suggest_int("batch_size", 10, 10000)
        num_classes = 1  # For binary classification
        learning_rate = 0.001

        model = DLModels.SimpleLSTM(input_size, hidden_size, num_layers, num_classes).to(torch_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model, train_loss_source, val_loss_srouce = Trainer.train_model(model, optimizer,
                                                                        None, None,
                                                                        train_set,
                                                                        val_set,
                                                                        batch_size,
                                                                        torch_device,
                                                                        device_package,
                                                                        Trainer.prefix_weighted_loss,
                                                                        trainer_hyper_para["max_epoch"],
                                                                        trainer_hyper_para["max_ob_iter"],
                                                                        trainer_hyper_para["score_margin"],
                                                                        print_iter=False)


        model.flatten()
        roc_auc, f1, f1_inverse, precision, precision_inverse, recall, recall_inverse = Trainer.eval_model(model, val_set,
                                                                                                           torch_device=torch_device,
                                                                                                           device_package=device_package)
        model_name = "LSTM_S_h" + str(hidden_size) + "_l" + str(num_layers) + "_" + args[2] + "_" + args[1]
        torch.save(model, "../../Model/Optuna/" + args[0] + "/LSTM/" + model_name + ".LSTM")

        training_stat = pd.DataFrame(columns=["TrainingLoss", "ValidationLoss"],
                                     data=np.hstack([train_loss_source.reshape((-1, 1)),
                                                     val_loss_srouce.reshape((-1, 1))]))
        training_stat.to_pickle("../../Model/Optuna/" + args[0] + "/LSTM/" + model_name + "_stat.pkl")

        del model
        device_package.empty_cache()
        return roc_auc


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    joblib.dump(study, "../../Model/Optuna/" + args[0] + "/LSTM/Study" + args[1] + ".pkl")