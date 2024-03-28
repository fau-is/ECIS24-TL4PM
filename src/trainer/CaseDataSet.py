import pandas as pd
import numpy as np
import os

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


class CaseDataset(Dataset):
    def __init__(self, project_data_path="../../Data/Training", split_pattern="811split", input_data="source",
                 data_version="_train", embedding_version="", time_feature=True, earliness_requirement=True):

        file_name = input_data + data_version + ".pkl"
        self.data_path = os.path.join(project_data_path, split_pattern, file_name)
        self.intime_requirement = earliness_requirement
        self.time_feature = time_feature

        self.data_all = pd.read_pickle(self.data_path)
        self.data_all = self.data_all.reset_index()

        self.data_all["CaseLength"] = self.data_all["Event_Name"].apply(len)
        self.max_case_len = self.data_all["CaseLength"].max()

        self.prefix_length = 1
        self.data_pool = self.data_all[["Case_ID", "Event_Name", "CaseLapse", "CaseLength", 'Label', "OverallCriteria"]].copy()

        self.act_encoding = pd.read_pickle("../../Embedding/embedding_" + input_data + embedding_version + ".pkl")
        self.data_pool["Feature"] = self.data_pool.apply(self.encode_feature, axis=1)

    def encode_feature(self, sample):
        activity_encoded = self.encode_act(sample["Event_Name"])
        feature = torch.from_numpy(activity_encoded)
        # print(feature.shape)
        lapse_time = torch.from_numpy(sample["CaseLapse"])
        if self.time_feature:
            feature_list = [feature, lapse_time.reshape((-1, 1))]
        else:
            feature_list = [feature]
        feature = torch.cat(feature_list, dim=1)
        return feature.numpy()

    def encode_act(self, feature):
        return np.array(self.act_encoding.loc[feature]["Values"].values.tolist())

    def shuffle_data(self):
        self.data_pool = self.data_pool.sample(frac=1)

    def one_hot_po(self, feature):
        return F.one_hot(torch.from_numpy(feature).long(), num_classes=self.sub_seq_size)

    def set_prefix_length(self, length):
        self.prefix_length = length

    def update_data_pool(self):
        data_temp = self.data_pool[self.data_pool["CaseLength"] > self.prefix_length]
        if self.intime_requirement:
            data_temp = data_temp[data_temp["CaseLapse"].apply(lambda x: x[self.prefix_length]) < data_temp["OverallCriteria"]]
        return data_temp

    def convert_feature_vec_np(self, data):
        data_tmp = np.stack(data[:self.prefix_length])
        data_output = torch.from_numpy(data_tmp)
        return data_output

    def __len__(self):
        data_temp = self.update_data_pool()
        return data_temp.shape[0]

    def __getitem__(self, idx):
        data_temp = self.update_data_pool()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = data_temp["Feature"].apply(self.convert_feature_vec_np).values[idx].tolist()
        y = data_temp["Label"].values[idx].tolist()

        if len(x) == 0 :
            return None
        if len(x) == 1:
            return torch.unsqueeze(x[0], 0), torch.tensor(y).reshape((-1,1))
        return torch.stack(x) , torch.tensor(y).reshape((-1,1))