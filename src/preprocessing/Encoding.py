import pandas as pd
import numpy as np
import re
import torch
from src.log import Reformat
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance


def encode_po(po):
    le = LabelEncoder()
    le.fit(po)
    return le.transform(po)


def generate_event_trace(path_to_data, path_to_label,
                         features=["Case_ID", "Timestamp", "Purchase_Order_ID", "Event_Name"],
                         case_column="Case_ID",
                         time_column="Timestamp"):
    data = pd.read_csv(path_to_data)
    data_select = data[features]
    data_select["Timestamp"] = pd.to_datetime(data_select["Timestamp"])
    le = LabelEncoder()
    le.fit(data_select["Event_Name"].values)
    data_select["Event_ID"] = le.transform(data_select["Event_Name"].values)
    data_select.sort_values(by='Timestamp', inplace = True)
    trace = Reformat.roll_sequence(data_select, case_column=case_column, time_column=time_column)
    trace["Purchase_Order_ID"] = trace["Purchase_Order_ID"].apply(encode_po)
    if path_to_label is not None:
        label = pd.read_pickle(path_to_label)
        label[case_column] = label[case_column].apply(int)
        label.set_index("Case_ID", inplace=True)
        trace = pd.merge(label[["trace_is_fit", "trace_fitness"]], trace, how="inner", on="Case_ID")
    return trace


def calculate_case_lapse(seq_time):
    time_diff = np.array(seq_time - seq_time[0])
    time_diff = time_diff / np.timedelta64(1, 's')
    return time_diff


def remove_abbreviation(event_list, abbreviation=["SRM:", '(E.Sys.)']):
    cleaned_event_list = []
    for event in event_list:
        pattern = r'(\s|^)(' + '|'.join(re.escape(word) for word in abbreviation) + r')(?=\s|$)'
        # pattern = '|'.join(re.escape(word) for word in abbreviation)
        cleaned_event = re.sub(pattern, '', event).lstrip().lower()
        cleaned_event_list.append(cleaned_event)
    return cleaned_event_list


def embedding_prototype_w2c(event_list, model):
    embedding_list = []
    for event in event_list:
        words = [word for word in event.split() if word in model.key_to_index]
        if len(words) == 0:
            return np.zeros(model.vector_size)
        sentence_vector = np.sum([model[word] for word in words], axis=0)
        embedding = sentence_vector / len(words)
        embedding_list.append(embedding)
    return np.stack(embedding_list)


def embedding_prototype_bert(event_list, tokenizer, bert_model):
    index_list = []
    for single_event in event_list:
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(single_event))
        index_list.append(input_ids)

    embedding_list = []
    for index in index_list:
        tensor_input_ids = torch.tensor([index])
        bert_model.eval()
        with torch.no_grad():
            outputs = bert_model(tensor_input_ids)
            last_hidden_states = outputs.last_hidden_state

        # The 'last_hidden_states' is a tensor with shape [1, seq_length, embedding_dim]
        # If you're interested in the embedding of the [CLS] token (for sentence classification tasks),
        # you can take `last_hidden_states[:, 0, :]` which is a tensor of shape [1, 1, embedding_dim].
        cls_embedding = last_hidden_states[:, 0, :].numpy()
        embedding_list.append(cls_embedding)

    embedding_list = np.stack(embedding_list).squeeze()
    return embedding_list

def embedding_prototype_bert_pca(event_list, tokenizer, bert_model, pca_model=None):
    index_list = []
    for single_event in event_list:
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(single_event))
        index_list.append(input_ids)

    embedding_list = []
    for index in index_list:
        tensor_input_ids = torch.tensor([index])
        bert_model.eval()
        with torch.no_grad():
            outputs = bert_model(tensor_input_ids)
            last_hidden_states = outputs.last_hidden_state

        # The 'last_hidden_states' is a tensor with shape [1, seq_length, embedding_dim]
        # If you're interested in the embedding of the [CLS] token (for sentence classification tasks),
        # you can take `last_hidden_states[:, 0, :]` which is a tensor of shape [1, 1, embedding_dim].
        cls_embedding = last_hidden_states[:, 0, :].numpy()
        embedding_list.append(cls_embedding)

    embedding_list = np.stack(embedding_list).squeeze()
    pca = pca_model
    if pca is None:
        pca = PCA()
        pca.fit(embedding_list)
    reduced_embeddings = pca.transform(embedding_list)
    return pca, reduced_embeddings


def embedding_prototype_gpt(event_list, tokenizer, model):
    index_list = []
    for single_event in event_list:
        input_ids = tokenizer.encode(single_event, return_tensors='pt')
        index_list.append(input_ids)

    embedding_list = []
    for index in index_list:
        outputs = model(index)
        last_hidden_states = outputs.last_hidden_state

        # The 'last_hidden_states' is a tensor with shape [1, seq_length, embedding_dim]
        # If you're interested in the embedding of the [CLS] token (for sentence classification tasks),
        # you can take `last_hidden_states[:, 0, :]` which is a tensor of shape [1, 1, embedding_dim].
        cls_embedding = last_hidden_states[:, -1, :].detach().numpy().squeeze()
        embedding_list.append(cls_embedding)

    embedding_list = np.stack(embedding_list).squeeze()
    return embedding_list


def embedding_prototype_st(event_list):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model.encode(event_list)


def calculate_embedding_distance(bpi_embedding, celonis_embedding, bpi_event, celonis_event):
    distance_list = []
    for i in range(celonis_embedding.shape[0]):
        distances = np.sqrt(np.sum((bpi_embedding - celonis_embedding[i, :]) ** 2, axis=1))
        distance_list.append(distances)
    distance_list = (np.stack(distance_list) / np.stack(distance_list).max()).tolist()
    distance_res_pd = pd.DataFrame(data=distance_list, columns=bpi_event)
    distance_res_pd.set_index(celonis_event)
    return distance_res_pd


def calculate_embedding_cosine_sim(bpi_embedding, celonis_embedding, bpi_event, celonis_event):
    distance_list = []
    vec_cos_sim = np.vectorize(distance.cosine, signature="(n),(n)->()")
    for i in range(celonis_embedding.shape[0]):
        distances = vec_cos_sim(bpi_embedding, celonis_embedding[i, :])
        distance_list.append(distances)
    distance_list = (np.stack(distance_list) / np.stack(distance_list).max()).tolist()
    distance_res_pd = pd.DataFrame(data=distance_list, columns=bpi_event)
    distance_res_pd.set_index(celonis_event)
    return distance_res_pd


def save_embedding(embedding_res, event_list, file_path):
    embedding_file = pd.DataFrame()
    embedding_file["Values"] = embedding_res.tolist()
    embedding_file.set_index(event_list,  inplace=True)
    embedding_file.to_pickle(file_path)
