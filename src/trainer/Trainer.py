import numpy as np
import torch
import copy
from torch import nn

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def prefix_weighted_loss(loss_list, sample_num):
    return np.sum(loss_list * sample_num) / np.sum(sample_num)


def BCE_counter_imblance(y, torch_device):
    class_weight = 1
    if (1-y).sum() * y.sum() > 0:
        class_weight = ((1-y).sum()/y.sum())
    pos_weight = torch.ones(1).to(torch_device) * class_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return criterion


def forward_model_batch(model, x, y, optimizer, criterion, loss_prefix, torch_device, training=True):
    if criterion is None:
        criterion = BCE_counter_imblance(y, torch_device)
    outputs = model(x)
    # Backward and optimize
    if training:
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        loss = criterion(outputs, y)
    loss_prefix = loss_prefix + loss.item()*x.shape[0]
    return loss_prefix


def train_model_epoch(model, training_set, optimizer, criterion, torch_device, batch_size=50, training=True):
    training_data_set = training_set
    batch_size = batch_size
    loss_prefix_list = []
    sample_num_list = []
    for prefix_len in range(1, training_data_set.max_case_len):
        loss_prefix = 0
        training_data_set.set_prefix_length(prefix_len)
        training_data_set.shuffle_data()
        input_data = training_data_set[:]
        if input_data is None:
            # print("Max length reached, abort")
            break
        sample_num = input_data[0].shape[0]
        # print("Starting training at prefix length: ", prefix_len, " with sample num: ", sample_num)
        sample_num_list.append(sample_num)

        batch_num = int(sample_num / batch_size)
        for i in range(batch_num):
            x = input_data[0][int(batch_size * i) : int(batch_size * (i+1))].float().to(torch_device)
            y = input_data[1][int(batch_size * i) : int(batch_size * (i+1))].float().to(torch_device)
            loss_prefix = forward_model_batch(model, x, y, optimizer, criterion, loss_prefix,
                                              torch_device, training=training)

        if sample_num > batch_size * batch_num:
            x = input_data[0][batch_size * batch_num :].float().to(torch_device)
            y = input_data[1][batch_size * batch_num :].float().to(torch_device)
            loss_prefix = forward_model_batch(model, x, y, optimizer, criterion, loss_prefix,
                                              torch_device, training=training)

        loss_prefix_list.append(loss_prefix)
    return np.array(loss_prefix_list), np.array(sample_num_list)


def train_model(model, optimizer, criterion, criterion_eval, training_set, val_set,
                batch_size, torch_device, device_package, eval_func=prefix_weighted_loss,
                max_epoch=100, max_ob_iter=20, score_margin=1e-4, print_iter=False):
    train_loss_list = []
    val_loss_list = []
    score = 1e5
    best_iter = 0
    best_model = None
    for iter_epoch in range(max_epoch):
        device_package.empty_cache()
        loss_train, sample_num_train = train_model_epoch(model, training_set, batch_size=batch_size, optimizer=optimizer,
                                                         criterion=criterion, torch_device=torch_device)
        device_package.empty_cache()
        loss_val, sample_num_val = train_model_epoch(model, val_set, batch_size=batch_size, optimizer=optimizer,
                                                     criterion=criterion_eval, torch_device=torch_device, training=False)

        score_train = eval_func(loss_train, sample_num_train)
        score_val = eval_func(loss_val, sample_num_val)
        train_loss_list.append(score_train)
        val_loss_list.append(score_val)

        if score_val < (score - score_margin):
            score = score_val
            best_model = copy.deepcopy(model)
            best_iter = iter_epoch

        if iter_epoch > best_iter + max_ob_iter:
            break
        if print_iter:
            print("Finished training iteration: ", iter_epoch, " with val loss: ", score_val, " train loss: ", score_train)
    device_package.empty_cache()
    return best_model, np.array(train_loss_list), np.array(val_loss_list)


def evaluate_model(model, test_set, torch_device, device_package):
    res_list = []
    ref_list = []
    sample_num_list = []
    device_package.empty_cache()
    for prefix_len in range(1, test_set.max_case_len):
        test_set.set_prefix_length(prefix_len)
        input_data = test_set[:]
        if input_data is None:
            # print("Max length reached, abort")
            break
        sample_num = input_data[0].shape[0]
        sample_num_list.append(sample_num)
        x = input_data[0].float().to(torch_device)
        y = input_data[1].float().to(torch_device)
        ref_list.append(y.cpu())
        outputs = model(x)
        prob = torch.sigmoid(outputs).detach().cpu()
        res_list.append(prob)

    device_package.empty_cache()
    return res_list, ref_list, sample_num_list


def eval_model(model, test_set, torch_device, device_package, decision_boundary=0.5,
               weighted=True, print_res=False):
    model.flatten()
    res, ref, num = evaluate_model(model, test_set, torch_device, device_package)
    res_prob = np.squeeze(torch.concat(res).numpy())
    res_class = copy.copy(res_prob)
    res_class[res_class < decision_boundary] = 0
    res_class[res_class >= decision_boundary] = 1
    ref_class = np.squeeze(torch.concat(ref).numpy()).astype(int)
    roc_auc = roc_auc_score(ref_class, res_prob)
    f1 = f1_score(ref_class, res_class)
    f1_inverse = f1_score(1-ref_class, 1-res_class)
    precision = precision_score(ref_class, res_class)
    precision_inverse = precision_score(1-ref_class, 1-res_class)
    recall = recall_score(ref_class, res_class)
    recall_inverse = recall_score(1-ref_class, 1-res_class)

    if print_res:
        print("roc_auc: ", roc_auc)
        print("f1: ", f1)
        print("f1 inverse: ", f1_inverse)
        print("Precision: ", precision)
        print("Precision inverse: ", precision_inverse)
        print("Recall: ", recall)
        print("Recall inverse: ", recall_inverse)

    if weighted:
        total_class_num = ref_class.shape[0]
        pos_class_num = ref_class.sum()
        neg_class_num = total_class_num - ref_class.sum()
        weighted_f1 = (f1*pos_class_num + f1_inverse*neg_class_num) / total_class_num
        weighted_precision = (precision*pos_class_num + precision_inverse*neg_class_num) / total_class_num
        weighted_recall = (recall*pos_class_num + recall_inverse*neg_class_num) / total_class_num

        if print_res:
            print("roc_auc: ", roc_auc)
            print("weighted f1: ", weighted_f1)
            print("weighted_precision: ", weighted_precision)
            print("weighted_recall: ", weighted_recall)

        return [roc_auc, weighted_f1, weighted_precision, weighted_recall]
    else:
        return [roc_auc, f1, f1_inverse, precision, precision_inverse, recall, recall_inverse]