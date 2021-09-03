import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# def save_checkpoint(save_path, model):

#     if save_path == None:
#         return

#     model.save_pretrained(save_path)

#     print(f'Model saved to ==> {save_path}')


# def load_checkpoint(load_path, model):

#     if load_path==None:
#         return

#     state_dict = torch.load(load_path, map_location=device)
#     print(f'Model loaded from <== {load_path}')

#     model.load_state_dict(state_dict['model_state_dict'])
#     return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    # print(f'Metrics saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return

    state_dict = torch.load(load_path, map_location=device)
    # print(f'Metrics loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def evaluate(model, test_loader):
    y_pred = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, leave=False):
            masks = batch[1].type(torch.LongTensor)
            masks = masks.to(device)
            comments = batch[0].type(torch.LongTensor)
            comments = comments.to(device)
            outputs = model(input_ids=comments, attention_mask=masks)

            logits = outputs[0]
            logits = torch.clamp(logits, min=-1.0, max=1.0)
            y_pred.extend(logits.tolist())
    
    y_pred = [p[0] for p in y_pred]
    
    return y_pred


def evaluate_metrics(model, test_loader, show=True):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            comments = batch[0].type(torch.LongTensor) 
            comments = comments.to(device) 
            masks = batch[1].type(torch.LongTensor) 
            masks = masks.to(device) 
            labels = batch[2].type(torch.FloatTensor)
            labels = labels.to(device)
            
            outputs = model(input_ids=comments, attention_mask=masks, labels=labels)
            loss, logits = outputs[:2]
            logits = torch.clamp(logits, min=-1.0, max=1.0)
            
            y_pred.extend(logits.tolist())
            y_true.extend(labels.tolist())
            
    y_pred = [p[0] for p in y_pred]
    
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    if show: print('RMSE:', rmse)
    
    inrange = [(abs(t-p) <= 1./12) if (t > 0) else (p < 0) for t, p in zip(y_true, y_pred)]
    acc1 = 100 * inrange.count(True) / len(inrange)
    if show: print ('[-0.5, 0.5] range accuracy:', acc1)
    
    inrange = [(abs(t-p) <= 1./6) if (t > 0) else (p < 0) for t, p in zip(y_true, y_pred)]
    acc2 = 100 * inrange.count(True) / len(inrange)
    if show: print ('[-1.0, 1.0] range accuracy:', acc2)
    
    return y_pred, rmse, acc1, acc2


def evaluate_test_metrics(test_loader, y_pred):
    y_true = []
    
    for batch in test_loader:
        comments = batch[0].type(torch.LongTensor) 
        comments = comments.to(device) 
        masks = batch[1].type(torch.LongTensor) 
        masks = masks.to(device) 
        labels = batch[2].type(torch.FloatTensor)
        labels = labels.to(device)

        y_true.extend(labels.tolist())
            
    print('RMSE:', mean_squared_error(y_true, y_pred, squared=False))
    
    inrange = [(abs(t-p) <= 1./12) if (t > 0) else (p == 0) for t, p in zip(y_true, y_pred)]
    acc = 100 * inrange.count(True) / len(inrange)
    print ('[-0.5, 0.5] range accuracy:', acc)
    
    inrange = [(abs(t-p) <= 1./6) if (t > 0) else (p == 0) for t, p in zip(y_true, y_pred)]
    acc = 100 * inrange.count(True) / len(inrange)
    print ('[-1.0, 1.0] range accuracy:', acc)
    
    return y_pred
    
    
def predict_ensemble(outputs):
    outputs = np.array(outputs)
    outputs_mean = np.mean(outputs, axis=0)
    outputs_var = np.std(outputs, axis=0)
    outputs_mean[outputs_mean < 0] = 0
    return list(outputs_mean), list(outputs_var)