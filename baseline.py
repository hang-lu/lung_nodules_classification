from model import generate_model
import torch
from torch.utils.data import DataLoader,Dataset
import os
import json
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,roc_auc_score
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from tqdm import tqdm
import numpy as np
import time


class LungDataset(Dataset):
    def __init__(self, nps, labels):
        super(LungDataset, self).__init__()
        self.nps = nps
        self.labels = labels

    def __getitem__(self, item):
        np_path, label = self.nps[item], self.labels[item]
        data = torch.from_numpy(np.load(np_path)).unsqueeze(0).type(torch.FloatTensor)
        label = torch.tensor(1 if label == 1 else 0).type(torch.LongTensor)
        return data, label

    def __len__(self):
        return len(self.nps)
def save_checkpoint(states, is_best, output_dir, loss,filename='checkpoint.pth.tar'):
    torch.save(states,os.path.join(output_dir,filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],os.path.join(output_dir,'model_best_{}.pth.tar'.format(loss)))
if __name__=='__main__':
    anno_path = './data/annos.json'
    np_dir = './data'
    random_state = 0
    train_batch_size = 64
    val_batch_size = 64
    device = torch.device("cuda:1")
    model_path = ""
    learning_rate = 0.001
    max_epoch = 1000
    best_accuracy = 1e-9
    val_interval=1
    out_dir = 'result_out1'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    annos = json.load(open(anno_path, 'r'))
    nps_labels = np.array([[os.path.join(np_dir, '{}.npy'.format(fig_id)), annos[fig_id]] for fig_id in annos])

    sss= StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=random_state)
    train_index,val_index=next((sss.split(nps_labels,nps_labels[:,1])))
    train_nps,val_nps=nps_labels[train_index,0],nps_labels[val_index,0]
    train_labels,val_labels= nps_labels[train_index,1].astype(np.int),nps_labels[val_index,1].astype(np.int)

    train_dataset = LungDataset(train_nps, train_labels)
    val_dataset = LungDataset(val_nps, val_labels)
    train_dataloader = DataLoader(train_dataset, train_batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, val_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = generate_model(50, n_input_channels=1, n_classes=2)
    if len(model_path):
        encoder_dic = torch.load(model_path, map_location='cuda:0')
        encoder_dic = encoder_dic['state_dict'] if 'state_dict' in encoder_dic else encoder_dic
        model_dict = model.state_dict()
        encoder_dic = {k: v for k, v in encoder_dic.items() if
                       (k in model_dict) and encoder_dic[k].shape == model_dict[k].shape}
        model_dict.update(encoder_dic)
        model.load_state_dict(model_dict, strict=False)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=40, verbose=True,
                                                           min_lr=1e-6)
    for epoch in range(max_epoch):
        train_pred = []
        train_gt = []
        running_loss = 0.0
        start = time.time()
        model.train()
        for i,(input_data,labels) in enumerate(train_dataloader):
            input_data = input_data.to(device)
            labels = labels.to(device)
            out=model(input_data)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predict= torch.softmax(out,dim=-1)
            train_pred.append(predict[:,1].detach().to('cpu').numpy())
            # train_pred.append(out.argmax(dim=-1).detach().to('cpu').numpy())
            train_gt.append(labels.detach().to('cpu').numpy())

        train_pred = np.concatenate(train_pred, axis=0)
        train_gt = np.concatenate(train_gt, axis=0)

        auc= roc_auc_score(y_true=train_gt,y_score=train_pred)
        print('train: [%4d/ %5d] loss: %.6f, auc: %.6f,  time: %f s' %
              (epoch + 1, i, running_loss, auc, time.time() - start))
        # acc = accuracy_score(y_true=train_gt,y_pred=train_pred)

        if epoch and epoch % val_interval==0:
            model.eval()
            val_loss= 0.0
            pred = []
            gt = []
            for i, (input_data, labels) in enumerate(val_dataloader, 0):
                input_data = input_data.to(device)
                labels = labels.to(device)
                out = model(input_data)
                loss = criterion(out, labels)
                val_loss += loss.item()
                predict = torch.softmax(out, dim=-1)
                pred.append(predict[:,1].detach().to('cpu').numpy())
                gt.append(labels.detach().to('cpu').numpy())
            gt = np.concatenate(gt, axis=0)
            pred = np.concatenate(pred, axis=0)
            val_loss = val_loss / (i + 1)
            auc_val= roc_auc_score(y_true=gt,y_score=pred)
            # acc_val = accuracy_score(y_true=gt, y_pred=pred)  # sum(gt==pred)/gt.shape[0]
            print('test: [%4d/ %4d] loss: %.6f accuracy %.6f' %
                  (epoch + 1, max_epoch, val_loss, auc_val))
            scheduler.step(auc_val)
            if optimizer.state_dict()['param_groups'][0]['lr'] < 5 * 1e-6:
                break
            if best_accuracy < auc_val:
                best_accuracy = auc_val
                model_save_name = 'checkpoint_epoch{:03d}_accuracy{:0.4f}.pth.tar'.format(epoch, auc_val)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict() if isinstance(model,
                                                                          torch.nn.DataParallel) else model.state_dict(),
                    'perf': val_loss,
                    'optimizer': optimizer.state_dict(),
                }, True, out_dir, auc_val, model_save_name)




