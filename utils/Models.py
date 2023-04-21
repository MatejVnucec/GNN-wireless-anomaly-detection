import importlib
import utils
import os

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import sklearn
import pytorch_lightning as pl

from torch.nn import Linear, CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, ChebConv, global_sort_pool
from torch.nn import Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv, GINEConv, GATv2Conv, GATConv

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ProgressBarBase
from pytorch_lightning.loggers import TensorBoardLogger
#from pytorch_lightning.loggers import CSVLogger
from dvclive.lightning import DVCLiveLogger


class GINE(pl.LightningModule):
    def __init__(self):
        super(GINE, self).__init__()
        
        if Config["graph"]["type"] in ("MTF_on_VG", "VG_on_MTF", "double_VG", "dual_VG"):
            edge_dim = 2
        else:
            edge_dim = 1
            
        dim_h = 32
    
        self.conv1 = GINEConv(
            Sequential(Linear(dim_h, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()), edge_dim=edge_dim)
        
        self.conv2 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()), edge_dim=edge_dim)
        
        self.conv3 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()), edge_dim=edge_dim)
        
        self.conv4 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()), edge_dim=edge_dim)
        
        self.conv5 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()), edge_dim=edge_dim)
        
        
        self.lin1 = Linear(dim_h*5, dim_h*5)
        self.lin2 = Linear(dim_h*5, len(class_weights))
    
    
    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Node embeddings 
        h1 = self.conv1(x, edge_index, edge_attr=edge_weight)
        h2 = self.conv2(h1, edge_index, edge_attr=edge_weight)
        h3 = self.conv3(h2, edge_index, edge_attr=edge_weight)
        h4 = self.conv4(h3, edge_index, edge_attr=edge_weight)
        h5 = self.conv5(h4, edge_index, edge_attr=edge_weight)
        
        # Graph-level readout
        
        h1 = global_max_pool(h1, batch)
        h2 = global_max_pool(h2, batch)
        h3 = global_max_pool(h3, batch)
        h4 = global_max_pool(h4, batch)
        h5 = global_max_pool(h5, batch)
        
        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3, h4, h5), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return h
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), lr=MConfig["learning_rate"], weight_decay=5e-4)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
                     
        out = model(train_batch)
        loss_function = CrossEntropyLoss(weight=class_weights).to(device)
        train_loss = loss_function(out, train_batch.y)
        
        correct=out.argmax(dim=1).eq(train_batch.y).sum().item()
        logs={"train_loss": train_loss}
        total=len(train_batch.y)
        
        batch_dictionary={"loss": train_loss, "log": logs, "correct": correct, "total": total}
        
        return train_loss
    
    
    def validation_step(self, val_batch, batch_idx):
      
        out = model(val_batch)
        loss_function = CrossEntropyLoss(weight=class_weights).to(device)
        val_loss = loss_function(out, val_batch.y)
        
        pred = out.argmax(-1)
        correct=out.argmax(dim=1).eq(val_batch.y).sum().item()
        total=len(val_batch.y)
        val_label = val_batch.y
        accuracy = (pred == val_label).sum() / pred.shape[0]
        
        logs={"train_loss": val_loss}
        batch_dictionary={"loss": val_loss, "log": logs, "correct": correct, "total": total}
        
        self.log("val_loss", val_loss)
        self.log("val_acc", accuracy)
        
    
    def test_step(self, test_batch, batch_idx):
        out = model(test_batch)
        loss_function = CrossEntropyLoss(weight=class_weights).to(device)
        test_loss = loss_function(out, test_batch.y)
        
        pred = out.argmax(-1)
        test_label = test_batch.y
        accuracy = (pred == test_label).sum() / pred.shape[0]
        self.log("test_true", test_label)
        self.log("test_pred", pred)
        self.log("test_acc", accuracy)
        return pred, test_label
        
    def test_epoch_end(self, outputs):
        #this function gives us in the outputs all acumulated pred and test_labels we returned in test_step
        #we transform the pred and test_label into a shape that the classification report can read
        true_array=[]
        pred_array = []
        for i in range(len(outputs)):
            true_array = np.append(true_array,outputs[i][1].cpu().numpy())
            pred_array = np.append(pred_array,outputs[i][0].cpu().numpy())            
        print(confusion_matrix(true_array, pred_array))
        print(classification_report(true_array, pred_array))
        return pred_array, true_array
    
class NetBCE(pl.LightningModule):
    def __init__(self):
        super(NetBCE, self).__init__()
        self.conv = nn.ModuleList([
            GATConv(1, 32, heads=4),
            GATConv(4 * 32, 32, heads=4),
            GATConv(4 * 32, len(class_weights), heads=6,concat=False)
        ])
        self.lin = nn.Sequential(
            torch.nn.Linear(1, 4 * 32),
            torch.nn.Linear(4 * 32, 4 * 32),
            torch.nn.Linear(4 * 32, len(class_weights))
        )
    
    def forward(self, data):
        x, edge_index, edge_weight = data.x.cuda(), data.edge_index.cuda(), data.edge_attr.cuda()
        x = F.elu(self.conv[0](x, edge_index, edge_weight) + self.lin[0](x))
        x = F.elu(self.conv[1](x, edge_index, edge_weight) + self.lin[1](x))
        x = self.conv[2](x, edge_index, edge_weight) + self.lin[2](x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), lr=MConfig["learning_rate"], weight_decay=5e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):        
        out = model(train_batch)
        loss_function = BCEWithLogitsLoss(weight=class_weights).to(device)
        
        train_loss = loss_function(out, train_batch.y)
        correct=out.argmax(dim=1).eq(train_batch.y).sum().item()
        logs={"train_loss": train_loss}
        total=len(train_batch.y)
        
        batch_dictionary={"loss": train_loss, "log": logs, "correct": correct, "total": total}
        
        return train_loss
    
    def validation_step(self, val_batch, batch_idx):
      
        out = model(val_batch)
        loss_function = BCEWithLogitsLoss(weight=class_weights).to(device)
        val_loss = loss_function(out, val_batch.y)
        
        ys, preds = [], []
        val_label = val_batch.y.cpu()
        ys.append(val_batch.y)
        preds.append((out > 0).float().cpu())     
        y, pred = torch.cat(ys, dim=0), torch.cat(preds, dim=0)
        accuracy = (pred == val_label).sum() / pred.shape[0]
    
        self.log("val_loss", val_loss)
        self.log("val_acc", accuracy)
    
    def test_step(self, test_batch, batch_idx):
        # this is the test loop
        out = model(test_batch)
        loss_function = BCEWithLogitsLoss(weight=class_weights).to(device)
        test_loss = loss_function(out, test_batch.y)
        
        ys, preds = [], []
        test_label = test_batch.y.cpu()
        ys.append(test_batch.y)
        preds.append((out > 0).float().cpu())
        
        y, pred = torch.cat(ys, dim=0), torch.cat(preds, dim=0)
        accuracy = (pred == test_label).sum() / pred.shape[0]
        
        self.log("test_acc", accuracy)
        return pred, y
        
    def test_epoch_end(self, outputs):
        #this function gives us in the outputs all acumulated pred and test_labels we returned in test_step
        #we transform the pred and test_label into a shape that the classification report can read
        global true_array, pred_array
        true_array=[outputs[i][1].cpu().numpy() for i in range(len(outputs))]
        pred_array = [outputs[i][0].cpu().numpy() for i in range(len(outputs))]
        pred_array = np.array(pred_array).reshape(-1, 1)
        true_array = np.array(true_array).reshape(-1, 1)
        print(confusion_matrix(true_array, pred_array))
        print(classification_report(true_array, pred_array))
        print("pred_array ",pred_array)

class NetCE(pl.LightningModule):
    def __init__(self):
        super(NetCE, self).__init__()
        self.conv = nn.ModuleList([
            GATConv(1, 32, heads=4),
            GATConv(4 * 32, 32, heads=4),
            GATConv(4 * 32, 32, heads=8),
            GATConv(8 * 32, len(class_weights), heads=6,concat=False)
            # GATConv(16 * 32, 32, heads=32)        
            # GATConv(32 * 32, 32, heads=64)        
            # GATConv(64 * 32, 32, heads=16)
            # GATConv(16 * 32, len(class_weights), heads=6,concat=False)
        ])
        self.lin = nn.Sequential(
            torch.nn.Linear(1, 4 * 32),
            torch.nn.Linear(4 * 32, 4 * 32),
            torch.nn.Linear(4 * 32, 8 * 32),
            torch.nn.Linear(8 * 32, len(class_weights))
            # torch.nn.Linear(16 * 32, 32 * 32)
            # torch.nn.Linear(32 * 32, 64 * 32)
            # torch.nn.Linear(64 * 32, 16 * 32)
            # torch.nn.Linear(16 * 32, len(class_weights))
        )
    
    def forward(self, data):
        x, edge_index, edge_weight = data.x.cuda(), data.edge_index.cuda(), data.edge_attr.cuda()
        x = F.elu(self.conv[0](x, edge_index, edge_weight) + self.lin[0](x))
        x = F.elu(self.conv[1](x, edge_index, edge_weight) + self.lin[1](x))
        x = F.elu(self.conv[2](x, edge_index, edge_weight) + self.lin[2](x))
        x = self.conv[3](x, edge_index, edge_weight) + self.lin[3](x)
        # x = F.elu(self.conv[4](x, edge_index, edge_weight) + self.lin[4](x))
        # x = F.elu(self.conv[5](x, edge_index, edge_weight) + self.lin[5](x))
        # x = F.elu(self.conv[6](x, edge_index, edge_weight) + self.lin[6](x))
        # x = self.conv[7](x, edge_index, edge_weight) + self.lin[7](x)
        return x
    
    def inference(self, data):
        with torch.no_grad():
            return self.forward(data)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), lr=MConfig["learning_rate"], weight_decay=5e-4)
        return optimizer
    
    def training_step(self, data, batch_idx):
        out = model(data)
        loss_function = CrossEntropyLoss(weight=class_weights).to(device)

        if hasattr(data, 'train_mask'):
            # If 'train_mask' attribute is present in data, use it for masking
            y_data = data.y[data.train_mask]
            out_data = out[data.train_mask]
        else:
            y_data = data.y
            out_data = out
            
        train_loss = loss_function(out_data, y_data.squeeze().to(torch.int64))        
        return train_loss

    def validation_step(self, data, batch_idx):
        out = model(data)
        loss_function = CrossEntropyLoss(weight=class_weights).to(device) #weight=class_weight
        
        if hasattr(data, 'val_mask'):
            # If 'val_mask' attribute is present in data, use it for masking
            y_data = data.y[data.val_mask]
            out_data = out[data.val_mask]
        else:
            y_data = data.y
            out_data = out

        val_loss = loss_function(out_data, y_data.squeeze().to(torch.int64))
        val_label = y_data.cpu()
        ys = [y_data]
        preds = [(out_data.argmax(-1)).float().cpu()]
        y, pred = torch.cat(ys, dim=0), torch.cat(preds, dim=0)
        pred = pred.reshape(-1, 1)
        accuracy = (pred == val_label).sum() / pred.shape[0]

        self.log("val_loss", val_loss)
        self.log("val_acc", accuracy)
    
    def test_step(self, data, batch_idx):
        # this is the test loop
        out = model(data)
        loss_function = CrossEntropyLoss(weight=class_weights).to(device) #weight=class_weight
        
        if hasattr(data, 'test_mask'):
            # If 'test_mask' attribute is present in data, use it for masking
            y_data = data.y[data.test_mask]
            out_data = out[data.test_mask]
        else:
            y_data = data.y
            out_data = out
        
        test_loss = loss_function(out_data, y_data.squeeze().to(torch.int64))
        test_label = y_data.cpu()
        ys = [y_data]
        preds = [(out_data.argmax(-1)).float().cpu()]
        percentages = torch.softmax(out_data, dim=-1).cpu().numpy() * 100

        y, pred = torch.cat(ys, dim=0), torch.cat(preds, dim=0)
        pred = pred.reshape(-1, 1)
        accuracy = (pred == test_label).sum() / pred.shape[0]

        self.log("test_acc", accuracy)

        return pred, y.squeeze(), percentages
    
    def test_epoch_end(self, outputs):
        # this function gives us in the outputs all accumulated pred and test_labels we returned in test_step
        # we transform the pred and test_label into a shape that the classification report can read
        global true_array, pred_array, percentage_array
        true_array = [outputs[i][1].cpu().numpy() for i in range(len(outputs))]
        pred_array = [outputs[i][0].cpu().numpy() for i in range(len(outputs))]
        percentage_array = [outputs[i][2] for i in range(len(outputs))]
        pred_array = np.concatenate(pred_array, axis=0 )
        true_array = np.concatenate(true_array, axis=0 )
        percentage_array = np.concatenate(percentage_array, axis=0)
        print(confusion_matrix(true_array, pred_array))
        print(classification_report(true_array, pred_array))

def last_version(MConfig):
    
    versions = os.listdir("DvcLiveLogger/"+ MConfig["name_of_save"] +"/checkpoints")
    versions.sort()
    vt=np.array([])
    for i in range(len(versions)):
        v=versions[i][11:-5]
        if v == '':
            v = 0
        else:
            v = int(v)
        vt=np.append(vt,v)

    version_dict = {version: tag for version, tag in zip(versions, vt)}
    sorted_versions = sorted(versions, key=lambda x: version_dict[x])
    return sorted_versions[-1]
        
def main(Container):
    global class_weights, device, model, Config, MConfig
    class_weights = Container.class_weights
    Config = Container.config
    importlib.reload(utils.config_dir)
    from utils.config_dir import Config
    MConfig=Config["main"]
    
    early_stop = EarlyStopping(monitor='val_acc',patience=MConfig["patience"], strict=False,verbose=False, mode='max')
    # val_checkpoint_acc = ModelCheckpoint(filename="max_acc-{epoch}-{step}-{val_acc:.3f}", monitor = "val_acc", mode="max")
    val_checkpoint_best_loss = ModelCheckpoint(filename="best_loss", monitor = "val_loss", mode="max")
    val_checkpoint_best_acc = ModelCheckpoint(filename="best_acc", monitor = "val_acc", mode="max")
    # val_checkpoint_loss = ModelCheckpoint(filename="min_loss-{epoch}-{step}-{val_loss:.3f}", monitor = "val_loss", mode="min")
    # latest_checkpoint = ModelCheckpoint(filename="latest-{epoch}-{step}", monitor = "step", mode="max",every_n_train_steps = 500,save_top_k = 1)
    #batchsizefinder = BatchSizeFinder(mode='power', steps_per_trial=3, init_val=2, max_trials=25, batch_arg_name='batch_size')
    #lr_finder = FineTuneLearningRateFinder(milestones=(5,10))
    # logger = TensorBoardLogger(save_file, name=name_of_save) # where the model saves the callbacks
    logger = DVCLiveLogger(run_name = MConfig["name_of_save"])
    # logger = None

    torch.manual_seed(MConfig["SEED"])
    # torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Container.add_device(device)
    output = Container.output

    train_size = int(Config["train/val/test"]["train"] * len(output))
    Temp_size = len(output) - train_size
    val_size = int(Config["train/val/test"]["val"]*Temp_size)
    test_size = Temp_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(output, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=MConfig["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(MConfig["batch_size"]/2), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    Container.loaders(train_loader, val_loader, test_loader)
    # mode
    if Config["graph"]["classif"] == "graph":
        model = GINE().double()
    elif Config["graph"]["classif"] == "node":
        if Config["main"]["loss"] == "BCE":
            model = NetBCE().double()#.to(device) 
        if Config["main"]["loss"] == "CE":
            model = NetCE().double()#.to(device)           
    #training
    trainer = pl.Trainer(logger=logger, max_epochs = MConfig["range_epoch"], callbacks=[val_checkpoint_best_loss,early_stop],accelerator='gpu',devices=1)#val_checkpoint_best_loss,latest_checkpoint, val_checkpoint_acc,val_checkpoint_loss
    trainer.fit(model, train_loader, val_loader)
    
    if Config["graph"]["classif"] == "graph":
        best_model = GINE().load_from_checkpoint("DvcLiveLogger/"+MConfig["name_of_save"]+"/checkpoints/" + last_version(MConfig)).double()
    elif Config["graph"]["classif"] == "node":
        if Config["main"]["loss"] == "BCE":
            best_model = NetBCE().load_from_checkpoint("DvcLiveLogger/"+MConfig["name_of_save"]+"/checkpoints/" + last_version(MConfig)).double()
        if Config["main"]["loss"] == "CE":
            best_model = NetCE().load_from_checkpoint("DvcLiveLogger/"+MConfig["name_of_save"]+"/checkpoints/" + last_version(MConfig)).double()

    Container.model(model,best_model)
    
    tester = pl.Trainer(accelerator='gpu',devices=1)
    tester.test(best_model, test_loader)
    
    Container.add_true_pred(true_array, pred_array, percentage_array)
    return Container

def just_test(Container):
    global class_weights, device, model
    model = Container.model_last
    device = Container.device
    class_weights = Container.class_weights
    tester = pl.Trainer(accelerator='gpu',devices=1)
    tester.test(model , Container.test_loader)
    Container.add_true_pred(true_array, pred_array, percentage_array)
    return Container