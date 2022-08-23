import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


# To visualize the process of training
def visualize_training_process(train_losses, train_accs, val_losses, val_accs, epochs):
    #　graph for checking accuracy
    plt.figure(figsize=(10,5))
    plt.plot(train_accs, label = 'train_acc')
    plt.plot(val_accs, label = 'val_acc')
    plt.xticks(np.arange(1, epochs+1, 1))
    plt.title('Accuracy Graph')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('./results/bert_epoch_5_acccuracy.png')
    plt.show()
    plt.close()

    #　graph for checking loss
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label = 'train_loss')
    plt.plot(val_losses, label = 'val_loss')
    plt.xticks(np.arange(1, epochs+1, 1))
    plt.title('Loss Graph')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('./results/bert_epoch_5_loss.png')
    plt.show()
    plt.close()
    
# To evaluate validation loss and accuracy in training loop
def caluculate_val_loss_acc(model, loss_fn, valid_dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    val_loss = 0.0
    val_correct = 0.0
    with torch.no_grad():
        for ids, mask, token_type_ids, target in tqdm(valid_dataloader):
            ids = torch.reshape(ids, (ids.shape[0], ids.shape[2]))
            mask = torch.reshape(mask, (mask.shape[0], mask.shape[2]))
            token_type_ids = torch.reshape(token_type_ids, (token_type_ids.shape[0], token_type_ids.shape[2]))
            
            ids = ids.to(device)
            mask = mask.to(device)
            token_type_ids = token_type_ids.to(device)
            
            output = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fn(output.cpu(), target)
            
            # calculate valid loss and accuracy
            # convert one-hot to label
            val_loss += loss.item() 
            val_pred = torch.argmax(output.cpu(), dim=-1)
            target_label = torch.argmax(target, dim=-1)
            val_correct += torch.sum(val_pred == target_label) 
            
        val_loss = val_loss / len(valid_dataloader.dataset)
        val_acc = val_correct / len(valid_dataloader.dataset)
            
    return val_loss, val_acc

# This function is for training loop
def train_model(epochs, train_dataloader, valid_dataloader, model, loss_fn, optimizer):
    # If you can use GPU, set up GPU. Otherwise, use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch+1, epochs))
        print("-----------------------------------")
        
        train_loss = 0.0
        train_correct = 0.0
        for ids, mask, token_type_ids, target in tqdm(train_dataloader):
            # We need to reshape data here, I dont know why
            ids = torch.reshape(ids, (ids.shape[0], ids.shape[2]))
            mask = torch.reshape(mask, (mask.shape[0], mask.shape[2]))
            token_type_ids = torch.reshape(token_type_ids, (token_type_ids.shape[0], token_type_ids.shape[2]))
            
            # send data to GPU
            ids = ids.to(device)
            mask = mask.to(device)
            token_type_ids = token_type_ids.to(device)
            target = target.to(device)
            
            # train and backforward
            optimizer.zero_grad()
            output = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fn(output, target) 
            loss.backward()
            optimizer.step()
            
            # calculate train loss and accuracy
            # convert one-hot to label
            train_loss += loss.item() 
            train_pred = torch.argmax(output, dim=-1)
            target_label = torch.argmax(target, dim=-1)
            train_correct += torch.sum(train_pred == target_label) 
            
        train_loss = train_loss / len(train_dataloader.dataset)
        train_acc = train_correct / len(train_dataloader.dataset)
        val_loss, val_acc = caluculate_val_loss_acc(model, loss_fn, valid_dataloader)
        
        print("Train Loss: {:.4f} Train Acc: {:.4f}".format(train_loss, train_acc))
        print("Valid Loss: {:.4f} Valid Acc: {:.4f}".format(val_loss, val_acc))
        train_losses.append(train_loss)
        train_accs.append(train_acc.cpu())
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    
    visualize_training_process(train_losses, train_accs, val_losses, val_accs, epochs)        
    return model

# this is for evaluating model using validation data
def evaluation(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    preds = []
    targets = []
    with torch.no_grad():
        for ids, mask, token_type_ids, target in tqdm(dataloader):
            ids = torch.reshape(ids, (ids.shape[0], ids.shape[2]))
            mask = torch.reshape(mask, (mask.shape[0], mask.shape[2]))
            token_type_ids = torch.reshape(token_type_ids, (token_type_ids.shape[0], token_type_ids.shape[2]))
            
            ids = ids.to(device)
            mask = mask.to(device)
            token_type_ids = token_type_ids.to(device)
            
            output = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            
            pred = torch.argmax(output.cpu(), dim=-1)
            target = torch.argmax(target, dim=-1)
            preds.extend(pred)
            targets.extend(target)
            
    return preds, targets

# This function is for training loop
def train_submit_model(epochs, dataloader, model, loss_fn, optimizer):
    # If you can use GPU, set up GPU. Otherwise, use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch+1, epochs))
        print("-----------------------------------")
        for ids, mask, token_type_ids, target in tqdm(dataloader):
            # We need to reshape data here, I dont know why
            ids = torch.reshape(ids, (ids.shape[0], ids.shape[2]))
            mask = torch.reshape(mask, (mask.shape[0], mask.shape[2]))
            token_type_ids = torch.reshape(token_type_ids, (token_type_ids.shape[0], token_type_ids.shape[2]))
            
            # send data to GPU
            ids = ids.to(device)
            mask = mask.to(device)
            token_type_ids = token_type_ids.to(device)
            target = target.to(device)
            
            # train and backforward
            optimizer.zero_grad()
            output = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fn(output, target) 
            loss.backward()
            optimizer.step()
       
    return model

def prediction(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    preds = []
    with torch.no_grad():
        for ids, mask, token_type_ids in tqdm(dataloader):
            ids = torch.reshape(ids, (ids.shape[0], ids.shape[2]))
            mask = torch.reshape(mask, (mask.shape[0], mask.shape[2]))
            token_type_ids = torch.reshape(token_type_ids, (token_type_ids.shape[0], token_type_ids.shape[2]))
            
            ids = ids.to(device)
            mask = mask.to(device)
            token_type_ids = token_type_ids.to(device)
            
            output = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            
            pred = torch.argmax(output.cpu(), dim=-1)
            preds.extend(pred.numpy())
            
    return preds