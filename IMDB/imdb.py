import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
else: 
    device = torch.device("cpu")
device = torch.device("cuda")

# import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased',return_dict = False)

# freeze all the parameters
for param in bert.parameters():
    param.requires_grad = False

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

train_url = "/data/kajay/CS822_project/IMDB/datasets/train.csv"
test_url = "/data/kajay/CS822_project/IMDB/datasets/test.csv"

def get_data(url):
    # load the training and test sets
    data = pd.read_csv(url, header = None, sep = "\t")
    data.columns = ["label", "reviews"]
    return data


def split_data(test_url):
    test_data = get_data(test_url)
    # split the training data to have validation sets
    test_review, val_review, test_label, val_label = train_test_split(test_data["reviews"], test_data["label"], 
                                                        test_size = 0.2, stratify = test_data["label"],random_state = 0)
    return test_review, val_review, test_label, val_label


def tokenize_encode(train_url, test_url):
    test_rev, val_rev, test_label, val_label = split_data(test_url)
    train_data = get_data(train_url)
    # tokenize and encode sequences in the training set
    tokens_train = tokenizer.batch_encode_plus(
        train_data["reviews"].tolist(),
        max_length = 300,
        padding = "max_length",
        truncation=True)

    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(
        val_rev.tolist(),
        max_length = 300,
        padding = "max_length",
        truncation=True)

    # tokenize and encode sequences in the test set
    tokens_test = tokenizer.batch_encode_plus(
        test_rev.tolist(),
        max_length = 300,
        padding = "max_length",
        truncation=True)
    return tokens_train, tokens_val, tokens_test


def to_tensor(train_url, test_url):
    tokens_train, tokens_val, tokens_test = tokenize_encode(train_url, test_url)
    test_review, val_review, test_label, val_label = split_data(test_url)
    train_data = get_data(train_url)
    
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_data["label"].tolist())

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(val_label.tolist())

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(test_label.tolist())
    
    # define a batch size
    batch_size = 32

    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)

    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)

    # dataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # wrap tensors
    val_data = TensorDataset(val_seq, val_mask, val_y)

    # sampler for sampling the data during validation
    val_sampler = SequentialSampler(val_data)

    # dataLoader for validation set
    val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)
    
    return train_dataloader, val_dataloader


class BERT_Arch(nn.Module):
    
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()

        self.bert = bert 
        
        # dropout layer
        self.dropout = nn.Dropout(0.1)
        
        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768,512)
        
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,2)

        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):
        #pass the inputs to the model  
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        
        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)
        
        # apply softmax activation
        x = self.softmax(x)

        return x
    

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

#if torch.cuda.device_count() > 1:
#    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
model = nn.DataParallel(model).cuda()

# push the model to GPU if available
model = model.to(device)

# define the optimizer
optimizer = AdamW(model.parameters(),lr = 1e-5)          # learning rate

# define the loss
cross_entropy = nn.NLLLoss()

# function to train the model
def train(train_url, test_url):
    train_dataloader, _ = to_tensor(train_url, test_url)
    model.train()   # set the model in training mode

    total_loss = 0
    
    # empty list to save model predictions
    total_preds=[]
    
    # iterate over batches
    for step,batch in enumerate(train_dataloader):
        
        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu if available
        batch = [r.to(device) for r in batch]
    
        sent_id, mask, labels = batch   # unpack the batch

        # clear previously calculated gradients 
        model.zero_grad()        

        # get model predictions for the current batch
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss += loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        #preds=preds.detach().cpu().numpy()   # use this if GPU is available
        preds = preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)
    
    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    #returns the loss and predictions
    return avg_loss, total_preds


# function for evaluating the model
def evaluate(train_url, test_url):
    
    _, val_dataloader = to_tensor(train_url, test_url)
    print("\nEvaluating...")
    
    # deactivate dropout layers
    model.eval()  # set the model in evaluation mode

    total_loss = 0
    
    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step,batch in enumerate(val_dataloader):
        
        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
                
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():
        
            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds,labels)

            total_loss = total_loss + loss.item()

            #preds = preds.detach().cpu().numpy()  # use this if GPU is available
            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader) 

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

def fine_tune(train_url, test_url, epochs):
    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses=[]
    valid_losses=[]

    # for each epoch
    for epoch in range(epochs):
        
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
        
        #train model
        train_loss, _ = train(train_url, test_url)
        
        #evaluate model
        valid_loss, _ = evaluate(train_url, test_url)
        
        #save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights.pt')
        
        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')
        
    return train_losses, valid_losses


train_losses, valid_losses = fine_tune(train_url, test_url, 30)
epochs = list(range(len(train_losses)))
# Plotting
plt.plot(epochs, train_losses, label = "Train losses")
plt.plot(epochs, valid_losses, label = "Validation losses")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.legend(loc="best")
plt.title("Losses for freezing all layers in IMDb")
plt.show()

plt.savefig("losses-for-freezing-all-layers-imdb.png")
