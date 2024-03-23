# %%
"""
Hugging Face BERT with custom classifier (PyTorch)

https://www.kaggle.com/code/angyalfold/hugging-face-bert-with-custom-classifier-pytorch/notebook
"""

import pandas as pd
from sklearn.model_selection import train_test_split

train_csv_path = './train.csv'
train_df = pd.read_csv(train_csv_path)

all_texts = train_df['text'].values.tolist()
all_labels = train_df['target'].values.tolist()

print("Tweets are loaded. Total # of tweets: {}.".format(len(all_texts)))
print("# of labels:")
print(train_df['target'].value_counts())

# As it turns out there are couple of tweets which occurs multiple times. Among those there are some whose labels aren't consistent throughout the occurrences.

frequent_tweets = {}
for t, l in zip(all_texts, all_labels):
    if all_texts.count(t) > 2:
        frequent_tweets[t] = [l] if t not in frequent_tweets else frequent_tweets[t] + [l]
        
print("The number of tweeets which appear multiple times: {}"
      .format(len(frequent_tweets.keys())))     

print("Tweets which have inconsistent labeling:")
print()

for t, ls in frequent_tweets.items():
    if not all(element == ls[0] for element in ls):
        print(t)
        print(ls)

# The amount of tweets with inconsistent labeling seems reasonably low so they can be fixed by hand. (Note: One could argue that deleting tweets with inconsistent labeling would be a better practice because modifing the input like that is an overreach, but for the sake of the example I go with releballing.)

# %%
should_be_real = [".POTUS #StrategicPatience is a strategy for #Genocide; refugees; IDP Internally displaced people; horror; etc. https://t.co/rqWuoy1fm4",
                 "#foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption",
                 "CLEARED:incident with injury:I-495  inner loop Exit 31 - MD 97/Georgia Ave Silver Spring"]

should_not_be_real = ["He came to a land which was engulfed in tribal war and turned it into a land of peace i.e. Madinah. #ProphetMuhammad #islam",
                     "Who is bringing the tornadoes and floods. Who is bringing the climate change. God is after America He is plaguing her",
                      "The Prophet (peace be upon him) said 'Save yourself from Hellfire even if it is by giving half a date in charity.'",
                     "Hellfire is surrounded by desires so be careful and donÛªt let your desires control you! #Afterlife",
                     "#Allah describes piling up #wealth thinking it would last #forever as the description of the people of #Hellfire in Surah Humaza. #Reflect",
                     "that horrible sinking feeling when youÛªve been at home on your phone for a while and you realise its been on 3G this whole time",
                     "To fight bioterrorism sir."]


def fix_labels(tweets_to_fix, correct_label):
    for i, (tweet, label) in enumerate(zip(all_texts, all_labels)):
        if any(tweet.startswith(t) for t in tweets_to_fix):
            all_labels[i] = correct_label

        
fix_labels(should_be_real, 1)
fix_labels(should_not_be_real, 0)

print("Relabeled {} tweets in total".format(len(should_be_real) + len(should_not_be_real)))

# %%
train_texts, val_texts, train_labels, val_labels = train_test_split(
    all_texts, all_labels,
    stratify = train_df['target']
)

print('Train data is read and split into training and validation sets.')
print('Size of train data (# of entries): {}'.format(len(train_texts)))
print('Size of validation data (# of entries): {}'.format(len(val_texts)))

# %%
# Data cleaning

"""
The most obvious step to take is to remove URLs as they are most likely just noise.

An additional consideration to take into account is that Hugging Face's tokenizer employs subword tokenization as detailed in their summary here. It essentialy means that if the tokenizer encounters a word which is unknown to it the word gets splitted into multiple tokens. Each new token gets the '##' prefix. For example: "annoyingly" becomes "annoying" + "##ly". Now it is easy to figure out which words are unknown to the model (just by searching for the '##' prefix) and thus gain ideas what sort of cleaning might worth implementing.

In this implementation URLs, @ links, non ascii characters are completely removed, the negation of some of the auxiliary verbs are fixed (eg.: shouldnt -> should not) and some of the personal pronouns (eg.: im -> i am)
"""

from ex_custom_classifier_helper import clean_tweet 


cleaned_train_texts = [clean_tweet(tweet) for tweet in train_texts]
print("Train tweets cleaned.")
cleaned_val_texts = [clean_tweet(tweet) for tweet in val_texts]
print("Validation tweets cleaned.")

# %%
from transformers import AutoTokenizer

model_name = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# The tokenizer's truncation=True setting ensures that the sequence of tokens is truncated if the sequence is longer than the maximal input length acceptable by the model. padding=True ensures that each sentence is padded to the longest sentence of the batch.
train_encodings = tokenizer(cleaned_train_texts, truncation=True, padding=True)
val_encodings = tokenizer(cleaned_val_texts, truncation=True, padding=True)
print('Train & validation texts encoded')

# %%
# Custom dataset

"""
PyTorch uses datasets and dataloaders to handle data (see their introductionary tutorial here https://pytorch.org/tutorials/beginner/basics/data_tutorial.html). It means that in order to make the handling of tweets straightforward a custom dataset has to be defined. (Named TweetDataset in this code)

A dataset is a data structure which makes it easy to iterate through the data in training and testing loops, therefore it needs to implement three methods of its base class (which is torch.utils.data.Dataset): __init__ (to initialize the dataset with the data), __len__ (to get the number of items in the dataset) and __getitem__ (to return the ith element of the dataset).
"""
import torch

class TweetDataset(torch.utils.data.Dataset):
    """
    Class to store the tweet data as PyTorch Dataset
    """
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        # an encoding can have keys such as input_ids and attention_mask
        # item is a dictionary which has the same keys as the encoding has
        # and the values are the idxth value of the corresponding key (in PyTorch's tensor format)
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
        
print(TweetDataset.__doc__)

# %%
# Custom model

"""
A pre-trained BERT model with custom classifier. 

The custom model consists of a pre-trained BERT model (a model which holds a semantical representation of English) and on the top of the BERT model there is a custom neural network which is trained to the specific task (tweet classification in this case). Therefore, it seems to be reasonable to have freeze_bert and unfreeze_bert methods apart from the mandatory __init__ and forward. Having this two additional methods makes it possible to sort of train the underlying BERT model and the custom classifier separately. (So train BERT and the custom head together, freeze BERT and then train the custom head on the classification task based on the previously trained BERT). The idea of freezing & unfreezing was taken from Milan Kalkenings' notebook.
"""
# device (turn on GPU acceleration for faster execution)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device used: {}.".format(device))

# %%
from torch import nn
from transformers import BertModel

in_features = 768 # it's 768 because that's the size of the output provided by the underlying BERT model

class BertWithCustomNNClassifier(nn.Module):
    """
    A pre-trained BERT model with a custom classifier.
    The classifier is a neural network implemented in this class.
    """
    
    def __init__(self, linear_size):
        super(BertWithCustomNNClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout1 = nn.Dropout()
        self.linear1 = nn.Linear(in_features=in_features, out_features=linear_size)
        self.batch_norm1 = nn.BatchNorm1d(num_features=linear_size)
        self.dropout2 = nn.Dropout(p=0.8)
        self.linear2 = nn.Linear(in_features=linear_size, out_features=1)
        self.batch_norm2 = nn.BatchNorm1d(num_features=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, tokens, attention_mask):
        bert_output = self.bert(input_ids=tokens, attention_mask=attention_mask)
        x = self.dropout1(bert_output[1])
        x = self.linear1(x)
        x = self.dropout2(x)
        x = self.batch_norm1(x)
        x = self.linear2(x)
        x = self.batch_norm2(x)
        return self.sigmoid(x)
        
    def freeze_bert(self):
        """
        Freezes the parameters of BERT so when BertWithCustomNNClassifier is trained
        only the wieghts of the custom classifier are modified.
        """
        for param in self.bert.named_parameters():
            param[1].requires_grad=False
    
    def unfreeze_bert(self):
        """
        Unfreezes the parameters of BERT so when BertWithCustomNNClassifier is trained
        both the wieghts of the custom classifier and of the underlying BERT are modified.
        """
        for param in self.bert.named_parameters():
            param[1].requires_grad=True

            
print(BertWithCustomNNClassifier.__doc__)

# %%
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def eval_prediction(y_batch_actual, y_batch_predicted):
    """Return batches of accuracy and f1 scores."""
    y_batch_actual_np = y_batch_actual.cpu().detach().numpy()
    y_batch_predicted_np = np.round(y_batch_predicted.cpu().detach().numpy())
    
    acc = accuracy_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np)
    f1 = f1_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np, average='weighted')
    
    return acc, f1

print(eval_prediction.__doc__)

# %%
# parameters
num_of_epochs = 1
learning_rate = 27e-6
batch_size = 16
hidden_layers = 8

print("Epochs: {}".format(num_of_epochs))
print("Learning rate: {:.6f}".format(learning_rate))
print("Batch size: {}".format(batch_size))
print("The number of hidden layers in the custom head: {}".format(hidden_layers))

# %%
model = BertWithCustomNNClassifier(linear_size=hidden_layers)
model.to(device)

# %%
from transformers import AdamW

# optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)
print('Initialized optimizer.')

# loss function
loss_fn = nn.BCELoss()
print('Initialized loss function.')

from torch.utils.data import DataLoader

# Dataset & dataloader
train_dataset = TweetDataset(train_encodings, train_labels)
val_dataset = TweetDataset(val_encodings, val_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
print('Created train & val datasets.')

# %%
def training_step(dataloader, model, optimizer, loss_fn, if_freeze_bert):
    """Method to train the model"""
    
    model.train()
    model.freeze_bert() if if_freeze_bert else model.unfreeze_bert()
      
    epoch_loss = 0
    size = len(dataloader.dataset)
 
    for i, batch in enumerate(dataloader):        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
    
        outputs = torch.flatten(model(tokens=input_ids, attention_mask=attention_mask))
                        
        optimizer.zero_grad()
        loss = loss_fn(outputs, labels.float())
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

            
print(training_step.__doc__)

# %%
def validation_step(dataloader, model, loss_fn):
    """Method to test the model's accuracy and loss on the validation set"""
    
    model.eval()
    model.freeze_bert()
    
    size = len(dataloader)
    f1, acc = 0, 0
    
    with torch.no_grad():
        for batch in dataloader:
            X = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y = batch['labels'].to(device)
                  
            pred = model(tokens=X, attention_mask=attention_mask)
            
            acc_batch, f1_batch = eval_prediction(y.float(), pred)                        
            acc += acc_batch
            f1 += f1_batch

        acc = acc/size
        f1 = f1/size
                
    return acc, f1
        
print(validation_step.__doc__)

# %%
from tqdm.auto import tqdm

tqdm.pandas()

best_acc, best_f1 = 0, 0
path = './best_model.pt'
if_freeze_bert = False

for i in tqdm(range(num_of_epochs)):
    print("Epoch: #{}".format(i+1))

    if i < 5:
        if_freeze_bert = False
        print("Bert is not freezed")
    else:
        if_freeze_bert = True
        print("Bert is freezed")
    
    training_step(train_loader, model,optimizer, loss_fn, if_freeze_bert)
    train_acc, train_f1 = validation_step(train_loader, model, loss_fn)
    val_acc, val_f1 = validation_step(val_loader, model, loss_fn)
    
    print("Training results: ")
    print("Acc: {:.3f}, f1: {:.3f}".format(train_acc, train_f1))
    
    print("Validation results: ")
    print("Acc: {:.3f}, f1: {:.3f}".format(val_acc, val_f1))
    
    if val_acc > best_acc:
        best_acc = val_acc    
        torch.save(model, path)

# %%
test_data = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')


clean_test_texts = [clean_tweet(tweet) for tweet in test_data['text'].values.tolist()]
test_encodings = tokenizer(clean_test_texts,
                           truncation=True, padding=True,
                           return_tensors='pt').to(device)

print("Encodings are ready.")



model = torch.load(path)
model.eval()
with torch.no_grad():
    predictions = model(tokens=test_encodings['input_ids'], attention_mask=test_encodings['attention_mask'])
    
binary_predictions = np.round(predictions.cpu().detach().numpy()).astype(int).flatten()
    
print("Predictions are ready.")

sample_submission['target'] = binary_predictions
sample_submission.to_csv('submission.csv', index=False)
print('Predictions are saved to submission.csv.')


