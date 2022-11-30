from email.utils import format_datetime
from datasets import load_dataset
from transformers import AutoTokenizer,AdamW
from transformers import AutoModelForSequenceClassification
import numpy as np
import evaluate
import torch
from transformers import TrainingArguments, Trainer
from transformers import XLNetTokenizer, XLNetModel
from torch.nn import CrossEntropyLoss
import time
from datasets import Dataset
from torch.utils.data import TensorDataset

import time
import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle

class XLNetForMultiLabelSequenceClassification(torch.nn.Module):
  
  def __init__(self, num_labels=2):
    super(XLNetForMultiLabelSequenceClassification, self).__init__()
    self.num_labels = num_labels
    self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
    self.classifier = torch.nn.Linear(768, num_labels)

    torch.nn.init.xavier_normal_(self.classifier.weight)

  def forward(self, input_ids, token_type_ids=None,\
              attention_mask=None, labels=None):
       
    # last hidden layer
    
    last_hidden_state = self.xlnet(input_ids=input_ids,\
                                   attention_mask=attention_mask,\
                                   token_type_ids=token_type_ids
                                  )
    
    # pool the outputs into a mean vector
    mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
    logits = self.classifier(mean_last_hidden_state)
    
    

    if labels is not None:
      loss = torch.FloatTensor([CrossEntropyLoss()(logits, labels)])
    
      return loss
    else:
      return logits
    
  def pool_hidden_state(self, last_hidden_state):
    """
    Pool the output vectors into a single mean vector 
    """
    last_hidden_state = last_hidden_state[0]
    mean_last_hidden_state = torch.mean(last_hidden_state, 1)
    return mean_last_hidden_state

from tqdm import tqdm, trange
def train(model, num_epochs,\
          optimizer,\
          train_dataloader, valid_dataloader,\
          model_save_path,\
          train_loss_set=[], valid_loss_set = [],\
          lowest_eval_loss=None, start_epoch=0,\
          device="cpu"
          ):
  """
  Train the model and save the model with the lowest validation loss
  """
  # We'll store a number of quantities such as training and validation loss, 
  # validation accuracy, and timings.
  training_stats = []
  # Measure the total training time for the whole run.
  total_t0 = time.time()

  model.to(device)

  # trange is a tqdm wrapper around the normal python range
  for i in trange(num_epochs, desc="Epoch"):
    # if continue training from saved model
    actual_epoch = start_epoch + i

    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set. 
    print("")
    print('======== Epoch {:} / {:} ========'.format(actual_epoch, num_epochs))
    print('Training...')
    
    # Measure how long the training epoch takes.
    t0 = time.time()
    
    # Set our model to training mode (as opposed to evaluation mode)
    model.train()

    # Tracking variables
    tr_loss = 0
    num_train_samples = 0

    # Train the data for one epoch
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        loss = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss.requires_grad = True
        # store train loss
        tr_loss += loss.item()
        num_train_samples += b_labels.size(0)
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()
        #scheduler.step()

    # Update tracking variables
    epoch_train_loss = tr_loss/num_train_samples
    train_loss_set.append(epoch_train_loss)

#     print("Train loss: {}".format(epoch_train_loss))
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(epoch_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
    
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.
    
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()
    
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Tracking variables 
    eval_loss = 0
    num_eval_samples = 0

    # Evaluate data for one epoch
    for batch in valid_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate validation loss
            loss = model(input_ids=b_input_ids, attention_mask=b_input_mask)
            # store valid loss
            loss=torch.argmax(loss,dim=1)
            eval_loss += sum(loss.cpu().numpy()==b_labels.cpu().numpy())
            num_eval_samples += b_labels.size(0)

    epoch_eval_loss = eval_loss/num_eval_samples
    valid_loss_set.append(epoch_eval_loss)

#     print("Valid loss: {}".format(epoch_eval_loss))
    
    # Report the final accuracy for this validation run.
#     avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
#     print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
#     avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(epoch_eval_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': actual_epoch,
            'Training Loss': epoch_train_loss,
            'Valid. Loss': epoch_eval_loss,
#             'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

    
    if lowest_eval_loss == None:
      lowest_eval_loss = epoch_eval_loss
      # save model
      save_model(model, model_save_path, actual_epoch,\
                 lowest_eval_loss, train_loss_set, valid_loss_set)
    else:
      if epoch_eval_loss < lowest_eval_loss:
        lowest_eval_loss = epoch_eval_loss
        # save model
        save_model(model, model_save_path, actual_epoch,\
                   lowest_eval_loss, train_loss_set, valid_loss_set)
  
  print("")
  print("Training complete!")

  print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
  return model, train_loss_set, valid_loss_set, training_stats


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# function to save and load the model form a specific epoch
def save_model(model, save_path, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist):
  """
  Save the model to the path directory provided
  """
  model_to_save = model.module if hasattr(model, 'module') else model
  checkpoint = {'epochs': epochs, \
                'lowest_eval_loss': lowest_eval_loss,\
                'state_dict': model_to_save.state_dict(),\
                'train_loss_hist': train_loss_hist,\
                'valid_loss_hist': valid_loss_hist
               }
  torch.save(checkpoint, save_path)
  print("Saving model at epoch {} with validation loss of {}".format(epochs,\
                                                                     lowest_eval_loss))
  return
  
def load_model(save_path):
  """
  Load the model from the path directory provided
  """
  checkpoint = torch.load(save_path)
  model_state_dict = checkpoint['state_dict']
  model = XLNetForMultiLabelSequenceClassification(num_labels=model_state_dict["classifier.weight"].size()[0])
  model.load_state_dict(model_state_dict)

  epochs = checkpoint["epochs"]
  lowest_eval_loss = checkpoint["lowest_eval_loss"]
  train_loss_hist = checkpoint["train_loss_hist"]
  valid_loss_hist = checkpoint["valid_loss_hist"]
  
  return model, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist


def tokenize_inputs(text_list, tokenizer, num_embeddings=120):
    """
    Tokenizes the input text input into ids. Appends the appropriate special
    characters to the end of the text to denote end of sentence. Truncate or pad
    the appropriate sequence length.
    """
    # tokenize the text, then truncate sequence to the desired length minus 2 for
    # the 2 special characters
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t)[:num_embeddings-2], text_list))
    # convert tokenized text into numeric ids for the appropriate LM
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # append special token "<s>" and </s> to end of sentence
    input_ids = [tokenizer.build_inputs_with_special_tokens(x) for x in input_ids]
    # pad sequences
    input_ids = pad_sequences(input_ids, maxlen=num_embeddings, dtype="long", truncating="post", padding="post")
    return input_ids

def create_attn_masks(input_ids):
    """
    Create attention masks to tell model whether attention should be applied to
    the input id tokens. Do not want to perform attention on padding tokens.
    """
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks

def check_name(names):
    wrong={'jiejie':['jj','jojo'],'beryl':['barrel'],'debt':['death'],'flandre':['andre'],
    'meiko':['mako'], 'tl':['liquid']}
    for i in wrong:
        if i in names:
            names+=wrong[i]
    return names



class twobert(torch.nn.Module):
    def __init__(self):
        super(twobert, self).__init__()
        self.bert1 = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=1)
        self.bert2 = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=1)


    def forward(self, input_ids, token_type_ids=None,\
                attention_mask=None,labels=None):
        
        
        res1 = self.bert1(input_ids=input_ids,\
                                    attention_mask=attention_mask,\
                                    token_type_ids=token_type_ids
                                    )
        res2 = self.bert2(input_ids=input_ids,\
                                    attention_mask=attention_mask,\
                                    token_type_ids=token_type_ids
                                    )
        return res1, res2




# tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased",model_max_length=400, do_lower_case=True)

# def tokenize_function(examples):
#     temp=tokenizer([ row[0] for row in examples["text"]], padding="max_length", truncation=True)
#     temp1=tokenizer([ row[1] for row in examples["text"]], padding="max_length", truncation=True)
#     for i in temp:
#         for j in range(len(temp[i])):
#             temp[i][j]=[temp[i][j],temp1[i][j]]
    
#     return temp

def tokenize_function(examples):
    
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def modify_text(sentence,b_names,r_names):
    i=0
    selected_b=[]
    selected_r=[]
    selected=[]
    sentence=sentence.replace('\n',' ').replace('\r',' ')
    sentence=re.sub(r"\d{1,2}\:\d{1,2}", "", sentence)
    words=sentence.split(' ')
    b_names=check_name(b_names)
    r_names=check_name(r_names)
    while i < len(words):
        
        if words[i].lower() in b_names:
            selected_b+=words[i:i+10]

            i+=10
        elif words[i].lower() in r_names:
            selected_r+=words[i:i+10]
            i+=10
        # if words[i].lower() in b_names or words[i].lower() in r_names:
        #     selected+=words[i:i+10]

        #     i+=10
        else:
            i+=1
    return [' '.join(selected_b),' '.join(selected_r)]
    # return ' '.join(selected)





device = "cuda:0" if torch.cuda.is_available() else "cpu"
# data_files = {"train": "train.csv","test": "test.csv"}
# dataset = load_dataset("./", data_files=data_files)

with open('total_onemin.pkl', 'rb') as f:
    data_dict = pickle.load(f)

dataset = Dataset.from_dict(data_dict)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", do_lower_case=True,model_max_length=400)
# dataset = dataset.map(lambda example: {'labels': 0 if example["team_one_name"].strip()==example["win_team"].strip() else 1, 
#  'clean': modify_text(example["text"].lower(),[example["team_one_player4"].lower(), example["team_one_player2"].lower(), example["team_one_player5"].lower(), example["team_one_player1"].lower(), example["team_one_color"].lower(), example["team_one_name"].lower(), example["team_one_player3"].lower()],
#     [example["team_two_player9"].lower(), example["team_two_player6"].lower(), example["team_two_color"].lower(), example["team_two_name"].lower(), example["team_two_player7"].lower(), example["team_two_player8"].lower(), example["team_two_player10"].lower()])}
#     ,remove_columns=['win_team',"team_one_player4", "team_two_player9", "team_one_player2", "team_one_player5", "team_two_player6", "team_two_player10", "team_one_player1", "team_two_color", 
#     "video_link", "team_one_color", "team_one_name", "team_one_player3", "team_two_name", "team_two_player7", "team_two_player8"])
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets=tokenized_datasets.train_test_split(test_size=0.1)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_test_dataset = tokenized_datasets["test"]

del dataset
# traindata=pd.read_csv("./train.csv")
# for i in range(len(traindata)):
#     traindata['win_team'][i]=0 if traindata["team_one_name"][i].strip()==traindata["win_team"][i].strip() else 1
# test=pd.read_csv("./test.csv")
# for i in range(len(test)):
#     test['win_team'][i]=0 if test["team_one_name"][i].strip()==test["win_team"][i].strip() else 1
# t_sentences = traindata.text.values
# t_labels = traindata.win_team.values.astype(float)

# te_sentences = test.text.values
# te_labels = test.win_team.values.astype(float)



# t_input_ids = tokenize_inputs(t_sentences, tokenizer, num_embeddings=120)
# t_attention_masks = create_attn_masks(t_input_ids)
# t_input_ids = torch.from_numpy(t_input_ids)
# t_attention_masks = torch.tensor(t_attention_masks)
# t_labels = torch.tensor(t_labels,dtype=torch.long)
# t_dataset = TensorDataset(t_input_ids, t_attention_masks, t_labels)
# te_input_ids = tokenize_inputs(te_sentences, tokenizer, num_embeddings=120)
# te_attention_masks = create_attn_masks(te_input_ids)
# te_input_ids = torch.from_numpy(te_input_ids)
# te_attention_masks = torch.tensor(te_attention_masks)
# te_labels = torch.tensor(te_labels,dtype=torch.long)
# te_dataset = TensorDataset(te_input_ids, te_attention_masks, te_labels)

# model = twobert().to(device)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=1).to(device)
# model = XLNetForMultiLabelSequenceClassification(num_labels=2).to(device)

# metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits,labels=eval_pred
    # predictions = np.argmax(logits, axis=-1)
    # lossfc=torch.nn.CrossEntropyLoss()
    lossfc =torch.nn.MSELoss()
    return {'MSE loss':lossfc(torch.tensor(logits), torch.tensor(labels))}

class CustomTrainer(Trainer):
    #TODO:add previouse rateinfor
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        res1= model(**inputs)
        # loss_fct = torch.nn.CrossEntropyLoss()
        loss_fct =torch.nn.MSELoss()
        loss = loss_fct(res1.logits, labels)
        return (loss, res1) if return_outputs else loss
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.get("labels")
    #     # forward pass
    #     res1,res2 = model(**inputs)
    #     loss_fct = torch.nn.CrossEntropyLoss()
    #     loss = loss_fct(torch.cat([res1.logits,res2.logits],dim=1), labels)
    #     temp=res1
    #     temp.logits=torch.cat([res1.logits,res2.logits],dim=1)
    #     return (loss, temp) if return_outputs else loss


training_args = TrainingArguments(save_strategy="no",output_dir="test_trainer", evaluation_strategy="epoch",logging_strategy="epoch",num_train_epochs=50,per_device_train_batch_size=8)
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=small_test_dataset,
    eval_dataset=small_test_dataset,
    compute_metrics=compute_metrics,
)
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=small_train_dataset,
#     eval_dataset=small_test_dataset,
#     compute_metrics=compute_metrics,
# )
trainer.train()
model.save_pretrained(r"D:\bert\replacement_half_inter_partial")
# model = XLNetForMultiLabelSequenceClassification(num_labels=2)
# num_epochs = 200
# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# model_save_path = output_model_file = r"D:\bert\fine_xlnet\xlnet.bin"
# optimizer = AdamW(model.parameters(),
#                   lr = 2e-5, # args.learning_rate - default is 5e-5
#                   # eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
#                  weight_decay=0.01,
#                  correct_bias=False
#                 )
# train_dataloader = DataLoader(
#             t_dataset,  # The training samples.
#             sampler = RandomSampler(t_dataset), # Select batches randomly
#             batch_size = 16 # Trains with this batch size.
#         )

# # For validation the order doesn't matter, so we'll just read them sequentially.
# validation_dataloader = DataLoader(
#             te_dataset, # The validation samples.
#             sampler = SequentialSampler(te_dataset), # Pull out batches sequentially.
#             batch_size = 16 # Evaluate with this batch size.
#         )
# # model_save_path = '/content/drive/My Drive/Disaster_Tweets/XLNet_tweet_classification_model/xlnet_tweet.bin'
# model, train_loss_set, valid_loss_set, training_stats = train(model=model,\
#                                                               num_epochs=num_epochs,\
#                                                               optimizer=optimizer,\
#                                                               train_dataloader=train_dataloader,\
#                                                               valid_dataloader=validation_dataloader,\
#                                                               model_save_path=model_save_path,\
#                                                               device=device
#                                                               )
