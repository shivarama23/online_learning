import splitfolders
import glob
import os
import numpy as np
import shutil
from pathlib import Path
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from datetime import date
from datasets import load_metric
from transformers import TrainingArguments, Trainer
import torch
import pandas as pd
import random

random.seed(21)


model_name = 'dmis-lab/biobert-base-cased-v1.1-mnli'
model_name_use = model_name.split('/')[-1]
print(model_name_use)

today = date.today()
td = today.strftime("%B %d, %Y").replace(' ','_').replace(',','')

class_category = [i.split('/')[-1] for i in glob.glob(r'Dataset_text/*')]

print(class_category)

# splitfolders.ratio(input=r'D:\NAFU\Rajeshwar\Solution_ImageClassifcation\Evaluation_Dataset',
#                   output=r'D:\NAFU\Rajeshwar\Solution_ImageClassifcation\Evaluation_Dataset',
#                   ratio=(0.9,0.0,0.1),seed=21)


# # Creating Train / Val / Test folders (One time use)
root_dir = f'dataset_for_pipeline_label_{len(class_category)}_{td}'
if not os.path.isdir(root_dir):
    os.mkdir(root_dir)

# for cat in class_category:
#     if not os.path.isdir(root_dir +'/train/'+cat):
#         os.makedirs(root_dir +'/train/'+cat)
#     if not os.path.isdir(root_dir +'/val/'+cat):
#         os.makedirs(root_dir +'/val/'+cat )
#     if not os.path.isdir(root_dir +'/test/'+cat):
#         os.makedirs(root_dir +'/test/'+cat)

# # Creating partitions of the data after shuffeling
# for cat in class_category:
#     print('CATEGORY: ',cat)

#     src = "Dataset_text/"+cat # Folder to copy images from

#     allFileNames = os.listdir(src)
#     np.random.shuffle(allFileNames)
#     train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
#                                                           [int(len(allFileNames)*0.7), int(len(allFileNames)*0.85)])


#     train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
#     val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
#     test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

#     print('Total images: ', len(allFileNames))
#     print('Training: ', len(train_FileNames))
#     print('Validation: ', len(val_FileNames))
#     print('Testing: ', len(test_FileNames))

#     # Copy-pasting images
#     for name in train_FileNames:
#         if os.path.getsize(name) > 0:

#             save_name = name.split('/')[-1]
#             dst = os.path.join(root_dir,'train/',cat)
#             dst_save_file = os.path.join(dst,save_name)
#             shutil.copy(name,dst_save_file)


#     for name in val_FileNames:
#         if os.path.getsize(name) > 0:
#             save_name = name.split('/')[-1]
#             dst = os.path.join(root_dir,'val/',cat)
#             dst_save_file = os.path.join(dst,save_name)
#             shutil.copy(name,dst_save_file)

#     for name in test_FileNames:
#         if os.path.getsize(name) > 0:
#             save_name = name.split('/')[-1]
#             dst = os.path.join(root_dir,'test/',cat)
#             dst_save_file = os.path.join(dst,save_name)
#             shutil.copy(name,dst_save_file)

# # ### CONSISTENCY IN Labelling :
    
# #     - Angiography_Images:0
# #     - Dialysis_Images:1
# #     - Lab_Reports_Images:2
# #     - Ot_Images:3
# #     - Discharge_text:4


# def read_doc_clf_split(split_dir):
#     split_dir = Path(split_dir)
#     # print(split_dir)
#     texts = []
#     labels = []
#     image_name = []
   
#     for cat in class_category:
#         # count= 0
#         for text_file in (split_dir/cat).iterdir():
#             texts.append(text_file.read_text(encoding='cp1252'))
#             if cat=="Angiography_text":
#                 labels.append(0)
#             elif cat=="Dialysis_text":
#                 labels.append(1)
#             elif cat=="Lab_Reports_text":
#                 labels.append(2)
#             elif cat=='Ot_text':
#                 labels.append(3)
#             else:
#                 labels.append(4)
                
#             name = os.path.basename(text_file).split('.')[0]
#             image_name.append(name)
                

#     return texts, labels,image_name


#Create the train-val-test split using stratified approach and use them in creating the Dataset object
train_texts, train_labels,train_image_name = read_doc_clf_split(f'{root_dir}/train/')
val_texts, val_labels,val_image_name = read_doc_clf_split(f'{root_dir}/val/')
test_texts, test_labels,test_image_name = read_doc_clf_split(f'{root_dir}/test/')

print(len(train_image_name) , len(val_texts) , len(test_texts))

#Create a mapping for your classes
label2id = {'Angiography_text':0,'Dialysis_text':1,'Lab_Reports_text':2, 'Ot_text':3,'Discharge_text':4}
id2label= { 0:'Angiography_text',1:'Dialysis_text',2:'Lab_Reports_text', 3:'Ot_text',4:'Discharge_text'}
print(label2id)

tokenizer = AutoTokenizer.from_pretrained(model_name,num_labels=len(class_category),
                                                           id2label=id2label,
                                                           label2id=label2id)

model_info = f'model_info_{td}'
if not os.path.isdir(model_info):
    os.mkdir(model_info)


tokenizer.save_pretrained(f'{model_info}/{model_name_use}_{td}_tokenizer/tokenizer/')

train_encodings = tokenizer(train_texts, truncation=True, padding=True,max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True,max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True,max_length=512)


class MultiClassification(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MultiClassification(train_encodings, train_labels)
val_dataset = MultiClassification(val_encodings, val_labels)
test_dataset = MultiClassification(test_encodings, test_labels)



metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


batch_size = 8
metric_name = "accuracy"

if not os.path.isdir(f'{model_info}/finetune_models'):
    os.mkdir(f'{model_info}/finetune_models')

args = TrainingArguments(
    output_dir=f'{model_info}/finetune_models/{model_name_use}_{td}_result/',
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir=f'{model_info}/finetune_models/{model_name_use}_{td}_log/',            # directory for storing logs
    logging_steps=10,
)


model = AutoModelForSequenceClassification.from_pretrained(model_name,ignore_mismatched_sizes = True,id2label=id2label,\
                                                          label2id=label2id)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
model.to(device)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset ,
    compute_metrics=compute_metrics      # evaluation dataset
)

trainer.train()
trainer.evaluate()

#Testing on testdataset

test_results = trainer.predict(test_dataset)

# # blind_results_predicted_label = list(np.argmax(blind_results.predictions,axis=1))

# test_results_predicted_label = torch.argmax(torch.softmax(torch.tensor(test_results.predictions),axis=1),axis=1)
# test_results_predicted_label = list(test_results_predicted_label.numpy())


# test_results_cnf_score = torch.max(torch.softmax(torch.tensor(test_results.predictions),axis=1),axis=1)
# test_results_cnf_score = list(test_results_cnf_score.values.numpy())

# pd.DataFrame(list(zip(test_image_name,test_dataset.labels,test_results_predicted_label,test_results_cnf_score)) \
#              ,columns=['ImageName','GroundTruth','PredictedLabel','ConfScore']).to_csv(f'{model_info}/{model_name_use}_Analysis_testdata_{td}.csv',index=False)
# global model

# #Define the model as a global variable
# def reload_model(path_of_new_model):
#     #change the model inside the function
#     #The model will change
#     model = AutoModelForSequenceClassification.from_pretrained(path_of_new_model)
#     tokenizer = AutoTokenizer.from_pretrained(path_of_new_model)
#     return "Done"

# @app.route("\predict"):
# def prediction():
#     model.predict()



# @app.route("\training"):
# def train_model():
#     #train your model here
#     #Saving the model
    
#     reload_model(path_of_saved_model)

