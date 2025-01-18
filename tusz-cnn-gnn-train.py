import torch
import os
import matplotlib.pyplot as plt
import dill as pickle
import collections
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy import signal
import pandas as pd
import time
import Model
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,confusion_matrix, roc_curve

import tusz_cnnmodel


seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])

def pre_signalfilter(datasetex, fs):
    datasetrow = datasetex.shape[0]
    datasetcolumn = datasetex.shape[1]
    datasetrank = datasetex.shape[2]
    datasetexf = np.zeros((datasetrow,datasetcolumn,datasetrank))
    b,a = signal.butter(4, 30, btype='lowpass', analog=False, fs=fs)
    for rank in range(datasetrank):
        for row in range(datasetrow):
            datasignal = datasetex[row,:,rank]
            datasignalf = signal.filtfilt( b,a,datasignal)
            datasetexf[row,:,rank]  = datasignalf
    return datasetexf

#可选择使用少量数据实验
def random(data_set):
    num_samples = data_set.shape[0] // 10
    random_indices = np.random.choice(data_set.shape[0], num_samples, replace=False)
    selected_data = data_set[random_indices, :, :]
    return selected_data

def process_pkl_data(data):
    samples = []
    for i in range(0, data.shape[1], 500):
        chunk = data[:, i:i + 500]
        if chunk.shape[1] == 500:
            samples.append(chunk)

    return np.array(samples)


def load_and_concatenate_pkl_files(folder_path):

    pkl_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl')]

    all_processed_data = []

    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            load_data = pickle.load(f)
            data = load_data.data

            processed_data = process_pkl_data(data)
            if processed_data.ndim == 3:

                all_processed_data.append(processed_data)
            else:

                print(f"Skipping {pkl_file} because processed data is not three-dimensional.")


    concatenated_data = np.concatenate(all_processed_data, axis=0)

    return concatenated_data


#ABSZ
absz_path = 'd:/dataSet/ABSZ/v1.5.2/raw_seizures'
absz_data = load_and_concatenate_pkl_files(absz_path)
# absz_data = random(absz_data)
print('absz_data.shape',absz_data.shape)
absz_label = np.zeros((absz_data.shape[0]))

#CPSZ
cpsz_path = 'd:/dataSet/CPSZ/v1.5.2/raw_seizures'
cpsz_data = load_and_concatenate_pkl_files(cpsz_path)
# cpsz_data = random(cpsz_data)
print('cpsz_data.shape',cpsz_data.shape)
cpsz_label = np.zeros((cpsz_data.shape[0]))

#FNSZ
fnsz_path = 'd:/dataSet/FNSZ/v1.5.2/raw_seizures'
fnsz_data = load_and_concatenate_pkl_files(fnsz_path)
# fnsz_data = random(fnsz_data)
print('fnsz_data.shape',fnsz_data.shape)
fnsz_label = np.zeros((fnsz_data.shape[0]))

#GNSZ
gnsz_path = 'd:/dataSet/GNSZ/v1.5.2/raw_seizures'
gnsz_data = load_and_concatenate_pkl_files(gnsz_path)
# gnsz_data = random(gnsz_data)
print('gnsz_data.shape',gnsz_data.shape)
gnsz_label = np.zeros((gnsz_data.shape[0]))

#SPSZ
spsz_path = 'd:/dataSet/SPSZ/v1.5.2/raw_seizures'
spsz_data = load_and_concatenate_pkl_files(spsz_path)
# spsz_data = random(spsz_data)
print('spsz_data.shape',spsz_data.shape)
spsz_label = np.zeros((spsz_data.shape[0]))

#TCSZ
tcsz_path = 'd:/dataSet/TCSZ/v1.5.2/raw_seizures'
tcsz_data = load_and_concatenate_pkl_files(tcsz_path)
# tcsz_data = random(tcsz_data)
print('tcsz_data.shape',tcsz_data.shape)
tcsz_label = np.zeros((tcsz_data.shape[0]))

#TNSZ
tnsz_path = 'd:/dataSet/TNSZ/v1.5.2/raw_seizures'
tnsz_data = load_and_concatenate_pkl_files(tnsz_path)
# tnsz_data = random(tnsz_data)
print('tnsz_data.shape',tnsz_data.shape)
tnsz_label = np.zeros((tnsz_data.shape[0]))

seizures_data = np.concatenate((absz_data, cpsz_data,fnsz_data,gnsz_data,spsz_data,tcsz_data,tnsz_data),axis=0)

#读取正常数据
normal_folder_path = 'e:/dataSet/tusz_normal/v1.5.2/raw_seizures'
normal_data = load_and_concatenate_pkl_files(normal_folder_path)
# normal_data = random(normal_data)
# print('normal_data.shape',normal_data.shape)

#==========================滤波
samples,channels,points=seizures_data.shape
seizures_data = seizures_data.reshape(channels,points,samples)
seizures_data = pre_signalfilter(seizures_data,250)
seizures_data = seizures_data.reshape(samples,channels,points)


samples,channels,points=normal_data.shape
normal_data = normal_data.reshape(channels,points,samples)
normal_data = pre_signalfilter(normal_data,250)
normal_data = normal_data.reshape(samples,channels,points)

print('seizures_data.shape',seizures_data.shape)
print('normal_data.shape',normal_data.shape)

#===========================================================标签
A_label = np.zeros((seizures_data.shape[0]))
B_label = np.zeros((normal_data.shape[0]))+1

print('A_label.shape',A_label.shape)
print('B_label.shape',B_label.shape)

time1 = time.time()
#===================================系数矩阵
chunks = torch.chunk(torch.from_numpy(seizures_data), chunks=20, dim=1)

array_list = []
for X in chunks:

    trainloader = DataLoader(dataset=X, batch_size=32, shuffle=False)

    cnn_model = tusz_cnnmodel.CNNModel().float().cuda()
    cnn_model.load_state_dict(torch.load('best_tusz_cnn_model_500.pth'))

    cnn_model.eval()
    predict=[]
    with torch.no_grad():
        for data in trainloader:
            if torch.cuda.is_available():
                pred = cnn_model.forward(data.float().cuda())
                max_values, _ = pred.max(dim=1)
                max_values_np = max_values.cpu().numpy()
                predict = np.concatenate((predict, max_values_np))
    array_list.append(predict)

print(len(array_list))
combined_array = np.array(array_list)
print(combined_array.shape)

time_01= time.time()

pearson_coefficient_matrix = np.corrcoef(combined_array,rowvar=True)
time_02 = time.time()
time_matrix=time_02-time_01
print('time_matrix',time_matrix)

flattened_matrix = pearson_coefficient_matrix.flatten()

threshold = np.percentile(flattened_matrix, 70)

print(threshold)

pearson_coefficient_matrix = (np.abs(pearson_coefficient_matrix) > threshold).astype(int)


np.fill_diagonal(pearson_coefficient_matrix, 0)
pearson_coefficient_matrix = torch.from_numpy(pearson_coefficient_matrix)

new_labels = ["FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
            "T3-C3", "C3-CZ", "CZ-C4", "C4-T4", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2"]
sns.heatmap(pearson_coefficient_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title(f"Non-Seizure Period")
plt.xticks(np.arange(len(new_labels)) + 0.5, new_labels, rotation=90)
plt.yticks(np.arange(len(new_labels)) + 0.5, new_labels, rotation=45)
plt.show()

#=================================================
for instance in range(seizures_data.shape[0]):
  seizures_data[instance] = StandardScaler().fit_transform(seizures_data[instance])

for instance in range(normal_data.shape[0]):
  normal_data[instance] = StandardScaler().fit_transform(normal_data[instance])

#==========================================================

all_data = np.concatenate((seizures_data, normal_data),axis=0)
all_label = np.concatenate((A_label,B_label),axis=0)

# print('all_data.shape',all_data.shape)
# print('all_label.shape',all_label.shape)

#分割数据集
X_train_temp, X_val_test, y_train_temp, y_val_test = train_test_split(
    all_data, all_label, test_size=0.2, random_state=42, shuffle=True
)

X_val, X_test, y_val, y_test = train_test_split(
    X_val_test, y_val_test, test_size=0.5, random_state=42, shuffle=True
)

train_data = X_train_temp
train_label = y_train_temp
val_data = X_val
val_label = y_val
test_data = X_test
test_label = y_test

train_data=torch.from_numpy(train_data)
# print(train_data.shape)
train_label=torch.from_numpy(train_label)
# print(train_label.shape)
val_data=torch.from_numpy(val_data)
val_label=torch.from_numpy(val_label)
test_data=torch.from_numpy(test_data)
test_label=torch.from_numpy(test_label)

dataset = TensorDataset(train_data, train_label)
trainloader = DataLoader(dataset, batch_size=64, shuffle=True)

valset = TensorDataset(val_data, val_label)
valloader = DataLoader(valset, batch_size=64, shuffle=True)

testset = TensorDataset(test_data, test_label)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

hidden_dim = 32  # 隐藏层的维度
input_dim = 500
output_dim = 2  # 输出维度
dropout_rate = 0.5

model = Model.Model(pearson_coefficient_matrix, train_data.shape[2], hidden_dim, output_dim)

if torch.cuda.is_available():
    model = model.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
if torch.cuda.is_available():
    model.cuda()

final_train_loss = []
final_val_acc = []
final_val_loss = []
wait = 0
patience = 10
best_f1 = None

for epoch in range(epochs):
    model.train()
    batch_train_loss = []
    all_train_preds = []
    all_train_targets = []

    for i, (data, target) in enumerate(trainloader):
        if torch.cuda.is_available():
            data, target = data.float().cuda(), target.cuda()

        pred = model(data)
        pred = pred.type(torch.float32)

        target = target.type(torch.int64)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_train_loss.append(loss.item())

        all_train_preds.extend(pred.argmax(dim=1).cpu().numpy())
        all_train_targets.extend(target.cpu().numpy())

    final_train_loss.append(np.mean(batch_train_loss))

    train_accuracy = accuracy_score(all_train_targets, all_train_preds)
    train_precision = precision_score(all_train_targets, all_train_preds)
    train_recall = recall_score(all_train_targets, all_train_preds)
    train_f1 = f1_score(all_train_targets, all_train_preds)

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {final_train_loss[epoch]:.4f}')
        print(
              f'Train Accuracy: {train_accuracy:.4f}, '
              f'Train Precision: {train_precision:.4f}, '
              f'Train Recall: {train_recall:.4f}, '
              f'Train F1 Score: {train_f1:.4f}')


    model.eval()
    batch_val_loss = []
    all_val_preds = []
    all_val_targets = []
    output = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(valloader):
            if torch.cuda.is_available():
                data= data.float().cuda()
                target = target.type(torch.int64).cuda()

            pred = model(data)
            val_loss = criterion(pred, target)
            batch_val_loss.append(val_loss.item())

            all_val_preds.extend(pred.argmax(dim=1).cpu().numpy())
            all_val_targets.extend(target.cpu().numpy())

            same_elements = pred.argmax(dim=1).cpu() == target.cpu()
            count_same = same_elements.sum().item()
            output += count_same

    final_val_loss.append(np.mean(batch_val_loss))

    val_accuracy = accuracy_score(all_val_targets, all_val_preds)
    val_precision = precision_score(all_val_targets, all_val_preds)
    val_recall = recall_score(all_val_targets, all_val_preds)
    val_f1 = f1_score(all_val_targets, all_val_preds)

    acc = output / len(valloader.dataset)
    final_val_acc.append(acc)

    if best_f1 is None or val_f1 > best_f1:
        best_f1 = val_f1
        print('best model')
        best_model = model
        torch.save(model.state_dict(), 'best_tusz_cnn_gcn_model.pth')

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Val Loss: {final_val_loss[epoch]:.4f}')
        print(
              f'Validation Accuracy: {val_accuracy:.4f}, '
              f'Validation Precision: {val_precision:.4f}, '
              f'Validation Recall: {val_recall:.4f}, '
              f'Validation F1 Score: {val_f1:.4f}, ')
        print('======================================================')



time2 = time.time()
plt.figure(figsize=(10, 6))
plt.plot(final_train_loss, label='Training Loss')
plt.plot(final_val_loss, label='Validation Loss')
plt.plot(final_val_acc,label='accuracy')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

def gcntest(test_data, test_label):
    model = Model.Model( pearson_coefficient_matrix, train_data.shape[2], hidden_dim,output_dim)
    model.load_state_dict(torch.load('best_tusz_cnn_gcn_model.pth'))
    model.cuda()
    model.eval()
    output = 0
    predict = []
    all_probs = []
    with torch.no_grad():
        for i, (data, target) in enumerate(testloader):
            if torch.cuda.is_available():
                pred = model(data.float().cuda())
                if pred.dim() == 1:
                    pred = pred.unsqueeze(0)
                probs = torch.softmax(pred, dim=1).cpu().numpy()
                pred = torch.max(pred.cpu(), dim=1)
                same_elements = pred.indices == target
                count_same = same_elements.sum().item()
                output += count_same
                value_list = [item.item() if item.numel() == 1 else item.numpy().tolist() for item in pred.indices]
                predict += value_list
                all_probs.extend(probs[:, 1])

    accuracy = accuracy_score(test_label.numpy(), predict)
    recall = recall_score(test_label.numpy(), predict)
    f1 = f1_score(test_label.numpy(), predict)
    auc = roc_auc_score(test_label.numpy(), all_probs)

    cm = confusion_matrix(test_label.numpy(), predict)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    return accuracy, sensitivity, specificity, recall, f1, auc, test_label.numpy(), all_probs

time3 = time.time()
accuracy, sensitivity, specificity, recall, f1, auc, test_labels, all_probs = gcntest(test_data, test_label)
print("test_data.shape", test_data.shape)
time4 = time.time()
print('耗时:', time2 - time1)
print('测试耗时:', time4 - time3)
print('正确率:', accuracy)
print('召回率:', recall)
print('灵敏度:', sensitivity)
print('特异性:', specificity)
print('f1_score:', f1)
print('AUC:', auc)

fpr, tpr, thresholds = roc_curve(test_labels, all_probs)
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

auc_results = {
    'accuracy': accuracy,
    'sensitivity': sensitivity,
    'specificity': specificity,
    'recall': recall,
    'f1_score': f1,
    'auc': auc,
    'fpr': fpr.tolist(),
    'tpr': tpr.tolist(),
    'thresholds': thresholds.tolist()
}

df = pd.DataFrame({
    'accuracy': [accuracy],
    'sensitivity': [sensitivity],
    'specificity': [specificity],
    'recall': [recall],
    'f1_score': [f1],
    'auc': [auc]
})

fpr_tpr_df = pd.DataFrame({
    'fpr': fpr,
    'tpr': tpr,
    'thresholds': thresholds
})

with pd.ExcelWriter('GCNmodel_tusz.xlsx') as writer:
    df.to_excel(writer, index=False, sheet_name='Metrics')
    fpr_tpr_df.to_excel(writer, index=False, sheet_name='ROC Data')