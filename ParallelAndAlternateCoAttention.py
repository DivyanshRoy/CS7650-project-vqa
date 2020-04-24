import numpy as np
import torch 
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt 

train_image = np.load("../preprocessed_data/transformed_train_image_features_vgg.npy")
print(train_image.shape)

val_image = np.load("../preprocessed_data/transformed_val_image_features_vgg.npy")
print(val_image.shape)

train_question = np.load("../preprocessed_data/new_filtered_train_questions.npy")
train_answer = np.load("../preprocessed_data/new_filtered_train_answers.npy").astype(int)
print(train_question.shape)
print(train_answer.shape)

val_question = np.load("../preprocessed_data/new_filtered_val_questions.npy")
val_answer = np.load("../preprocessed_data/new_filtered_val_answers.npy").astype(int)
print(val_question.shape)
print(val_answer.shape)


class ParallelCoAttention(nn.Module):
  def __init__(self, d, t, k, vocab_size, dropout):
    super(ParallelCoAttention, self).__init__()
    self.d = d 
    self.embedding_dim = d
    self.t = t
    self.k = k
    self.vocab_size = vocab_size
    self.dropout = nn.Dropout(dropout)

    self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
    self.W_b = nn.Linear(self.d, self.d)
    self.W_q = nn.Linear(self.d, self.k)
    self.W_v = nn.Linear(self.d, self.k)
    self.w_hv = nn.Linear(self.k, 1)
    self.w_hq = nn.Linear(self.k, 1)
    self.tanh = nn.Tanh()
    self.softmax = nn.Softmax(dim =1)


  def forward(self, questions, images):
    questions = self.embedding(questions)
    x = self.dropout(self.W_b(questions))
    C = self.tanh(torch.bmm(x,images)) 

    H_v = self.tanh(self.dropout(self.W_v(torch.transpose(images, 1, 2)) + torch.bmm(torch.transpose(C,1,2),self.W_q(questions)))) 
    a_v = self.softmax(self.dropout(self.w_hv(H_v)))
    a_v = torch.transpose(a_v, 1,2) 
    v_hat = torch.sum(a_v * images, axis= 2) 

    H_q = self.tanh(self.dropout(self.W_q(questions) + torch.bmm(C,self.W_v(torch.transpose(images, 1, 2))))) 
    a_q = self.softmax(self.dropout(self.w_hq(H_q)))
    q_hat = torch.sum(a_q*questions, axis =1) 
    return (q_hat, v_hat)

class AlternateCoAttention(nn.Module):
  def __init__(self, d, k, vocab_size, dropout):
    super(AlternateCoAttention, self).__init__()
    self.d = d
    self.k = k
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(self.vocab_size, self.d)
    self.dropout = nn.Dropout(dropout)

    self.W_x_1 = nn.Linear(self.d, self.k)
    self.W_g_1 = nn.Linear(self.d, self.k)
    self.W_hx_1 = nn.Linear(self.k, 1)

    self.W_x_2 = nn.Linear(self.d, self.k)
    self.W_g_2 = nn.Linear(self.d, self.k)
    self.W_hx_2 = nn.Linear(self.k, 1)

    self.W_x_3 = nn.Linear(self.d, self.k)
    self.W_g_3 = nn.Linear(self.d, self.k)
    self.W_hx_3 = nn.Linear(self.k, 1)

    self.tanh = nn.Tanh()
    self.softmax = nn.Softmax(dim = 1)

  def forward(self, questions, images):
    questions = self.embedding(questions)
    H_1 = self.tanh(self.dropout(self.W_x_1(questions)))
    a_x_1 = self.softmax(self.dropout(self.W_hx_1(H_1))) 
    x_hat_1 = torch.sum(a_x_1 * questions, axis=1) 

    H_2 = self.tanh(self.dropout(self.W_x_2(torch.transpose(images,1,2)) + self.W_g_2(x_hat_1).unsqueeze(1)))
    a_x_2 = self.softmax(self.dropout(self.W_hx_2(H_2)))
    x_hat_2 = torch.sum(a_x_2*torch.transpose(images,1,2), axis=1)

    H_3 = self.tanh(self.dropout(self.W_x_3(questions) + self.W_g_3(x_hat_2).unsqueeze(1)))
    a_x_3 = self.softmax(self.dropout(self.W_hx_3(H_3)))
    x_hat_3 = torch.sum(a_x_3*questions, axis=1)
    return (x_hat_2, x_hat_3)

class AnswerGeneration(nn.Module):
  def __init__(self, d, d_prime, dropout):
    super(AnswerGeneration, self).__init__()
    self.d = d
    self.d_prime = d_prime
    self.dropout = nn.Dropout(dropout)

    self.tanh = nn.Tanh()
    self.softmax = nn.Softmax(dim = 1)
    self.W = nn.Linear(self.d, self.d_prime)
    self.W_h = nn.Linear(self.d_prime,  104) 

  def forward(self, q_hat, v_hat):
    h = self.tanh(self.dropout(self.W(q_hat + v_hat)))
    return self.W_h(h)

class MainModel(nn.Module):
  def __init__(self, d, t, k, d_prime, vocab_size, dropout):
    super(MainModel, self).__init__()
    self.d = d
    self.t = t
    self.k = k 
    self.d_prime = d_prime
    self.vocab_size = vocab_size
    self.dropout = dropout

    self.parallel = ParallelCoAttention(self.d, self.t, self.k, self.vocab_size, self.dropout)
    self.alternate = AlternateCoAttention(self.d, self.k, self.vocab_size, self.dropout)
    self.answer = AnswerGeneration(self.d, self.d_prime, self.dropout)

  def forward(self, questions, images):
    q_hat_p, v_hat_p = self.parallel(questions, images)
    v_hat_a, q_hat_a = self.alternate(questions, images)
    answer_p = self.answer(q_hat_p, v_hat_p)
    answer_a = self.answer(q_hat_a, v_hat_a)
    return (answer_p, answer_a)

d = 512
t = 25
k = 512
d_prime = 128 
vocab_size = 4400
dropout = 0.25

model = MainModel(d, t, k, d_prime, vocab_size, dropout)
#model.load_state_dict(torch.load("../output/model.pt"))


tensor_x = torch.Tensor(train_question).type(torch.long)
tensor_y = torch.Tensor(train_image).type(torch.float)
tensor_z = torch.Tensor(train_answer).type(torch.long).squeeze()

trainset = data.TensorDataset(tensor_x, tensor_y, tensor_z)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 5000, shuffle = True, num_workers=2)

tensor_x = torch.Tensor(val_question).type(torch.long)
tensor_y = torch.Tensor(val_image).type(torch.float)
tensor_z = torch.Tensor(val_answer).type(torch.long).squeeze()

valset = data.TensorDataset(tensor_x, tensor_y, tensor_z)
valloader = torch.utils.data.DataLoader(valset, batch_size = 5000, shuffle = True, num_workers = 2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=4e-4, weight_decay=1e-4, momentum=0.99)

def get_accuracy(predictions, labels):
  predictions = F.softmax(predictions,dim=1)
  predictions = torch.max(predictions, axis=1)[1]
  ab = torch.abs(predictions-labels)
  ab = ab.detach().numpy()
  mn = np.minimum(ab, 1)
  eq = 1-mn
  correct = np.sum(eq)
  total = eq.shape[0]
  return correct, total

train_loss_plot = []
val_loss_plot = []

for epoch in range(150):  # loop over the dataset multiple times

    running_loss = 0.0
    correct = 0
    total = 0
    total_loss = 0.0
    model.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        questions, images, labels = data
      
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs_p, outputs_a = model(questions, images)
        batch_correct, batch_total = get_accuracy(outputs_a, labels)
        correct += batch_correct
        total += batch_total
        loss = criterion(outputs_a, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        total_loss += running_loss
        running_loss = 0.0
    

    running_val_loss = 0.0
    total_val_loss = 0.0
    val_correct = 0
    val_total = 0
    model.eval()
    with torch.no_grad():
      for i, data in enumerate(valloader, 0):
          # get the inputs; data is a list of [inputs, labels]
          questions, images, labels = data

          # forward + backward + optimize
          outputs_p, outputs_a = model(questions, images)
          batch_correct, batch_total = get_accuracy(outputs_a, labels)
          val_correct += batch_correct
          val_total += batch_total
          loss = criterion(outputs_a, labels)
          
          running_val_loss += loss.item()
          total_val_loss += running_val_loss
          running_val_loss = 0.0
          
    with open("../output/output.txt","a") as f:
        f.write("Epoch: "+str(epoch)+" Train Loss: "+str(total_loss)+" Val Loss "+str(total_val_loss)+" Train Correct "+str(correct)+" Val Correct "+str(val_correct)+" Train-Accuracy: "+str(correct/total)+" Val-Accuracy: "+str(val_correct/val_total))
        f.write("\n")
    f.close()

    print("Epoch: ",epoch, " Train Loss ",total_loss," Val Loss ", total_val_loss, " Train Accuracy ",correct/total, " Val Accuracy ", val_correct/val_total)

    train_loss_plot.append(total_loss)
    val_loss_plot.append(total_val_loss)

    with open("../output/train_loss.txt","a") as f:
        f.write(str(total_loss))
        f.write("\n")
    f.close()
   
    with open("../output/val_loss.txt","a") as f:
        f.write(str(total_val_loss))
        f.write("\n")
    f.close()
    torch.save(model.state_dict(),"../output/model.pt")

print('Finished Training')

