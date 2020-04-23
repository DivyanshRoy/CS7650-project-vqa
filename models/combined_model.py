import numpy as np
import torch 
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt 

answer_type = "number"


train_image = np.load("vgg_train_image_features_"+answer_type+".npy")
train_image = np.reshape(train_image, newshape=[train_image.shape[0], train_image.shape[1], train_image.shape[2]*train_image.shape[3]])
print(train_image.shape)

val_image = np.load("vgg_val_image_features_"+answer_type+".npy")
val_image = np.reshape(val_image, newshape=[val_image.shape[0], val_image.shape[1], val_image.shape[2]*val_image.shape[3]])
print(val_image.shape)

train_question = np.load("train_questions_"+answer_type+".npy")
val_question = np.load("val_questions_"+answer_type+".npy")

qword_dict = {}
cnt = 0
for i in range(train_question.shape[0]):
  for j in range(train_question.shape[1]):
    word = train_question[i,j]
    if word not in qword_dict:
      qword_dict[word] = cnt
      cnt += 1

for i in range(val_question.shape[0]):
  for j in range(val_question.shape[1]):
    word = val_question[i,j]
    if word not in qword_dict:
      qword_dict[word] = cnt
      cnt += 1

for i in range(train_question.shape[0]):
  for j in range(train_question.shape[1]):
    word = train_question[i,j]
    train_question[i,j] = qword_dict[word]

for i in range(val_question.shape[0]):
  for j in range(val_question.shape[1]):
    word = val_question[i,j]
    val_question[i,j] = qword_dict[word]

# print("cnt = ",cnt)

train_answer = np.load("train_answers_"+answer_type+".npy").astype(int)
# print(train_question.shape)
# print(train_answer.shape)
# print(train_answer.dtype)

val_answer = np.load("val_answers_"+answer_type+".npy").astype(int)
# print(val_question.shape)
# print(val_answer.shape)
# print(val_answer.dtype)

# print(train_question[:2])
# print(val_question[:2])

#all_images = np.concatenate((train_image,val_image), axis=0)
#all_questions = np.concatenate((train_question,val_question),axis=0)
#all_answers = np.concatenate((train_answer,val_answer), axis = 0)

#print(all_images.shape)
#print(all_questions.shape)
#print(all_answers.shape)

#indices = np.arange(6000)
#random.shuffle(indices)

#all_images = all_images[indices,:]
#all_questions = all_questions[indices,:]
#all_answers = all_answers[indices, :]

#train_image = all_images [:4000, :]
#val_image = all_images[4000:,:]

#train_question = all_questions[:4000,:]
#val_question = all_questions[4000:,:]

#train_answer = all_answers[:4000,:].astype(int)
#val_answer = all_answers[4000:,:].astype(int)

#print(train_image.shape)
#print(train_question.shape)
#print(train_answer.shape)

#print(val_image.shape)
#print(val_question.shape)
#print(val_answer.shape)

#a ,b = np.unique(train_answer, return_counts=True)
#print(a)
#print(b)

#a ,b = np.unique(val_answer, return_counts=True)
#print(a)
#print(b)

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
    # print("here 1")
    questions = self.embedding(questions)
    # print("here 2")
    x = self.dropout(self.W_b(questions))#self.W_b(questions)
    # print("here 3")
    C = self.tanh(torch.bmm(x,images)) 
    # print("here 4")

    H_v = self.tanh(self.dropout(self.W_v(torch.transpose(images, 1, 2)) + torch.bmm(torch.transpose(C,1,2),self.W_q(questions)))) #self.tanh(self.W_v(torch.transpose(images, 1, 2)) + torch.bmm(torch.transpose(C,1,2),self.W_q(questions)))
    # print("here 5")
    a_v = self.softmax(self.dropout(self.w_hv(H_v)))#self.softmax(self.w_hv(H_v))
    # print("here 6")
    a_v = torch.transpose(a_v, 1,2) 
    # print("here 7")
    v_hat = torch.sum(a_v * images, axis= 2) 
    # print("here 8")

    H_q = self.tanh(self.dropout(self.W_q(questions) + torch.bmm(C,self.W_v(torch.transpose(images, 1, 2))))) #self.tanh(self.W_q(questions) + torch.bmm(C,self.W_v(torch.transpose(images, 1, 2))))
    # print("here 9")
    a_q = self.softmax(self.dropout(self.w_hq(H_q)))#self.softmax(self.w_hq(H_q))
    # print("here 10")
    q_hat = torch.sum(a_q*questions, axis =1) 
    # print("here 11")
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
    # print("there 1")
    questions = self.embedding(questions)
    # print("there 2")
    H_1 = self.tanh(self.dropout(self.W_x_1(questions)))#self.tanh(self.W_x_1(questions))
    # print("there 3")
    a_x_1 = self.softmax(self.dropout(self.W_hx_1(H_1))) #self.softmax(self.W_hx_1(H_1))
    # print("there 4")
    x_hat_1 = torch.sum(a_x_1 * questions, axis=1) 

    # print("there 5")
    H_2 = self.tanh(self.dropout(self.W_x_2(torch.transpose(images,1,2)) + self.W_g_2(x_hat_1).unsqueeze(1)))#self.tanh(self.W_x_2(torch.transpose(images,1,2)) + self.W_g_2(x_hat_1).unsqueeze(1))
    # print("there 6")
    a_x_2 = self.softmax(self.dropout(self.W_hx_2(H_2)))#self.softmax(self.W_hx_2(H_2))
    # print("there 7")
    x_hat_2 = torch.sum(a_x_2*torch.transpose(images,1,2), axis=1)

    # print("there 8")
    H_3 = self.tanh(self.dropout(self.W_x_3(questions) + self.W_g_3(x_hat_2).unsqueeze(1)))#self.tanh(self.W_x_3(questions) + self.W_g_3(x_hat_2).unsqueeze(1))
    # print("there 9")
    a_x_3 = self.softmax(self.dropout(self.W_hx_3(H_3)))#self.softmax(self.W_hx_3(H_3))
    # print("there 10")
    x_hat_3 = torch.sum(a_x_3*questions, axis=1)
    # print("there 11")
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
    self.W_h = nn.Linear(self.d_prime,  100) #no of classes for yesno -2, number - 100, other - 1000 --verify again 

  def forward(self, q_hat, v_hat):
    h = self.tanh(self.dropout(self.W(q_hat + v_hat)))#self.tanh(self.W(q_hat + v_hat))
    return self.W_h(h)#self.softmax(self.W_h(h))

class Fusion(nn.Module):
  def __init__(self, shared_size):
    self.shared_size = shared_size
    super(Fusion, self).__init__()
    self.qp_qa = nn.Linear(512, 512)
    self.bn1 = nn.BatchNorm1d(512)
    self.qpa_qs = nn.Linear(512, self.shared_size)
    self.bn2 = nn.BatchNorm1d(self.shared_size)

    self.qa_qp = nn.Linear(512, 512)
    self.bn3 = nn.BatchNorm1d(512)
    self.qap_qs = nn.Linear(512, self.shared_size)
    self.bn4 = nn.BatchNorm1d(self.shared_size)

    self.vp_va = nn.Linear(512, 512)
    self.bn5 = nn.BatchNorm1d(512)
    self.vpa_vs = nn.Linear(512, self.shared_size)
    self.bn6 = nn.BatchNorm1d(self.shared_size)

    self.va_vp = nn.Linear(512, 512)
    self.bn7 = nn.BatchNorm1d(512)
    self.vap_vs = nn.Linear(512, self.shared_size)
    self.bn8 = nn.BatchNorm1d(self.shared_size)

  def forward(self, q_hat_p, v_hat_p, q_hat_a, v_hat_a):
    fq1 = self.bn2(self.qpa_qs((self.bn1(self.qp_qa(q_hat_p)+q_hat_a))))
    fq2 = self.bn4(self.qap_qs((self.bn3(self.qa_qp(q_hat_a)+q_hat_p))))
    fq = fq1*fq2

    fv1 = self.bn6(self.vpa_vs((self.bn5(self.vp_va(v_hat_p)+v_hat_a))))
    fv2 = self.bn8(self.vap_vs((self.bn7(self.va_vp(v_hat_a)+v_hat_p))))
    fv = fv1*fv2

    return fq, fv


class MainModel(nn.Module):
  def __init__(self, d, t, k, d_prime, vocab_size, dropout, shared_size):
    super(MainModel, self).__init__()
    self.d = d
    self.t = t
    self.k = k 
    self.d_prime = d_prime
    self.vocab_size = vocab_size
    self.dropout = dropout
    self.shared_size = shared_size

    self.parallel = ParallelCoAttention(self.d, self.t, self.k, self.vocab_size, self.dropout)
    self.alternate = AlternateCoAttention(self.d, self.k, self.vocab_size, self.dropout)
    self.fusion = Fusion(shared_size)
    self.answer = AnswerGeneration(self.shared_size, self.d_prime, self.dropout)
    


  def forward(self, questions, images):
    # print("parallel start")
    q_hat_p, v_hat_p = self.parallel(questions, images)
    # print("parallel end")
    # print("alt start")
    v_hat_a, q_hat_a = self.alternate(questions, images)
    # print(q_hat_p.size())
    # print(v_hat_p.size())
    # print(q_hat_a.size())
    # print(v_hat_a.size())
    fq, fv = self.fusion(q_hat_p, v_hat_p, q_hat_a, v_hat_a)
    # print("alt end")
    # answer_p = self.answer(q_hat_p, v_hat_p)
    # answer_a = self.answer(q_hat_a, v_hat_a)
    answer = self.answer(fq, fv)
    return answer
    # return (answer_p, answer_a)

d = 512
t = 25
k = 512
d_prime = 128 
vocab_size = cnt
dropout = 0.5
shared_size = 129

model = MainModel(d, t, k, d_prime, vocab_size, dropout, shared_size)
# model.load_state_dict(torch.load("output/model.pt"))


tensor_x = torch.Tensor(train_question).type(torch.long)
tensor_y = torch.Tensor(train_image).type(torch.float)
tensor_z = torch.Tensor(train_answer).type(torch.long).squeeze()

trainset = data.TensorDataset(tensor_x, tensor_y, tensor_z)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 1000, shuffle = True, num_workers=2)

tensor_x = torch.Tensor(val_question).type(torch.long)
tensor_y = torch.Tensor(val_image).type(torch.float)
tensor_z = torch.Tensor(val_answer).type(torch.long).squeeze()

valset = data.TensorDataset(tensor_x, tensor_y, tensor_z)
valloader = torch.utils.data.DataLoader(valset, batch_size = 1000, shuffle = True, num_workers = 2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=4e-5, weight_decay=1e-6, momentum=0.99)

def get_accuracy(predictions, labels):
  predictions = F.softmax(predictions,dim=1)
  predictions = torch.max(predictions, axis=1)[1]
  #predictions = predictions.detach().numpy()
  #correct = predictions.eq(labels).sum()
  ab = torch.abs(predictions-labels)
  ab = ab.detach().numpy()
  mn = np.minimum(ab, 1)
  eq = 1-mn
  correct = np.sum(eq)
  total = eq.shape[0]
  #total = predictions.shape[0]
  return correct, total

train_loss_plot = []
val_loss_plot = []

for epoch in range(100):  # loop over the dataset multiple times

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
        # outputs_p, outputs_a = model(questions, images)
        # batch_correct, batch_total = get_accuracy(outputs_p, labels)
        outputs = model(questions, images)
        batch_correct, batch_total = get_accuracy(outputs, labels)
        correct += batch_correct
        total += batch_total
        # loss = criterion(outputs_a, labels)
        loss = criterion(outputs, labels)
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
          # outputs_p, outputs_a = model(questions, images)
          # batch_correct, batch_total = get_accuracy(outputs_p, labels)
          outputs = model(questions, images)
          batch_correct, batch_total = get_accuracy(outputs, labels)
          val_correct += batch_correct
          val_total += batch_total
          # loss = criterion(outputs_a, labels)
          loss = criterion(outputs, labels)

          running_val_loss += loss.item()
          total_val_loss += running_val_loss
          running_val_loss = 0.0
          
    # with open("output/output.txt","a") as f:
    #     f.write("Epoch: "+str(epoch)+" Train Loss: "+str(total_loss)+" Val Loss "+str(total_val_loss)+" Train Correct "+str(correct)+" Val Correct "+str(val_correct)+" Train-Accuracy: "+str(correct/total)+" Val-Accuracy: "+str(val_correct/val_total))
    #     f.write("\n")
    # f.close()

    print("Epoch: ",epoch, " Train Loss ",total_loss," Val Loss ", total_val_loss, " Train Accuracy ",correct/total, " Val Accuracy ", val_correct/val_total)

    train_loss_plot.append(total_loss)
    val_loss_plot.append(total_val_loss)

    # with open("output/train_loss.txt","a") as f:
    #     f.write(str(total_loss))
    #     f.write("\n")
    # f.close()
   
    # with open("output/val_loss.txt","a") as f:
    #     f.write(str(total_val_loss))
    #     f.write("\n")
    # f.close()
    # torch.save(model.state_dict(),"output/model.pt")
    #plt.plot(np.arange(epoch+1),train_loss_plot)
    #plt.plot(np.arange(epoch+1),val_loss_plot)
    #plt.show()

print('Finished Training')