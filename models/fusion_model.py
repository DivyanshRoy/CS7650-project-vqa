import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import numpy as np


class SimpleLSTM(nn.Module):

  def __init__(self, weights_matrix, hidden_size, num_layers, dropout_amount):
      super(SimpleLSTM, self).__init__()
      non_trainable = True
      num_embeddings, embedding_dim = weights_matrix.size()
      num_embeddings = 4400
      embedding_dim = 50
      emb_layer = nn.Embedding(num_embeddings, embedding_dim, sparse=False)
      # emb_layer.load_state_dict({'weight': weights_matrix})
      # if non_trainable:
          # emb_layer.weight.requires_grad = False
      # self.fc = nn.Linear(512, embedding_dim)
      self.embedding = emb_layer
      self.num_embeddings = num_embeddings
      self.embedding_dim = embedding_dim
      self.hidden_size = hidden_size
      self.num_layers = num_layers
      self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout_amount)
      
  def forward(self, inp):
      # e = self.embedding(inp)
      # images = self.fc(images)
      # images = images.view(images.size(0), 1, images.size(1))
      # v = e*images
      # return self.gru(v)
      # print(self.embedding(inp).size())
      return self.gru(self.embedding(inp))
  
  
  def init_hidden(self, batch_size):
      return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

class FusionModel(nn.Module):
  def __init__(self, weights_matrix, lstm_hidden_size, lstm_num_layers, model_input, answer_type, fusion_type, shared_size, dropout_amount, Top100):
    super(FusionModel, self).__init__()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    self.model_input = model_input
    self.answer_type = answer_type
    self.fusion_type = fusion_type
    self.lstm_model = SimpleLSTM(weights_matrix, lstm_hidden_size, lstm_num_layers, dropout_amount)
    lstm_num_layers = 1
    # self.lstm_model.train()
    if model_input == "resnet 18 image features, question":
      image_embedding_dim = 512
    elif model_input == "resnet 192 image features, question":
      image_embedding_dim = 2048
    elif model_input == "new resnet 18 features":
      image_embedding_dim = 512
    elif model_input == "new resnet 152 features":
      image_embedding_dim = 2048
    
    if self.fusion_type == "pointwise_mul" and image_embedding_dim==(lstm_hidden_size*lstm_num_layers):
      self.fc = nn.Linear(lstm_hidden_size*lstm_num_layers, image_embedding_dim)
      self.bn1 = nn.BatchNorm1d(image_embedding_dim)
      self.combined_embedding_dim = image_embedding_dim
    elif self.fusion_type == "try":
      self.img_dense1 = nn.Linear(image_embedding_dim, lstm_hidden_size*lstm_num_layers*1)
      self.bn1 = nn.BatchNorm1d(lstm_hidden_size*lstm_num_layers*1)
      self.img_dense2 = nn.Linear(lstm_hidden_size*lstm_num_layers*1, shared_size)
      self.bn2 = nn.BatchNorm1d(shared_size)

      self.q_dense1 = nn.Linear(lstm_hidden_size*lstm_num_layers*1, image_embedding_dim)
      self.bn3 = nn.BatchNorm1d(image_embedding_dim)
      self.q_dense2 = nn.Linear(image_embedding_dim, shared_size)
      self.bn4 = nn.BatchNorm1d(shared_size)
      self.combined_embedding_dim = shared_size
    else:
      # self.combined_embedding_dim = image_embedding_dim + (lstm_hidden_size*lstm_num_layers*25)
      self.combined_embedding_dim = image_embedding_dim + image_embedding_dim
      self.fc = nn.Linear(lstm_hidden_size*lstm_num_layers, image_embedding_dim)
      self.bn1 = nn.BatchNorm1d(image_embedding_dim)
      # self.combined_embedding_dim = (lstm_hidden_size*lstm_num_layers)*25
      # self.combined_embedding_dim = image_embedding_dim
    if model_input == "resnet 18 image features, question" and Top100 == True:
      print("100")
      self.fc1 = nn.Linear(self.combined_embedding_dim, 100)
    elif model_input == "new resnet 18 features":
      # interim_size = 96
      # interim_size = 256
      interim_size = 512
      # self.fc1 = nn.Linear(self.combined_embedding_dim, interim_size)
      # self.fc2 = nn.Linear(interim_size, 13)
      self.fc1 = nn.Linear(self.combined_embedding_dim, 104)
      
    elif model_input == "new resnet 152 features":
      # self.fc1 = nn.Linear(self.combined_embedding_dim, 13)
      interim_size = 512
      self.fc1 = nn.Linear(self.combined_embedding_dim, interim_size)
      self.bn1 = nn.BatchNorm1d(interim_size)
      self.fc2 = nn.Linear(interim_size, 64)
      self.bn2 = nn.BatchNorm1d(64)
      self.fc3 = nn.Linear(64, 13)
      
    elif self.answer_type == "yesno":
      self.fc1 = nn.Linear(self.combined_embedding_dim, 16)
      self.fc2 = nn.Linear(16, 2)
      # self.fc3 = nn.Linear(8, 1)
    elif self.answer_type == "number":      
      self.fc1 = nn.Linear(self.combined_embedding_dim, 100)
      # self.fc2 = nn.Linear(512, 100)
    elif self.answer_type == "other":      
      self.fc1 = nn.Linear(self.combined_embedding_dim, 1000)
    # if self.answer_type == "yesno":
    #   self.sigmoid = nn.Sigmoid()
    #   # pass
    # else:
    self.softmax = nn.Softmax(dim=1)
    self.dropout = nn.Dropout(p=dropout_amount)
    
  # def forward(self, inputs):
  def forward(self, inputs, questions, q_indices):
    # print(inputs.size())
    if self.model_input == "resnet 18 image features, question" or self.model_input == "resnet 192 image features, question" or self.model_input == "new resnet 18 features" or self.model_input == "new resnet 152 features":
      image_features = inputs

      questions = questions.type(torch.long)
      qout, question_embeddings = self.lstm_model(questions)
      # print(qout.size())
      # print(question_embeddings.size())
      # question_embeddings = qout[:,:,:].view(1, qout.size(0), -1)
      # question_embeddings = torch.reshape(qout, (1, qout.size(0),-1))
      # question_embeddings = question_embeddings.view(question_embeddings.size(1),-1)
      q_indices = q_indices.long()
      # q_indices = q_indices.view(-1)
      # qi = q_indices.detach().numpy()
      q_i = torch.zeros_like(qout).long()
      q_indices = q_indices.view(q_indices.size(0),1,1)
      # print(q_i.size())
      
      q_i += q_indices
      # print(q_indices[:,0,0])
      # print(q_indices.size())
      # print(q_i.size())
      # print(torch.max(q_indices))
      # print(torch.min(q_indices))
      # print(q_i[:5])
      # q_indices = q_indices.view(q_indices.size(0))
      question_embeddings = torch.gather(qout, 1, q_i)[:,0,:]
      # print(question_embeddings[:2])
      # question_embeddings = qout[:, qi, :]
      # question_embeddings = qout[:,q_indices,:]
      # question_embeddings = qout[:,4,:]
      question_embeddings = question_embeddings.view(question_embeddings.size(0), -1)

      # question_embeddings = question_embeddings.permute(1,2,0)
      # print(question_embeddings.size())
      # question_embeddings = question_embeddings.view(question_embeddings.size(0),-1)
      # print(question_embeddings.size())
      # question_embeddings = question_embeddings2.view(question_embeddings2.size(0),-1)
      # question_embeddings = torch.reshape(question_embeddings, (question_embeddings.size(0),-1))
      # question_embeddings = question_embeddings.view(1000,128)
      # image_features = self.dropout(image_features)
      # question_embeddings = self.dropout(question_embeddings)
      if self.fusion_type == "pointwise_mul" and image_features.size(1)==question_embeddings.size(1):
        question_embeddings = self.bn1(F.relu(self.fc(question_embeddings)))
        combined_embeddings = image_features*question_embeddings
      elif self.fusion_type == "try":
        img2q = self.dropout(self.bn1(self.img_dense1(image_features)))
        # print(img2q.size())
        # print(question_embeddings.size())
        img2qq = img2q*question_embeddings
        # img2qq = img2q
        sh1 = self.dropout(self.bn2(self.img_dense2(img2qq)))

        q2img = self.dropout(self.bn3(self.q_dense1(question_embeddings)))
        q2img2 = q2img*image_features
        # q2img2 = q2img
        sh2 = self.dropout(self.bn4(self.q_dense2(q2img2)))

        combined_embeddings = sh1*sh2
        # combined_embeddings = sh2
      else:
        # print(image_features.size())
        # print(question_embeddings.size())
        question_embeddings = self.bn1(F.relu(self.fc(question_embeddings)))
        combined_embeddings = torch.cat((image_features,question_embeddings), 1)
        # print(combined_embeddings.size())
        # combined_embeddings = question_embeddings
        # combined_embeddings = image_features
      combined_embeddings = self.dropout(combined_embeddings)
      
    
    # if self.image_features == False:
    #   image_features = self.image_features(images)
    # else:
    #   image_features = images
    # questions = questions.type(torch.long)
    # _, question_embeddings = self.lstm_model(questions)
    # question_embeddings = question_embeddings.view(question_embeddings.size(1),question_embeddings.size(2))
    # combined_embeddingsw = torch.cat((image_features,question_embeddings), 1)
    # x = F.relu(self.fc1(inputs))
    x = F.relu(self.fc1(combined_embeddings))
    # x = F.relu(self.fc2(x))
    if self.model_input == "new resnet 152 features":
      x = F.relu(self.fc3(x))
    if self.answer_type == "yesno":
      # x = x
      x = F.relu(self.fc2(x))
      # x = F.relu(self.fc3(x))
      # x = self.sigmoid(x)
      # pass
    # else:
      # x = F.relu(self.fc2(x))
    x = self.softmax(x)
    return x
  
  # def image_features(self, images):
  #   dt = images.dtype
  #   # scaler = transforms.Scale((224, 224))
  #   # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  #   # to_tensor = transforms.ToTensor()
  #   mean = torch.tensor(np.array([0.485, 0.456, 0.406]), dtype=dt).view(1,-1,1,1)#.cuda()
  #   std = torch.tensor(np.array([0.229, 0.224, 0.225]), dtype=dt).view(1,-1,1,1)#.cuda()
  #   transformed_images = (images-mean)/std
  #   image_embeddings = torch.zeros(images.size(0), 512)
  #   def copy_data(m, i, o):
  #       image_embeddings.copy_(o.data.view(o.data.size(0), o.data.size(1)))
  #   h = self.cnn_layer.register_forward_hook(copy_data)
  #   self.cnn_model(transformed_images)
  #   h.remove()
  #   return image_embeddings


