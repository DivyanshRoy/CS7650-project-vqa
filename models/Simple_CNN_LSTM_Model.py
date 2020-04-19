import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import numpy as np


class SimpleLSTM(nn.Module):

  def __init__(self, weights_matrix, hidden_size, num_layers):
      super(SimpleLSTM, self).__init__()
      non_trainable = True
      num_embeddings, embedding_dim = weights_matrix.size()
      emb_layer = nn.Embedding(num_embeddings, embedding_dim)
      emb_layer.load_state_dict({'weight': weights_matrix})
      if non_trainable:
          emb_layer.weight.requires_grad = False
      self.embedding = emb_layer
      self.num_embeddings = num_embeddings
      self.embedding_dim = embedding_dim
      self.hidden_size = hidden_size
      self.num_layers = num_layers
      self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
      
  def forward(self, inp):
      return self.gru(self.embedding(inp))
  
  
  def init_hidden(self, batch_size):
      return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

class SimpleCNNLSTM(nn.Module):
  def __init__(self, weights_matrix, lstm_hidden_size, lstm_num_layers, model_input, answer_type, fusion_type, Top100):
    super(SimpleCNNLSTM, self).__init__()
    self.model_input = model_input
    self.answer_type = answer_type
    self.fusion_type = fusion_type
    self.lstm_model = SimpleLSTM(weights_matrix, lstm_hidden_size, lstm_num_layers)
    # self.lstm_model.train()
    if model_input == "resnet 18 image features, question":
      image_embedding_dim = 512
    elif model_input == "resnet 192 image features, question":
      image_embedding_dim = 2048
    
    if self.fusion_type == "pointwise_mul" and image_embedding_dim==(lstm_hidden_size*lstm_num_layers):
      self.combined_embedding_dim = image_embedding_dim
    else:
      self.combined_embedding_dim = image_embedding_dim + (lstm_hidden_size*lstm_num_layers)
    
    if model_input == "resnet 18 image features, question" and Top100 == True:
      self.fc1 = nn.Linear(self.combined_embedding_dim, 100)
    elif self.answer_type == "yesno":
      self.fc1 = nn.Linear(self.combined_embedding_dim, 10)
      self.fc2 = nn.Linear(10, 2)
      # self.fc3 = nn.Linear(8, 1)
    elif self.answer_type == "number":      
      self.fc1 = nn.Linear(self.combined_embedding_dim, 512)
      self.fc2 = nn.Linear(512, 100)
    elif self.answer_type == "other":      
      self.fc1 = nn.Linear(self.combined_embedding_dim, 1000)
    if self.answer_type == "yesno":
      self.sigmoid = nn.Sigmoid()
      # pass
    else:
      self.softmax = nn.Softmax(dim=1)
    # self.dropout = nn.Dropout(p=0.1)
    
  # def forward(self, inputs):
  def forward(self, inputs, questions=None):
    if self.model_input == "resnet 18 image features, question" or self.model_input == "resnet 192 image features, question":
      image_features = inputs

      questions = questions.type(torch.long)
      _, question_embeddings = self.lstm_model(questions)
      question_embeddings = question_embeddings.view(question_embeddings.size(1),-1)
      # image_features = self.dropout(image_features)
      # question_embeddings = self.dropout(question_embeddings)
      # print("1d")
      if self.fusion_type == "pointwise_mul" and image_features.size(1)==question_embeddings.size(1):
        combined_embeddings = image_features*question_embeddings
      else:
        combined_embeddings = torch.cat((image_features,question_embeddings), 1)
        
      # print("2d")
    
    # if self.image_features == False:
    #   image_features = self.image_features(images)
    # else:
    #   image_features = images
    # questions = questions.type(torch.long)
    # _, question_embeddings = self.lstm_model(questions)
    # question_embeddings = question_embeddings.view(question_embeddings.size(1),question_embeddings.size(2))
    # combined_embeddings = torch.cat((image_features,question_embeddings), 1)
    # x = F.relu(self.fc1(inputs))
    x = F.relu(self.fc1(combined_embeddings))
    # print("3d")
    if self.answer_type == "yesno":
      # x = x
      x = F.relu(self.fc2(x))
      # x = F.relu(self.fc3(x))
      x = self.sigmoid(x)
      # pass
    elif self.answer_type == "other":
      x = self.softmax(x)
    else:
      x = F.relu(self.fc2(x))
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


