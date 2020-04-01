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
  def __init__(self, weights_matrix, lstm_hidden_size, lstm_num_layers, image_features=False):
    super(SimpleCNNLSTM, self).__init__()
    self.image_features = image_features
    self.lstm_model = SimpleLSTM(weights_matrix, lstm_hidden_size, lstm_num_layers)
    if self.image_features == False:
      self.cnn_model = models.resnet18(pretrained=True)
      self.cnn_layer = self.cnn_model._modules.get('avgpool')
      self.cnn_model.eval()

    self.fc1 = nn.Linear(2560, 1000)
    self.softmax = nn.Softmax(dim=1)
    
  def forward(self, images, questions):
    if self.image_features == False:
      image_features = self.image_features(images)
    else:
      image_features = images
    questions = questions.type(torch.long)
    _, question_embeddings = self.lstm_model(questions)
    question_embeddings = question_embeddings.view(question_embeddings.size(1),question_embeddings.size(2))
    combined_embeddings = torch.cat((image_features,question_embeddings), 1)
    x = F.relu(self.fc1(combined_embeddings))
    x = self.softmax(x)
    return x
  
  def image_features(self, images):
    dt = images.dtype
    # scaler = transforms.Scale((224, 224))
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # to_tensor = transforms.ToTensor()
    mean = torch.tensor(np.array([0.485, 0.456, 0.406]), dtype=dt).view(1,-1,1,1)#.cuda()
    std = torch.tensor(np.array([0.229, 0.224, 0.225]), dtype=dt).view(1,-1,1,1)#.cuda()
    transformed_images = (images-mean)/std
    image_embeddings = torch.zeros(images.size(0), 512)
    def copy_data(m, i, o):
        image_embeddings.copy_(o.data.view(o.data.size(0), o.data.size(1)))
    h = self.cnn_layer.register_forward_hook(copy_data)
    self.cnn_model(transformed_images)
    h.remove()
    return image_embeddings


