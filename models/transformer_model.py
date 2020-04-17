import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        # print(output.size())
        # output = self.decoder(output)
        # print(output.size())
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



# class SimpleLSTM(nn.Module):

#   def __init__(self, weights_matrix, hidden_size, num_layers):
#       super(SimpleLSTM, self).__init__()
#       non_trainable = True
#       num_embeddings, embedding_dim = weights_matrix.size()
#       emb_layer = nn.Embedding(num_embeddings, embedding_dim, sparse=True)
#       emb_layer.load_state_dict({'weight': weights_matrix})
#       if non_trainable:
#           emb_layer.weight.requires_grad = False
#       self.embedding = emb_layer
#       self.num_embeddings = num_embeddings
#       self.embedding_dim = embedding_dim
#       self.hidden_size = hidden_size
#       self.num_layers = num_layers
#       self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
      
#   def forward(self, inp):
#       return self.gru(self.embedding(inp))
  
  
#   def init_hidden(self, batch_size):
#       return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

class FusionModel(nn.Module):
  def __init__(self, weights_matrix, lstm_hidden_size, lstm_num_layers, model_input, answer_type, fusion_type, shared_size, dropout_amount, Top100):
    super(FusionModel, self).__init__()
    self.model_input = model_input
    self.answer_type = answer_type
    self.fusion_type = fusion_type
    # self.lstm_model = SimpleLSTM(weights_matrix, lstm_hidden_size, lstm_num_layers)
    if model_input == "resnet 18 image features, question":
      ntokens = 2700
    elif model_input == "resnet 192 image features, question":
      ntokens = 1750
    elif model_input == "new resnet 18 features":
      ntokens = 3100
    elif model_input == "new resnet 152 features":
      ntokens = 3100
    
    emsize = 512
    nhead = 2
    nhid = 512
    nlayers = 2
    self.transformer_model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout_amount)
    # self.lstm_model.train()
    self.lstm_hidden_size = 25*emsize
    self.lstm_num_layers = 1
    # print(self.lstm_hidden_size)
    if model_input == "resnet 18 image features, question":
      image_embedding_dim = 512
    elif model_input == "resnet 192 image features, question": #typo: it should be resnet 152 here
      image_embedding_dim = 2048
    elif model_input == "new resnet 18 features":
      image_embedding_dim = 512
    elif model_input == "new resnet 152 features":
      image_embedding_dim = 2048
    # print(self.fusion_type)
    if self.fusion_type == "pointwise_mul" and image_embedding_dim==(lstm_hidden_size*lstm_num_layers):
      self.combined_embedding_dim = image_embedding_dim
    elif self.fusion_type == "try":
      self.img_dense1 = nn.Linear(image_embedding_dim, self.lstm_hidden_size*self.lstm_num_layers)
      self.img_dense2 = nn.Linear(self.lstm_hidden_size*self.lstm_num_layers, shared_size)

      self.q_dense1 = nn.Linear(self.lstm_hidden_size*self.lstm_num_layers, image_embedding_dim)
      self.q_dense2 = nn.Linear(image_embedding_dim, shared_size)
      self.combined_embedding_dim = shared_size
    else:
      self.combined_embedding_dim = image_embedding_dim + self.lstm_hidden_size*self.lstm_num_layers
      # print(self.combined_embedding_dim)
    if model_input == "resnet 18 image features, question" and Top100 == True:
      print("100")
      self.fc1 = nn.Linear(self.combined_embedding_dim, 100)
    elif model_input == "new resnet 18 features":
      self.fc1 = nn.Linear(self.combined_embedding_dim, 13)
    elif model_input == "new resnet 152 features":
      self.fc1 = nn.Linear(self.combined_embedding_dim, 13)
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
  def forward(self, inputs, questions=None):
    # print(inputs.size())
    if self.model_input == "resnet 18 image features, question" or self.model_input == "resnet 192 image features, question" or self.model_input == "new resnet 18 features" or self.model_input == "new resnet 152 features":
      image_features = inputs

      questions = questions.type(torch.long)
      # qout, question_embeddings = self.lstm_model(questions)
      # print(qout.size())
      # print(question_embeddings.size())
      # question_embeddings = qout[:,-1,:].view(1, qout.size(0), qout.size(2))
      question_embeddings = self.transformer_model(questions)
      
      # question_embeddings = question_embeddings.view(question_embeddings.size(1),-1)
      question_embeddings = question_embeddings.view(question_embeddings.size(0),-1)
      # question_embeddings = question_embeddings[:,-1]
      # print("q: ",question_embeddings.size())
      # print("i: ",image_features.size())
      # image_features = self.dropout(image_features)
      # question_embeddings = self.dropout(question_embeddings)
      if self.fusion_type == "pointwise_mul" and image_features.size(1)==question_embeddings.size(1):
        combined_embeddings = image_features*question_embeddings
      elif self.fusion_type == "try":
        img2q = self.dropout(self.img_dense1(image_features))
        # print(img2q.size())
        # print(question_embeddings.size())
        img2qq = img2q*question_embeddings
        sh1 = self.dropout(self.img_dense2(img2qq))

        q2img = self.dropout(self.q_dense1(question_embeddings))
        q2img2 = q2img*image_features
        sh2 = self.dropout(self.q_dense2(q2img2))

        combined_embeddings = sh1*sh2
      else:
        combined_embeddings = torch.cat((image_features,question_embeddings), 1)
      combined_embeddings = self.dropout(combined_embeddings)
    # print("c: ",combined_embeddings.size())  
    
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


