import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        ##Batch Normalisation & initialize weight in dense layer
        self.batch= nn.BatchNorm1d(embed_size,momentum = 0.01)
        self.embed.weight.data.normal_(0., 0.02)
        self.embed.bias.data.fill_(0)

    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.batch(self.embed(features))
        
        return features
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size = embed_size,hidden_size = hidden_size,
                            num_layers = num_layers, batch_first=True)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.35)
        self.init_weights()
        
    def init_weights(m):
        """Initialize weights."""
        if type(m) == nn.Linear:
            I.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
        if isinstance(m, nn.Conv2d):
            I.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.01)   
        

    def forward(self, features, captions):
        """ Decode feature vectors and generate captions """
        ###Remove end tag
        captions = captions[:, :-1]
        embed = self.dropout(self.embedding_layer(captions))
        embed = torch.cat((features.unsqueeze(1), embed), dim = 1)
        #packed = pack_padded_sequence(embed, lengths, batch_first=True) 
        lstm_outputs, _ = self.lstm(embed)
        out = self.linear(lstm_outputs)
        
        return out

    def sample(self, inputs, states=None, max_len=20): 
            " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) " 
            predicted_sentence = []
            
            for i in range(max_len):      # max sampling length
                
                # Get output and states from LSTM layer
                lstm_out, states = self.lstm(inputs, states)  #(batch_size, 1, hidden_size)
                lstm_out = lstm_out.squeeze(1)
                
                # Get output of the linear layer
                outputs = self.linear(lstm_out)
                
                # Get the best predicted 
                predicted = outputs.max(1)[1]
                                                    
                # Append predicted item to predicted sentence
                predicted_sentence.append(predicted.item())
                # Update input for next sequence
                inputs = self.embedding_layer(predicted).unsqueeze(1)
                
            #predicted_sentence = torch.cat(predicted_sentence, 0)
            return predicted_sentence