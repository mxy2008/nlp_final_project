import sys
import torch
from torch import nn
from torch import autograd
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Function

sys.path.append('..')
from misc import *

class Model(nn.Module):
    def __init__(self, args, weights=None):
        super(Model, self).__init__()

        self.args = args
        #self.padding_id = padding_id
        n_d = self.n_d = self.args.hidden_dim # hidden dimension
        n_e = self.n_e = self.args.embed_dim # embedding dimension
        depth = self.depth = self.args.depth # layer depth
        dropout = self.dropout = self.args.dropout
        self.Relu = nn.ReLU()

        if self.args.layer == 'lstm':
            print "current model is LSTM"
            self.lstm = nn.LSTM(
                input_size = n_e,
                hidden_size = n_d,
                num_layers = depth,
                # bidirectional = True,
                # dropout = dropout,
            )

            self.hidden_1 = None
            self.hidden_2 = None

        if self.args.layer == 'cnn':
            print "current model is CNN"
            self.cnn = nn.Conv1d(
                in_channels = n_e,
                out_channels = n_d, #670 in the paper
                kernel_size = self.args.kernel_size,
                padding = (self.args.kernel_size-1)/2
            #dropout = dropout,
            )

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.args.cuda:
            return (autograd.Variable(torch.zeros(self.depth, batch_size, self.n_d)).cuda(),
                    autograd.Variable(torch.zeros(self.depth, batch_size, self.n_d)).cuda())
        else:
            return (autograd.Variable(torch.zeros(self.depth, batch_size, self.n_d)),
                    autograd.Variable(torch.zeros(self.depth, batch_size, self.n_d)))


    def forward(self, xt, xb, mask_t, mask_b):
        
        # lstm
        if self.args.layer == 'lstm':
            #print self.hidden_1[0].data.shape, self.hidden_2[0].data.shape
            output_t, _ = self.lstm(xt, self.hidden_1)
            output_b, _ = self.lstm(xb, self.hidden_2)

        if self.args.layer == 'cnn':
            xt = xt.permute(1,2,0)
            xb = xb.permute(1,2,0)
            output_t = self.cnn(self.Relu(xt))
            output_t = output_t.permute(2,0,1)
            output_b = self.cnn(self.Relu(xb))
            output_b = output_b.permute(2,0,1)

        if self.args.average:
            ht = average_without_padding(output_t, mask_t)
            hb = average_without_padding(output_b, mask_b)
        else:
            ht = output_t[-1]
            hb = output_b[-1]

        # get final vector encoding 
        h_final = (ht+hb)*0.5
        h_final = f.normalize(h_final, p=2, dim=1)
        
        return h_final

class GradReverse(Function):
    def __init__(self, args):
        super(GradReverse, self).__init__()
        self.args = args

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.args.lam)

def grad_reverse(args, x):
    return GradReverse(args)(x)

class domain_classifier(nn.Module):
    def __init__(self, args):
        super(domain_classifier, self).__init__()

        self.args = args 
        self.fc1 = nn.Linear(args.hidden_dim, 100) 
        self.fc2 = nn.Linear(100, 1)
        self.drop = nn.Dropout2d(0.25)

    def forward(self, x):
        x = grad_reverse(self.args, x)
        x = F.leaky_relu(self.drop(self.fc1(x)))
        x = self.fc2(x)
        return F.sigmoid(x)
