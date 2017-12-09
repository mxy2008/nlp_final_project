import torch
from torch import nn
from torch import autograd
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Function

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args

        n_d = self.n_d = args.hidden_dim # hidden dimension
        n_e = self.n_e = args.embedding_dim # embedding dimension
        depth = self.depth = args.depth # layer depth
        dropout = self.dropout = args.dropout
        self.Relu = nn.ReLU()

        if args.layer == 'lstm':
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

        if args.layer == 'cnn':
            print "current model is CNN"
            self.cnn = nn.Conv1d(
                in_channels = n_e,
                out_channels = n_d, #670 in the paper
                kernel_size = args.kernel_size,
                padding = (args.kernel_size-1)/2
            #dropout = dropout,
            )

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if args.cuda:
            return (autograd.Variable(torch.zeros(self.depth, batch_size, self.n_d)).cuda(),
                    autograd.Variable(torch.zeros(self.depth, batch_size, self.n_d)).cuda())
        else:
            return (autograd.Variable(torch.zeros(self.depth, batch_size, self.n_d)),
                autograd.Variable(torch.zeros(self.depth, batch_size, self.n_d)))


    def forward(self, xt, xb, mask_t, mask_b):
        # lstm
        if args.layer == 'lstm':
            output_t, self.hidden_1 = self.lstm(xt, self.hidden_1)
            output_b, self.hidden_1 = self.lstm(xb, self.hidden_2)

        if args.layer == 'cnn':
            xt = xt.permute(1,2,0)
            xb = xb.permute(1,2,0)
            output_t = self.cnn(self.Relu(xt))
            output_t = output_t.permute(2,0,1)
            output_b = self.cnn(self.Relu(xb))
            output_b = output_b.permute(2,0,1)

        if args.average:
            ht = average_without_padding(output_t, mask_t)
            hb = average_without_padding(output_b, mask_b)
        else:
            ht = output_t[-1]
            hb = output_b[-1]

        # get final vector encoding 
        h_final = (ht+hb)*0.5
        h_final = F.normalize(h_final, p=2, dim=1)

        return h_final


class GradReverse(Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -args.lam)

def grad_reverse(x):
    return GradReverse()(x)

class domain_classifier(nn.Module):
    def __init__(self):
        super(domain_classifier, self).__init__()
        self.fc1 = nn.Linear(args['hidden_dim'], 100) 
        self.fc2 = nn.Linear(100, 1)
        self.drop = nn.Dropout2d(0.25)

    def forward(self, x):
        x = grad_reverse(x)
        x = F.leaky_relu(self.drop(self.fc1(x)))
        x = self.fc2(x)
        return F.sigmoid(x)
