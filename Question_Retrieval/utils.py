import numpy as np 
import torch
from torch.autograd import Variable

def normalize_3d(x, eps=1e-8):
    #TODO
    # x is len*batch*d
    # l2 is len*batch*1
    #l2 = x.norm(2,axis=2).dimshuffle((0,1,'x'))
    l2 = torch.norm(x, 2).dimshuffle((0,1,'x'))#??
    return x/(l2+eps)

def average_without_padding(x, ids, padding_id, eps=1e-8):
    # len*batch*1
    # neq: Returns a variable representing the result of logical inequality (a!=b)
    # dimshuffle: (0, 'x', 1) -> AxB to Ax1xB, (1, 'x', 0) -> AxB to Bx1xA
    # (0,1,'x') -> AxB to AxBx1?
    mask = np.not_equal(ids, padding_id).astype('float')
    
    mask = Variable(torch.from_numpy(mask)).float().view(ids.shape[0],ids.shape[1],1)
    result = torch.sum(x*mask,dim=0)/ (torch.sum(mask,dim=0)+eps)
    #print 'result shape', result.data.shape
    return result 