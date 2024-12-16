import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from sheaf_models import AttentionSheafLearner
from main.model.cont_models import GeneralSheafDiffusion, GraphLaplacianDiffusion, BundleSheafDiffusion, DiagSheafDiffusion
from main.model.sheaf_base import SheafDiffusion
from main.model.parser import get_parser
import torch.sparse as sparse


parser = get_parser()
args = parser.parse_args()

class LearnableSparseCOO(nn.Module):
    def __init__(self, indices, size, init_value=0.0):
        super(LearnableSparseCOO, self).__init__()
        
        self.indices = indices
        self.size = size
        
        # Initialize learnable values for the non-zero entries
        num_nonzero = indices.size(1)
        #self.values = nn.Parameter(torch.full((num_nonzero,), init_value))  # Default initialization
        
        # Alternatively, to initialize randomly:
        self.values = nn.Parameter(torch.rand(num_nonzero))  # Random initialization
        

    def forward(self):
        # Apply sigmoid to the learnable values to constrain them between 0 and 1
        constrained_values = torch.sigmoid(self.values)
        #constrained_values = (constrained_values > 0.5).float()
        # Create and return the sparse COO tensor
        return torch.sparse_coo_tensor(self.indices, constrained_values, self.size)


class GumbelSoftmaxClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(GumbelSoftmaxClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)
        
    def forward(self, x, temperature=1.0, hard=False):
        # Compute logits
        logits = self.linear(x)
        
        # Apply Gumbel-Softmax
        gumbel_softmax_out = F.gumbel_softmax(logits, tau=temperature, hard=hard)
        
        class_indices = torch.argmax(gumbel_softmax_out, dim=1)
        
        return class_indices


class DeepSN(nn.Module):
    def __init__(self, ninput, nfeat, nhid, nclass, dropout, edge_index):
        super(DeepSN, self).__init__()
        self.dropout = dropout

        self.sheaf = SheafGraphLayer(nfeat, nclass, dropout=dropout, edge_index = edge_index.coalesce().indices(), concat=False)
        
       
        
        self.transform = nn.Linear(ninput, nfeat)
        
        self.phi_1 = nn.Parameter(torch.randn(edge_index.shape[0], nfeat))
        self.phi_2 = nn.Parameter(torch.randn(edge_index.shape[0], nfeat))
        
        self.kappa_1 = nn.Parameter(torch.randn(edge_index.shape[0], nfeat))
        self.kappa_2 = nn.Parameter(torch.randn(edge_index.shape[0], nfeat))
        

        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        
        nn.init.xavier_uniform_(self.phi_1)
        nn.init.xavier_uniform_(self.phi_2)
        nn.init.xavier_uniform_(self.kappa_1)
        nn.init.xavier_uniform_(self.kappa_2)
        
    def sparse_mm(self, a, b):
        indices = a.coalesce().indices()
        values = a.coalesce().values()
        return torch.sparse_coo_tensor(indices, values, a.size())
        

    def forward(self, x, adj, y_i, n):
        x = F.dropout(self.transform(x), self.dropout, training=self.training)
        y_a = y_i
        for i in range(n):
            
            x = self.sheaf(x, adj, y_a) 
            #x[torch.isinf(x) | torch.isnan(x)] = 0
            x = x + torch.sigmoid(torch.sigmoid(self.beta)*(self.phi_1*x) / (self.kappa_1+x + 1e-8))
            x_d = sparse.mm(self.sparse_mm(adj, y_a), x) - sparse.mm(self.sparse_mm(adj, 1-y_a), x)
            x = x + torch.sigmoid(torch.sigmoid(self.gamma)*(self.phi_2*x_d) / (self.kappa_2+torch.abs(x_d) + 1e-8) )
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.elu(x)
            y = torch.sigmoid(x).mean(dim=1, keepdim=True)
            y_a = y.squeeze()#self.classifier(x, temperature=0.1, hard=False)
        return y


    
    
class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SheafGraphLayer(nn.Module):


    def __init__(self, in_features, out_features, dropout, edge_index, concat=True):
        super(SheafGraphLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.sheaf = GeneralSheafDiffusion(edge_index, args)
        #self.sheaf = GraphLaplacianDiffusion(edge_index, args)
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj, y_a):
        dv = 'cuda' if input.is_cuda else 'cpu'
        
        #self.sheaf.update_edge_index(adj.indices())

        N = input.size()[0]
        if adj.layout == torch.sparse_coo:
            edge = adj.coalesce().indices()
        else:
            edge = adj.nonzero().t()
            
        h = self.sheaf(input, adj, y_a) 

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h)
        else:
            # if this layer is last layer,
            return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
