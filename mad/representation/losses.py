import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.spatial.distance import cdist
import numpy as np

# helper
def Mat(Lvec):
    N = Lvec.size(0)
    Mask = Lvec.repeat(N,1)
    Same = (Mask==Mask.t())
    return Same.clone().fill_diagonal_(0), ~Same#same diff
def fun_CosSim(Mat_A, Mat_B, norm=1, ):#N by F
    N_A = Mat_A.size(0)
    N_B = Mat_B.size(0)

    D = Mat_A.mm(torch.t(Mat_B))
    D.fill_diagonal_(-norm)
    return D

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0,1))

    return res

def fun_EuclideanSim(a, b, norm = 1):#N by F
    matrix = torch.cdist(a, b, p = 2)
#     max_val = torch.max( matrix.flatten())
#     result = max_val - matrix  # diagnol will be all the same
#     result.fill_diagonal_(-norm) # set diagnal to -1

    return matrix

def fun_mahaSim(a, b, norm = 1):#N by F

    a = a.clone().detach().numpy()
    b = b.clone().detach().numpy()

    vi = np.linalg.pinv(np.cov(a.T))
    matrix = cdist(a, b, 'mahalanobis', VI=vi)

    matrix = torch.tensor(matrix)
    matrix.requires_grad = True

    return matrix

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, distance):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = self.distance(anchor, positive)  # .pow(.5)
        distance_negative = self.distance(anchor, negative)  # .pow(.5)
#         print('Pos:', distance_positive, '\tNeg:', distance_negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class HardHardNegativesBatchLoss(nn.Module):
    """
    Constastive loss
    create 2 matrices, find the worst pair of positive, and the worst pair of pos-neg. Margin-based loss
    """

    def __init__(self, margin, distance):
        super(HardHardNegativesBatchLoss, self).__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, positive, negative, size_average=True):
        # make sure positive/negative shape is: batch x dimension

        positive = F.normalize(positive, p = 2, dim = 1)
        negative = F.normalize(negative, p = 2, dim = 1)

        sim_pos = torch.cdist(positive, positive, p = 2).flatten() #
#         non_zero = torch.nonzero( sim_pos )
        loss1 = torch.max( sim_pos )

        sim_pos_neg = torch.cdist(positive, negative, p = 2).flatten() #
        non_zero = torch.nonzero( sim_pos_neg )
        loss2 = torch.min( sim_pos_neg[non_zero] ) # why max also works?


        losses = F.relu( loss1 - loss2 + self.margin)
        return losses


class HardHardNegativesBatchLossAll(nn.Module):
    """
    Constastive loss
    create 2 matrices, find the worst pair of positive, and the worst pair of pos-neg. Margin-based loss
    """

    def __init__(self, margin, distance):
        super(HardHardNegativesBatchLossAll, self).__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, positive, negative, size_average=True):
        # make sure positive/negative shape is: batch x dimension


        fvec = torch.cat([positive, negative],dim = 0) # stack positive and negative into a feature vector column
        lvec = torch.tensor(np.array( [1]*positive.shape[0] + [0]*negative.shape[0] )) # labels

        # number of samples
        N = lvec.size(0)

        # feature normalization
        fvec_norm = F.normalize(fvec, p = 2, dim = 1)

        # Same/Diff label Matting in dissimilarity Matrix
        Same, Diff = Mat(lvec.view(-1))
        dissim = torch.cdist(fvec_norm, fvec_norm, p = 2)

        # finding the max per row
        D_detach_P = dissim.clone().detach()

        D_detach_P[Diff] = -1 # find on same label

        V_pos, I_pos = D_detach_P.max(1) # find max per row, v_pos is value, I_pos index

        # # extracting pos score
        Pos = dissim[torch.arange(0,N), I_pos] # grab the max of each row
        Pos = Pos.clone().detach().cpu()

        # finding the min per row
        D_detach_P = dissim.clone().detach()

        D_detach_P[Same] = 10000 # set same label = 1000, don't want these
        zero_mask = D_detach_P == 0
        D_detach_P[zero_mask ] = 1000 # want the min but not 0

        V_neg, I_neg = D_detach_P.min(1) # find min per row, v_ is value, I_ index

        # # extracting score
        Neg = dissim[torch.arange(0,N), I_neg] # grab the max of each row
        Neg = Neg.clone().detach().cpu()

#         HardTripletMask = Neg < Pos # <-- Useless

        # triplets
        Triplet_val = torch.stack([Pos,Neg],1) # concat column-wise pos-pos, pos-neg


        losses = -F.log_softmax(Triplet_val[:]/0.1, dim=1)[:,0].sum()

        losses.requires_grad = True

        return losses

class EasyEasyNegativesBatchLoss(nn.Module):
    """
    Constastive loss
    create 2 matrices, find the worst pair of positive, and the worst pair of pos-neg. Margin-based loss
    """

    def __init__(self, margin, distance):
        super(EasyEasyNegativesBatchLoss, self).__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, positive, negative, size_average=True):
        # make sure positive/negative shape is: batch x dimension

        positive = F.normalize(positive, p = 2, dim = 1)
        negative = F.normalize(negative, p = 2, dim = 1)

        sim_pos = torch.cdist(positive, positive, p = 2).flatten() #
        non_zero = torch.nonzero( sim_pos )
        if sum(non_zero) == 0:
            loss1 = torch.tensor([0.0])
            loss1.requires_grad = True
        else:
            loss1 = torch.min( sim_pos[non_zero] )

        sim_pos_neg = torch.cdist(positive, negative, p = 2).flatten() #
        non_zero = torch.nonzero( sim_pos_neg )
        loss2 = torch.max( sim_pos_neg[non_zero] ) # change this to min too

        losses = F.relu( loss1 - loss2 + self.margin)
        return losses

class HardHardNegativesBatchLoss(nn.Module):
    """
    Constastive loss
    create 2 matrices, find the worst pair of positive, and the worst pair of pos-neg. Margin-based loss
    """

    def __init__(self, margin, distance):
        super(HardHardNegativesBatchLoss, self).__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, positive, negative, size_average=True):
        # make sure positive/negative shape is: batch x dimension

        positive = F.normalize(positive, p = 2, dim = 1)
        negative = F.normalize(negative, p = 2, dim = 1)

        sim_pos = torch.cdist(positive, positive, p = 2).flatten() #
        non_zero = torch.nonzero( sim_pos )
        if sum(non_zero) == 0:
            loss1 = torch.tensor([0.0])
            loss1.requires_grad = True
        else:
            loss1 = torch.max( sim_pos[non_zero] )

        sim_pos_neg = torch.cdist(positive, negative, p = 2).flatten() #
        non_zero = torch.nonzero( sim_pos_neg )
        loss2 = torch.min( sim_pos_neg[non_zero] ) # change this to min too

        losses = F.relu( loss1 - loss2 + self.margin)
        return losses

class EasyEasyNegativesBatchLossAll(nn.Module):
    """
    Constastive loss
    create 2 matrices, find the worst pair of positive, and the worst pair of pos-neg. Margin-based loss
    """

    def __init__(self, margin, distance):
        super(EasyEasyNegativesBatchLossAll, self).__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, positive, negative, size_average=True):
        # make sure positive/negative shape is: batch x dimension

        fvec = torch.cat([positive, negative],dim = 0) # stack positive and negative into a feature vector column
        lvec = torch.tensor(np.array( [1]*positive.shape[0] + [0]*negative.shape[0] )) # labels

        # number of samples
        N = lvec.size(0)

        # feature normalization
        fvec_norm = F.normalize(fvec, p = 2, dim = 1)

        # Same/Diff label Matting in dissimilarity Matrix
        Same, Diff = Mat(lvec.view(-1))
        dissim = torch.cdist(fvec_norm, fvec_norm, p = 2)

        # finding the max per row
        D_detach_P = dissim.clone().detach()

        D_detach_P[Diff] = 10000 # find on same label, set diff to 1000
        zero_mask = D_detach_P == 0
        D_detach_P[zero_mask ] = 10000

        V_pos, I_pos = D_detach_P.min(1) # find max per row, v_pos is value, I_pos index

        # # extracting pos score
        Pos = dissim[torch.arange(0,N), I_pos] # grab the max of each row
        Pos = Pos.clone().detach().cpu()

        # finding the max per row
        D_detach_P = dissim.clone().detach()

        D_detach_P[Same] = -10000 # set same label = -1000, don't want these


        V_neg, I_neg = D_detach_P.max(1) # find max per row, v_ is value, I_ index

        # # extracting score
        Neg = dissim[torch.arange(0,N), I_neg] # grab the max of each row
        Neg = Neg.clone().detach().cpu()

#         EasyHardTripletMask = Neg > Pos

        # triplets
        Triplet_val = torch.stack([Pos,Neg],1) # concat column-wise pos-pos, pos-neg

        losses = -F.log_softmax(Triplet_val[:]/0.1, dim=1)[:,0].sum()

        losses.requires_grad = True

        return losses


class OnlineTripletLoss(nn.Module): # does not work yet
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)

class SCTLossEuclidean(nn.Module):
    def __init__(self, method, lam=1):
        super(SCTLossEuclidean, self).__init__()

        if method=='sct':
            self.sct = True
            self.semi = False
        elif method=='hn':
            self.sct = False
            self.semi = False
        elif method=='shn':
            self.sct = False
            self.semi = True
        else:
            print('loss type is not supported')

        self.lam = lam

    def forward(self, positive, negative):
        loss = 0
        fvec = torch.cat([positive, negative],dim = 0) # stack positive and negative into a feature vector column
        lvec = torch.tensor(np.array( [1]*positive.shape[0] + [0]*negative.shape[0] )) # labels

        # number of samples
        N = lvec.size(0)

        # feature normalization
        fvec_norm = F.normalize(fvec, p = 2, dim = 1)

        # Same/Diff label Matting in Similarity Matrix
        Same, Diff = Mat(lvec.view(-1))

        # Similarity Matrix
#         CosSim = fun_CosSim(fvec_norm, fvec_norm)
        Sim  = fun_EuclideanSim(fvec_norm, fvec_norm)

        # finding max similarity on same label pairs
        # for each sample, compare to others and retrieve the max sim
        D_detach_P = Sim.clone().detach()

        D_detach_P[Diff] = -1

#         D_detach_P[D_detach_P>0.9999] = -1 # check invalid since cosine sim can't larger than 0.99
        D_detach_P[D_detach_P>100000] = -1
        V_pos, I_pos = D_detach_P.min(1) # find max per row, v_pos is value, I_pos index

        # # valid positive pairs(prevent pairs with duplicated images)
#         Mask_pos_valid = (V_pos>-1)&(V_pos<1) # because cosine sim must in range [-1,1]
        Mask_pos_valid = (V_pos>-100000)&(V_pos<100000)

        # # extracting pos score
        Pos = Sim[torch.arange(0,N), I_pos] # grab the max of each row
        Pos_log = Pos.clone().detach().cpu()

        # finding max similarity on diff label pairs
        D_detach_N = Sim.clone().detach()
        D_detach_N[Same] = -1

        # Masking out non-Semi-Hard Negative
        if self.semi:
            D_detach_N[(D_detach_N>(V_pos.repeat(N,1).t()))&Diff] = -1

        V_neg, I_neg = D_detach_N.min(1)

        # valid negative pairs
#         Mask_neg_valid = (V_neg>-1)&(V_neg<1)
        Mask_neg_valid = (V_neg>-100000)&(V_neg<100000)


        # extracting neg score
        Neg = Sim[torch.arange(0,N), I_neg]
        Neg_log = Neg.clone().detach().cpu()

        # Mask all valid triplets
        Mask_valid = Mask_pos_valid&Mask_neg_valid

        # Mask hard/easy triplets

        HardTripletMask = ((Neg>Pos) ) & Mask_valid # 0.8 is an abitrary threshold for cosine
        EasyTripletMask = ((Neg<Pos) ) & Mask_valid

        # triplets
        Triplet_val = torch.stack([Pos,Neg],1) # concat column-wise pos-pos, pos-neg
        Triplet_idx = torch.stack([I_pos,I_neg],1)

        Triplet_val_log = Triplet_val.clone().detach().cpu()
        Triplet_idx_log = Triplet_idx.clone().detach().cpu()

        if self.sct: # SCT setting
            loss_hardtriplet = Neg[HardTripletMask].sum()
            loss_easytriplet = -F.log_softmax(Triplet_val[EasyTripletMask,:]/0.1, dim=1)[:,0].sum()

            N_hard = HardTripletMask.float().sum()
            N_easy = EasyTripletMask.float().sum()

            if torch.isnan(loss_hardtriplet) or N_hard==0:
                loss_hardtriplet, N_hard = 0, 0
#                 print('No hard triplets in the batch')

            if torch.isnan(loss_easytriplet) or N_easy==0:
                loss_easytriplet, N_easy = 0, 0
#                 print('No easy triplets in the batch')

            N = N_easy + N_hard
            if N==0: N=1
            loss = torch.tensor( (loss_easytriplet + self.lam*loss_hardtriplet)/N )
            loss.requires_grad = True
        else: # Standard Triplet Loss setting

            loss = -F.log_softmax(Triplet_val[Mask_valid,:]/0.1, dim=1)[:,0].mean()
            if  torch.isnan(loss):
                loss = torch.tensor( [0.0] )
                loss.requires_grad = True

        return loss

class SCTLossMaha(nn.Module):
    def __init__(self, method = 'sct', lam=1):
        super(SCTLossMaha, self).__init__()

        if method=='sct':
            self.sct = True
            self.semi = False
        elif method=='hn':
            self.sct = False
            self.semi = False
        elif method=='shn':
            self.sct = False
            self.semi = True
        else:
            print('loss type is not supported')

        self.lam = lam

    def forward(self, positive, negative):
        loss = 0
        fvec = torch.cat([positive, negative],dim = 0) # stack positive and negative into a feature vector column
        lvec = torch.tensor(np.array( [1]*positive.shape[0] + [0]*negative.shape[0] )) # labels

        # number of samples
        N = lvec.size(0)

        # feature normalization
        fvec_norm = F.normalize(fvec, p = 2, dim = 1)

        # Same/Diff label Matting in Similarity Matrix
        Same, Diff = Mat(lvec.view(-1))

        # Similarity Matrix
#         CosSim = fun_CosSim(fvec_norm, fvec_norm)
        Sim  = fun_mahaSim(fvec_norm, fvec_norm)

        # finding max similarity on same label pairs
        # for each sample, compare to others and retrieve the max sim
        D_detach_P = Sim.clone().detach()

        D_detach_P[Diff] = -1

#         D_detach_P[D_detach_P>0.9999] = -1 # check invalid since cosine sim can't larger than 0.99
        D_detach_P[D_detach_P>100000] = -1
        V_pos, I_pos = D_detach_P.min(1) # find max per row, v_pos is value, I_pos index

        # # valid positive pairs(prevent pairs with duplicated images)
#         Mask_pos_valid = (V_pos>-1)&(V_pos<1) # because cosine sim must in range [-1,1]
        Mask_pos_valid = (V_pos>-100000)&(V_pos<100000)

        # # extracting pos score
        Pos = Sim[torch.arange(0,N), I_pos] # grab the max of each row
        Pos_log = Pos.clone().detach().cpu()

        # finding max similarity on diff label pairs
        D_detach_N = Sim.clone().detach()
        D_detach_N[Same] = -1

        # Masking out non-Semi-Hard Negative
        if self.semi:
            D_detach_N[(D_detach_N<(V_pos.repeat(N,1).t()))&Diff] = -1

        V_neg, I_neg = D_detach_N.min(1)

        # valid negative pairs
#         Mask_neg_valid = (V_neg>-1)&(V_neg<1)
        Mask_neg_valid = (V_neg>-100000)&(V_neg<100000)


        # extracting neg score
        Neg = Sim[torch.arange(0,N), I_neg]
        Neg_log = Neg.clone().detach().cpu()

        # Mask all valid triplets
        Mask_valid = Mask_pos_valid&Mask_neg_valid

        # Mask hard/easy triplets

        HardTripletMask = ((Neg>Pos) ) & Mask_valid # 0.8 is an abitrary threshold for cosine
        EasyTripletMask = ((Neg<Pos) ) & Mask_valid

        # triplets
        Triplet_val = torch.stack([Pos,Neg],1) # concat column-wise pos-pos, pos-neg
        Triplet_idx = torch.stack([I_pos,I_neg],1)

        Triplet_val_log = Triplet_val.clone().detach().cpu()
        Triplet_idx_log = Triplet_idx.clone().detach().cpu()

        if self.sct: # SCT setting
            loss_hardtriplet = Neg[HardTripletMask].sum()
            loss_easytriplet = -F.log_softmax(Triplet_val[EasyTripletMask,:]/0.1, dim=1)[:,0].sum()

            N_hard = HardTripletMask.float().sum()
            N_easy = EasyTripletMask.float().sum()

            if torch.isnan(loss_hardtriplet) or N_hard==0:
                loss_hardtriplet, N_hard = 0, 0
#                 print('No hard triplets in the batch')

            if torch.isnan(loss_easytriplet) or N_easy==0:
                loss_easytriplet, N_easy = 0, 0
#                 print('No easy triplets in the batch')

            N = N_easy + N_hard
            if N==0: N=1
            loss = torch.tensor( (loss_easytriplet + self.lam*loss_hardtriplet)/N )
            loss.requires_grad = True
        else: # Standard Triplet Loss setting

            loss = -F.log_softmax(Triplet_val[Mask_valid,:]/0.1, dim=1)[:,0].mean()
            if  torch.isnan(loss):
                loss = torch.tensor( [0.0] )
                loss.requires_grad = True

        return loss


class SCTLossCosine(nn.Module):
    def __init__(self, method, lam=1):
        super(SCTLossCosine, self).__init__()

        if method=='sct':
            self.sct = True
            self.semi = False
        elif method=='hn':
            self.sct = False
            self.semi = False
        elif method=='shn':
            self.sct = False
            self.semi = True
        else:
            print('loss type is not supported')

        self.lam = lam

    def forward(self, positive, negative):
        loss = 0
        fvec = torch.cat([positive, negative],dim = 0) # stack positive and negative into a feature vector column
        lvec = torch.tensor(np.array( [1]*positive.shape[0] + [0]*negative.shape[0] )) # labels

        # number of samples
        N = lvec.size(0)

        # feature normalization
        fvec_norm = F.normalize(fvec, p = 2, dim = 1)

        # Same/Diff label Matting in Similarity Matrix
        Same, Diff = Mat(lvec.view(-1))

        # Similarity Matrix
        CosSim = fun_CosSim(fvec_norm, fvec_norm)
#         CosSim = sim_matrix(fvec_norm, fvec_norm)


        # finding max similarity on same label pairs
        # for each sample, compare to others and retrieve the max sim
        D_detach_P = CosSim.clone().detach()

        D_detach_P[Diff] = -1

        D_detach_P[D_detach_P>0.9999] = -1 # check invalid since cosine sim can't larger than 0.99

        V_pos, I_pos = D_detach_P.max(1) # find max per row, v_pos is value, I_pos index

        # # valid positive pairs(prevent pairs with duplicated images)
        Mask_pos_valid = (V_pos>-1)&(V_pos<1) # because cosine sim must in range [-1,1]

        # # extracting pos score
        Pos = CosSim[torch.arange(0,N), I_pos] # grab the max of each row
        Pos_log = Pos.clone().detach().cpu()

        # finding max similarity on diff label pairs
        D_detach_N = CosSim.clone().detach()
        D_detach_N[Same] = -1

        # Masking out non-Semi-Hard Negative
        if self.semi:
            D_detach_N[(D_detach_N>(V_pos.repeat(N,1).t()))&Diff] = -1

        V_neg, I_neg = D_detach_N.max(1)

        # valid negative pairs
        Mask_neg_valid = (V_neg>-1)&(V_neg<1)


        # extracting neg score
        Neg = CosSim[torch.arange(0,N), I_neg]
        Neg_log = Neg.clone().detach().cpu()

        # Mask all valid triplets
        Mask_valid = Mask_pos_valid&Mask_neg_valid

        # Mask hard/easy triplets
        HardTripletMask = ((Neg>Pos) | (Neg>0.8)) & Mask_valid # 0.8 is an abitrary threshold for cosine
        EasyTripletMask = ((Neg<Pos) & (Neg<0.8)) & Mask_valid


        # triplets
        Triplet_val = torch.stack([Pos,Neg],1) # concat column-wise pos-pos, pos-neg
        Triplet_idx = torch.stack([I_pos,I_neg],1)

        Triplet_val_log = Triplet_val.clone().detach().cpu()
        Triplet_idx_log = Triplet_idx.clone().detach().cpu()

        if self.sct: # SCT setting
            loss_hardtriplet = Neg[HardTripletMask].sum()
            loss_easytriplet = -F.log_softmax(Triplet_val[EasyTripletMask,:]/0.1, dim=1)[:,0].sum()

            N_hard = HardTripletMask.float().sum()
            N_easy = EasyTripletMask.float().sum()

            if torch.isnan(loss_hardtriplet) or N_hard==0:
                loss_hardtriplet, N_hard = 0, 0
#                 print('No hard triplets in the batch')

            if torch.isnan(loss_easytriplet) or N_easy==0:
                loss_easytriplet, N_easy = 0, 0
#                 print('No easy triplets in the batch')

            N = N_easy + N_hard
            if N==0: N=1
            loss = torch.tensor( (loss_easytriplet + self.lam*loss_hardtriplet)/N )
            loss.requires_grad = True
        else: # Standard Triplet Loss setting
            loss = -F.log_softmax(Triplet_val[Mask_valid][:]/0.1, dim=1)[:,0].mean()

            if  torch.isnan(loss):
                loss = torch.tensor( [0.0] )
                loss.requires_grad = True

        return loss
class EasyPositiveHardNegativeLossCosine(nn.Module):
    def __init__(self, sig=1):
        super(EasyPositiveHardNegativeLossCosine, self).__init__()

        self.semi = False

        self.sigma = sig

    def forward(self, positive, negative):
        loss = 0
        fvec = torch.cat([positive, negative],dim = 0) # stack positive and negative into a feature vector column
        lvec = torch.tensor(np.array( [1]*positive.shape[0] + [0]*negative.shape[0] )) # labels

        # number of samples
        N = lvec.size(0)

        # feature normalization
        fvec_norm = F.normalize(fvec, p = 2, dim = 1)

        # Same/Diff label Matting in Similarity Matrix
        Same, Diff = Mat(lvec.view(-1))

        # Similarity Matrix
        CosSim = sim_matrix(fvec_norm, fvec_norm)

        # finding max similarity on same label pairs
        # for each sample, compare to others and retrieve the max sim
        D_detach_P = CosSim.clone().detach()

        D_detach_P[Diff] = -1

        D_detach_P[D_detach_P>0.9999] = -1 # check invalid since cosine sim can't larger than 0.99

        V_pos, I_pos = D_detach_P.max(1) # find max per row, v_pos is value, I_pos index

        # prevent duplicated pairs
        Mask_not_drop_pos = (V_pos>0)

        # # extracting pos score
        Pos = CosSim[torch.arange(0,N), I_pos] # grab the max of each row
        Pos_log = Pos.clone().detach().cpu()


        # finding max similarity on diff label pairs
        D_detach_N = CosSim.clone().detach()
        D_detach_N[Same] = -1

        V_neg, I_neg = D_detach_N.max(1)

        # prevent invalid pairs
        Mask_not_drop_neg = (V_neg>-1)

        # extracting neg score
        Neg = CosSim[torch.arange(0,N), I_neg]
        Neg_log = Neg.clone().detach().cpu()

         # triplets
        T = torch.stack([Pos,Neg],1)
        Mask_not_drop = Mask_not_drop_pos&Mask_not_drop_neg

        # loss
        Prob = -F.log_softmax(T/self.sigma,dim=1)[:,0]
        loss = Prob[Mask_not_drop].mean()

        if torch.isnan(loss):
            loss = torch.tensor( [0.0] )
            loss.requires_grad = True

        return loss

class EasyPositiveHardNegativeLossMaha(nn.Module):
    def __init__(self, sig=1):
        super(EasyPositiveHardNegativeLossMaha, self).__init__()

        self.semi = False

        self.sigma = sig

    def forward(self, positive, negative):
        loss = 0
        fvec = torch.cat([positive, negative],dim = 0) # stack positive and negative into a feature vector column
        lvec = torch.tensor(np.array( [1]*positive.shape[0] + [0]*negative.shape[0] )) # labels

        # number of samples
        N = lvec.size(0)

        # feature normalization
        fvec_norm = F.normalize(fvec, p = 2, dim = 1)

        # Same/Diff label Matting in Similarity Matrix
        Same, Diff = Mat(lvec.view(-1))

        # Similarity Matrix
        CosSim = fun_mahaSim(fvec_norm, fvec_norm)

        # finding max similarity on same label pairs
        # for each sample, compare to others and retrieve the max sim
        D_detach_P = CosSim.clone().detach()

        D_detach_P[Diff] = -1

        # D_detach_P[D_detach_P>0.9999] = -1 # check invalid since cosine sim can't larger than 0.99

        V_pos, I_pos = D_detach_P.min(1) # find min per row, v_pos is value, I_pos index

        # prevent duplicated pairs
        Mask_not_drop_pos = (V_pos>0)

        # # extracting pos score
        Pos = CosSim[torch.arange(0,N), I_pos] # grab the max of each row
        Pos_log = Pos.clone().detach().cpu()


        # finding min on diff label pairs
        D_detach_N = CosSim.clone().detach()
        D_detach_N[Same] = -1

        V_neg, I_neg = D_detach_N.min(1)

        # prevent invalid pairs
        Mask_not_drop_neg = (V_neg>0)

        # extracting neg score
        Neg = CosSim[torch.arange(0,N), I_neg]
        Neg_log = Neg.clone().detach().cpu()

         # triplets
        T = torch.stack([Pos,Neg],1)
        Mask_not_drop = Mask_not_drop_pos&Mask_not_drop_neg

        # loss
        Prob = -F.log_softmax(T/self.sigma,dim=1)[:,0]
        loss = Prob[Mask_not_drop].mean()

        if torch.isnan(loss):
            loss = torch.tensor( [0.0] )
            loss.requires_grad = True

        return loss

class EasyPositiveSemiHardNegativeLossMaha(nn.Module):
    def __init__(self, sig=1):
        super(EasyPositiveSemiHardNegativeLossMaha, self).__init__()

        self.semi = True

        self.sigma = sig

    def forward(self, positive, negative):
        loss = 0
        fvec = torch.cat([positive, negative],dim = 0) # stack positive and negative into a feature vector column
        lvec = torch.tensor(np.array( [1]*positive.shape[0] + [0]*negative.shape[0] )) # labels

        # number of samples
        N = lvec.size(0)

        # feature normalization
        fvec_norm = F.normalize(fvec, p = 2, dim = 1)

        # Same/Diff label Matting in Similarity Matrix
        Same, Diff = Mat(lvec.view(-1))

        # Similarity Matrix
        CosSim = fun_mahaSim(fvec_norm, fvec_norm)

        # finding max similarity on same label pairs
        # for each sample, compare to others and retrieve the max sim
        D_detach_P = CosSim.clone().detach()

        D_detach_P[Diff] = -1

        # D_detach_P[D_detach_P>0.9999] = -1 # check invalid since cosine sim can't larger than 0.99

        V_pos, I_pos = D_detach_P.min(1) # find min per row, v_pos is value, I_pos index

        # prevent duplicated pairs
        Mask_not_drop_pos = (V_pos>0)

        # # extracting pos score
        Pos = CosSim[torch.arange(0,N), I_pos] # grab the max of each row
        Pos_log = Pos.clone().detach().cpu()


        # finding min on diff label pairs
        D_detach_N = CosSim.clone().detach()
        D_detach_N[Same] = -1

        if self.semi:
            D_detach_N[(D_detach_N<(V_pos.repeat(N,1).t()))&Diff]=-1#extracting SHN

        V_neg, I_neg = D_detach_N.min(1)

        # prevent invalid pairs
        Mask_not_drop_neg = (V_neg>0)

        # extracting neg score
        Neg = CosSim[torch.arange(0,N), I_neg]
        Neg_log = Neg.clone().detach().cpu()

         # triplets
        T = torch.stack([Pos,Neg],1)
        Mask_not_drop = Mask_not_drop_pos&Mask_not_drop_neg

        # loss
        Prob = -F.log_softmax(T/self.sigma,dim=1)[:,0]
        loss = Prob[Mask_not_drop].mean()

        if torch.isnan(loss):
            loss = torch.tensor( [0.0] )
            loss.requires_grad = True

        return loss

class EasyPositiveSemiHardNegativeLossCosine(nn.Module):
    def __init__(self, sig=1):
        super(EasyPositiveSemiHardNegativeLossCosine, self).__init__()

        self.semi = True

        self.sigma = sig

    def forward(self, positive, negative):
        loss = 0
        fvec = torch.cat([positive, negative],dim = 0) # stack positive and negative into a feature vector column
        lvec = torch.tensor(np.array( [1]*positive.shape[0] + [0]*negative.shape[0] )) # labels

        # number of samples
        N = lvec.size(0)

        # feature normalization
        fvec_norm = F.normalize(fvec, p = 2, dim = 1)

        # Same/Diff label Matting in Similarity Matrix
        Same, Diff = Mat(lvec.view(-1))

        # Similarity Matrix
        CosSim = fun_CosSim(fvec_norm, fvec_norm)

        # finding max similarity on same label pairs
        # for each sample, compare to others and retrieve the max sim
        D_detach_P = CosSim.clone().detach()

        D_detach_P[Diff] = -1

        D_detach_P[D_detach_P>0.9999] = -1 # check invalid since cosine sim can't larger than 0.99

        V_pos, I_pos = D_detach_P.max(1) # find max per row, v_pos is value, I_pos index

        # prevent duplicated pairs
        Mask_not_drop_pos = (V_pos>0)

        # # extracting pos score
        Pos = CosSim[torch.arange(0,N), I_pos] # grab the max of each row
        Pos_log = Pos.clone().detach().cpu()


        # finding max similarity on diff label pairs
        D_detach_N = CosSim.clone().detach()
        D_detach_N[Same] = -1
        if self.semi:
            D_detach_N[(D_detach_N>(V_pos.repeat(N,1).t()))&Diff]=-1#extracting SHN

        V_neg, I_neg = D_detach_N.max(1)

        # prevent invalid pairs
        Mask_not_drop_neg = (V_neg>-1)

        # extracting neg score
        Neg = CosSim[torch.arange(0,N), I_neg]
        Neg_log = Neg.clone().detach().cpu()

         # triplets
        T = torch.stack([Pos,Neg],1)
        Mask_not_drop = Mask_not_drop_pos&Mask_not_drop_neg

        # loss
        Prob = -F.log_softmax(T/self.sigma,dim=1)[:,0]
        loss = Prob[Mask_not_drop].mean()

        if torch.isnan(loss):
            loss = torch.tensor( [0.0] )
            loss.requires_grad = True

        return loss
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, positive, negative, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """


        fvec = torch.cat([positive, negative],dim = 0)
        labels = torch.tensor(np.array( [1]*positive.shape[0] + [0]*negative.shape[0] ))

        features = fvec

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # if len(features.shape) < 3:
        #     raise ValueError('`features` needs to be [bsz, n_views, ...],'
        #                      'at least 3 dimensions are required')
        # if len(features.shape) > 3:
        #     features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=0, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, x1,x2 ):

        embeddings = torch.cat([x1, x2],dim = 0) # stack positive and negative into a feature vector column
        target = torch.tensor(np.array( [1]*x1.shape[0] + [0]*x2.shape[0] )) # labels

#         print('Emb size:', embeddings.shape[0])
#         print('target size:', target.shape[0])

        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, x1,x2,x3 ):

        embeddings = torch.cat([x1, x2, x3],dim = 0) # stack positive and negative into a feature vector column
        target = torch.tensor(np.array( [1]*x1.shape[0]*2 + [0]*x3.shape[0] )) # labels

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)

import torch.distributed as dist
import diffdist.functional as distops
# https://github.com/alinlab/CSI/blob/master/training/sup/sup_simclr_CSI.py

def get_similarity_matrix(outputs, chunk=2, multi_gpu=False):
    '''
        Compute similarity matrix
        - outputs: (B', d) tensor for B' = B * chunk
        - sim_matrix: (B', B') tensor
    '''

    if multi_gpu:
        outputs_gathered = []
        for out in outputs.chunk(chunk):
            gather_t = [torch.empty_like(out) for _ in range(dist.get_world_size())]
            gather_t = torch.cat(distops.all_gather(gather_t, out))
            outputs_gathered.append(gather_t)
        outputs = torch.cat(outputs_gathered)

    sim_matrix = torch.mm(outputs, outputs.t())  # (B', d), (d, B') -> (B', B')

    return sim_matrix


def NT_xent(positive, negative, temperature=0.5, chunk=2, eps=1e-8):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''
    fvec = torch.cat([positive, negative],dim = 0)
    sim_matrix = get_similarity_matrix( fvec )

    device = sim_matrix.device

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    loss = torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)

    return loss

class Supervised_NT_xent(nn.Module):
    def __init__(self, temperature):
        super(Supervised_NT_xent, self).__init__()
        self.temperature = temperature # 0.5 works for diffu and super
        self.chunk = 2
        self.eps=1e-8
        self.multi_gpu = False

    def forward(self, positive, negative):
        '''
            Compute NT_xent loss
            - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
        '''
        fvec = torch.cat([positive, negative],dim = 0)
        labels = torch.tensor(np.array( [1]*positive.shape[0] + [0]*negative.shape[0] ))
        sim_matrix = get_similarity_matrix( fvec )
        device = sim_matrix.device

        if self.multi_gpu:
            gather_t = [torch.empty_like(labels) for _ in range(dist.get_world_size())]
            labels = torch.cat(distops.all_gather(gather_t, labels))
        # labels = labels.repeat(2)

        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()

        B = sim_matrix.size(0) // self.chunk  # B = B' / chunk

        eye = torch.eye(B * self.chunk).to(device)  # (B', B')
        sim_matrix = torch.exp(sim_matrix / self.temperature) * (1 - eye)  # remove diagonal

        denom = torch.sum(sim_matrix, dim=1, keepdim=True)
        sim_matrix = -torch.log(sim_matrix / (denom + self.eps) + self.eps)  # loss matrix

        labels = labels.contiguous().view(-1, 1)
        Mask = torch.eq(labels, labels.t()).float().to(device)
        #Mask = eye * torch.stack([labels == labels[i] for i in range(labels.size(0))]).float().to(device)
        Mask = Mask / (Mask.sum(dim=1, keepdim=True) + self.eps)

        loss = torch.sum(Mask * sim_matrix) / (2 * B)

        return loss
