import torch
import torch.nn as nn
from torch.nn.functional import normalize

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        #nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class encoder(nn.Module):
    def __init__(self, input_dim, hid_dimen1):
        super(encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, hid_dimen1)
        )

        self.encoder.apply(xavier_init)

    def forward(self, x):
        return self.encoder(x)



class decoder(nn.Module):
    def __init__(self, hid_dimen1, input_dim):
        super(decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hid_dimen1, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

        self.decoder.apply(xavier_init)

    def forward(self, x):
        return self.decoder(x)


class MLP(nn.Module):
    def __init__(self, input_dim, num_class):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, num_class)
        )
        self.net.apply(xavier_init)

    def forward(self, x):
        return self.net(x)

class FeatureExtract(nn.Module):
    def __init__(self, n_input, n_z, nLabel):
        super(FeatureExtract, self).__init__()
        self.num_views = len(n_input)
        self.hid_dimen1 = 512
        self.hid_dimen2 = 128
        self.encoder_list1 = nn.ModuleList()
        self.decoder_list1 = nn.ModuleList()
        for i in range(self.num_views):
            self.encoder_list1.append(encoder(n_input[i],self.hid_dimen1))

        for i in range(self.num_views):
            self.decoder_list1.append(decoder(self.hid_dimen1,n_input[i]))

        self.label_dimension1 = 32
        self.label_dimension2 = 16
        self.share_classifier_label = nn.ModuleList()
        for i in range(self.num_views):
            self.share_classifier_label.append(MLP(self.hid_dimen1, nLabel))
        self.fusion_classifier_label=MLP(self.hid_dimen1, nLabel)
        self.fusion_classifier_label1=MLP(self.hid_dimen1, nLabel)

        self.cf_learn = nn.Sequential(
            nn.Linear(self.hid_dimen1, self.hid_dimen2),
        )

    def forward(self, mul_X, dist, mode):
        batch_size = mul_X[0].shape[0]
        summ = 0
        share_zs1 = []
        share_zs2 = []
        cfs = []
        share_zs_fusion = 0
        share_zs_classification = []             

        for enc_i, enc in enumerate(self.encoder_list1):
            z_i = enc(mul_X[enc_i])

            tmp = z_i
            cf = normalize(self.cf_learn(z_i))
            cfs.append(cf)
            share_zs2.append(tmp)
            logit = self.share_classifier_label[enc_i](tmp)
            share_zs_classification.append(logit)  # per-view classification logits

        if mode == 'train':
            x_bar_list1 = []
            for dec_i, dec in enumerate(self.decoder_list1):
                tmp = dec(share_zs2[dec_i])
                x_bar_list1.append(tmp)

            eps = 1e-8
            weights = []
            
            for v in range(self.num_views):
                logits = share_zs_classification[v]
                p = torch.softmax(logits, dim=1)
                weighted_entropy = -(p * (p + eps).log()).sum(dim=1)
                denom = weighted_entropy + eps
                weights.append(1.0 / denom)                   # [batch]

            # aggregate and normalise the weights
            adjusted_weight_sum = torch.stack(weights, dim=0).sum(dim=0)  # [V, batch] -> [batch]
            final_weights = [adjusted_weight / adjusted_weight_sum for adjusted_weight in weights]  # final normalised weights

            # fuse the view embeddings
            z_fused = torch.zeros_like(share_zs2[0])  # [batch, n_z]
            for v in range(self.num_views):
                z_fused = z_fused + share_zs2[v] * final_weights[v].unsqueeze(1)
            

            logit_fusion = self.fusion_classifier_label(z_fused)
            share_zs_fusion = logit_fusion

            return share_zs2, share_zs_classification, share_zs_fusion, x_bar_list1, cfs

        elif mode == 'val':

            eps = 1e-8
            weights = []
            
            for v in range(self.num_views):
                logits = share_zs_classification[v]
                p = torch.softmax(logits, dim=1)
                weighted_entropy = -(p * (p + eps).log()).sum(dim=1)
                denom = weighted_entropy + eps
                weights.append(1.0 / denom)                   # [batch]

            # aggregate and normalise the weights
            adjusted_weight_sum = torch.stack(weights, dim=0).sum(dim=0)  # [V, batch] -> [batch]
            final_weights = [adjusted_weight / adjusted_weight_sum for adjusted_weight in weights]  # final normalised weights

            # fuse the view embeddings
            z_fused = torch.zeros_like(share_zs2[0])  # [batch, n_z]
            for v in range(self.num_views):
                z_fused = z_fused + share_zs2[v] * final_weights[v].unsqueeze(1)
            

            logit_fusion = self.fusion_classifier_label(z_fused)
            share_zs_fusion = logit_fusion

            return share_zs_classification, share_zs_fusion
