import torch
import torch.nn as nn


class BilinearFusion(nn.Module):
    """

    """
    def __init__(self, skip=1, use_bilinear=1, gate1=1, gate2=1, dim1=32, dim2=32, scale_dim1=1, scale_dim2=1, mmhid=64, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1+dim2+2 if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            o2 = self.linear_o2(vec2)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, o1, o2), 1)
        out = self.encoder2(out)
        return out


class TrilinearFusion_B(nn.Module):
    def __init__(self, skip=1, use_bilinear=1, gate1=1, gate2=1, gate3=1, dim1=32, dim2=32, dim3=32, scale_dim1=1,
                 scale_dim2=1, scale_dim3=1, mmhid=96, dropout_rate=0.25):
        super(TrilinearFusion_B, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        self.gate3 = gate3

        dim1_og, dim2_og, dim3_og, dim1, dim2, dim3 = dim1, dim2, dim3, dim1 // scale_dim1, dim2 // scale_dim2, dim3 // scale_dim3
        skip_dim = dim1 + dim2 + dim3 + 3 if skip else 0

        ### Path
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim3_og, dim1) if use_bilinear else nn.Sequential(
            nn.Linear(dim1_og + dim3_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        ### Graph
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim2_og, dim1_og, dim2) if use_bilinear else nn.Sequential(
            nn.Linear(dim2_og + dim1_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        ### Omic
        self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
        self.linear_z3 = nn.Bilinear(dim1_og, dim3_og, dim3) if use_bilinear else nn.Sequential(
            nn.Linear(dim1_og + dim3_og, dim3))
        self.linear_o3 = nn.Sequential(nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=0.25)
        self.encoder1 = nn.Sequential(nn.Linear((dim1 + 1) * (dim2 + 1) * (dim3 + 1), mmhid), nn.ReLU(),
                                      nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid + skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))


    def forward(self, vec1, vec2, vec3):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec3) if self.use_bilinear else self.linear_z1(
                torch.cat((vec1, vec3), dim=1))  # Gate Path with Omic
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec2, vec1) if self.use_bilinear else self.linear_z2(
                torch.cat((vec2, vec1), dim=1))  # Gate Graph with Omic
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            o2 = self.linear_o2(vec2)

        if self.gate3:
            h3 = self.linear_h3(vec3)
            z3 = self.linear_z3(vec1, vec3) if self.use_bilinear else self.linear_z3(
                torch.cat((vec1, vec3), dim=1))  # Gate Omic With Path
            o3 = self.linear_o3(nn.Sigmoid()(z3) * h3)
        else:
            o3 = self.linear_o3(vec3)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o3 = torch.cat((o3, torch.cuda.FloatTensor(o3.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        o123 = torch.bmm(o12.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o123)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, o1, o2, o3), 1)
        out = self.encoder2(out)
        return out



class QualinearFusion(nn.Module):
    def __init__(self, skip=1, use_bilinear=1,
                 gate1=1, gate2=1, gate3=1, gate4=1,
                 dim1=32, dim2=32, dim3=32, dim4=32,
                 scale_dim1=1,scale_dim2=1, scale_dim3=1, scale_dim4=1,
                 mmhid=96, dropout_rate=0.25):
        super(QualinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        self.gate3 = gate3
        self.gate4 = gate4

        dim1_og, dim2_og, dim3_og, dim4_og, dim1, dim2, dim3, dim4 = dim1, dim2, dim3, dim4, dim1 // scale_dim1, dim2 // scale_dim2, dim3 // scale_dim3, dim4 // scale_dim4
        # skip_dim = dim1 + dim2 + dim3 + 3 if skip else 0
        skip_dim = dim1 + dim2 + dim3 + dim4 + 4 if skip else 0

        ### tab
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim3_og, dim1) if use_bilinear else nn.Sequential(
            nn.Linear(dim1_og + dim3_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        ### scale1
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim2_og, dim1_og, dim2) if use_bilinear else nn.Sequential(
            nn.Linear(dim2_og + dim1_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        ### scale2
        self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
        self.linear_z3 = nn.Bilinear(dim1_og, dim3_og, dim3) if use_bilinear else nn.Sequential(
            nn.Linear(dim1_og + dim3_og, dim3))
        self.linear_o3 = nn.Sequential(nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=dropout_rate))

        ### scale3
        self.linear_h4 = nn.Sequential(nn.Linear(dim4_og, dim4), nn.ReLU())
        self.linear_z4 = nn.Bilinear(dim1_og, dim3_og, dim4) if use_bilinear else nn.Sequential(
            nn.Linear(dim1_og + dim3_og, dim4))
        self.linear_o4 = nn.Sequential(nn.Linear(dim4, dim4), nn.ReLU(), nn.Dropout(p=dropout_rate))


        self.post_fusion_dropout = nn.Dropout(p=0.25)
        self.encoder1 = nn.Sequential(nn.Linear((dim1 + 1) * (dim2 + 1) * (dim3 + 1) * (dim4 + 1), mmhid), nn.ReLU(),
                                      nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid + skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))


    def forward(self, vec1, vec2, vec3, vec4):
        ### Gated Multimodal Units
        print(vec1.shape, vec2.shape, vec3.shape, vec4.shape)
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec3) if self.use_bilinear else self.linear_z1(
                torch.cat((vec1, vec3), dim=1))  # Gate Path with Omic
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec2, vec1) if self.use_bilinear else self.linear_z2(
                torch.cat((vec2, vec1), dim=1))  # Gate Graph with Omic
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            o2 = self.linear_o2(vec2)

        if self.gate3:
            h3 = self.linear_h3(vec3)
            z3 = self.linear_z3(vec1, vec3) if self.use_bilinear else self.linear_z3(
                torch.cat((vec1, vec3), dim=1))  # Gate Omic With Path
            o3 = self.linear_o3(nn.Sigmoid()(z3) * h3)
        else:
            o3 = self.linear_o3(vec3)

        if self.gate4:
            h4 = self.linear_h4(vec4)
            z4 = self.linear_z4(vec1, vec4) if self.use_bilinear else self.linear_z4(
                torch.cat((vec1, vec4), dim=1))  # Gate Omic With Path
            o4 = self.linear_o4(nn.Sigmoid()(z4) * h4)
        else:
            o4 = self.linear_o4(vec4)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o3 = torch.cat((o3, torch.cuda.FloatTensor(o3.shape[0], 1).fill_(1)), 1)
        o4 = torch.cat((o4, torch.cuda.FloatTensor(o4.shape[0], 1).fill_(1)), 1)

        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        o123 = torch.bmm(o12.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
        o1234 = torch.bmm(o123.unsqueeze(2), o4.unsqueeze(1)).flatten(start_dim=1)

        out = self.post_fusion_dropout(o1234)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, o1, o2, o3, o4), 1)
        out = self.encoder2(out)
        return out