import torch
import snn_bound_layers as sbl
from snn_fire import time_step, get_time_step_certify

torch.set_printoptions(threshold=10000, linewidth=500, precision=2)


def _get_bound(A, x_U, x_L, sign, t):
    A = A.view(A.size(0), A.size(1), -1)
    x_U = x_U.view(x_U.size(0), -1, 1)
    x_L = x_L.view(x_L.size(0), -1, 1)

    mid, diff = (x_U + x_L) / 2.0, (x_U - x_L) / 2.0

    if diff.min() < 0:
        print('error forward backward at time step ', t)

    bound = A.bmm(mid) + sign * A.abs().bmm(diff)
    bound = bound.squeeze(-1)

    return bound




class MNIST_CONV(torch.nn.Module):
    def __init__(self):
        super(MNIST_CONV, self).__init__()

        self.conv1  = sbl.BoundConv2d(1, 64, stride=2)
        self.mem_c1 = sbl.BoundMemUpdate()

        self.conv2  = sbl.BoundConv2d(64, 128, stride=2)
        self.mem_c2 = sbl.BoundMemUpdate()

        self.flat   = sbl.BoundFlatten()

        self.fc1    = sbl.BoundLinear(7 * 7 * 128, 256)
        self.mem_f1 = sbl.BoundMemUpdate()

        self.fc2    = sbl.BoundLinear(256, 10)
        self.mem_f2 = sbl.BoundMemUpdate()

        self.classify = sbl.BoundLast()

        self.layer_lst = [self.mem_c1, self.mem_c2, self.flat, self.mem_f1, self.mem_f2, self.classify]
        self.opts_lst = [self.conv1, self.conv2, None, self.fc1, self.fc2, None]
        self.get_para()


    def get_para(self):
        self.time_step_certify = get_time_step_certify()

        for layer in self.layer_lst:
            layer.get_para()


    def forward(self, x_):
        x = torch.stack([x_ for _ in range(time_step)], dim=0)
        x = x > torch.rand(x.size()).cuda()
        x = x.float()

        m, s = self.mem_c1(self.conv1, x)
        m, s = self.mem_c2(self.conv2, s)

        s = self.flat(torch.stack(s, dim=0))

        m, s = self.mem_f1(self.fc1, s)
        m, s = self.mem_f2(self.fc2, s)

        output = self.classify(s)

        return output


    def snn_bound(self, x_, eps, labels): 

        s_U = torch.stack([x_ + eps for _ in range(self.time_step_certify)], dim=0)
        s_L = torch.stack([x_ - eps for _ in range(self.time_step_certify)], dim=0)
        tmp = torch.rand(s_U.size()).cuda()
        s_U, s_L = (s_U > tmp), (s_L > tmp)

        s_U, s_L = s_U.float(), s_L.float()
        self.s_U = s_U.clone()
        self.s_L = s_L.clone()

        self.diff = (s_U - s_L).sum()

        for l in range(len(self.layer_lst)):

            if isinstance(self.layer_lst[l], sbl.BoundFlatten):
                s_U, s_L = self.layer_lst[l].bound(s_U, s_L)

            elif isinstance(self.layer_lst[l], sbl.BoundLast):
                output_U, output_L = self.layer_lst[l].bound(s_U, s_L, labels)  

            else:
                m_U, s_U, m_L, s_L = self.layer_lst[l].snn_bound(self.opts_lst[l], s_U, s_L)
            
        return output_U, output_L


    def snn_crown_ibp(self, x_, eps, labels):
        # apply forward IBP
        ibp_U, ibp_L = self.snn_bound(x_, eps, labels)

        xA_L, bias_L = 0, 0, 

        for l in range(len(self.layer_lst) - 1, -1, -1):

            if isinstance(self.layer_lst[l], sbl.BoundFlatten):
                xA_L, tmp_bias_L = self.layer_lst[l].bound_back(xA_L)

            elif isinstance(self.layer_lst[l], sbl.BoundLast):
                xA_L, tmp_bias_L = self.layer_lst[l].bound_back()  

            else:
                xA_L, tmp_bias_L = self.layer_lst[l].snn_bound_back(self.opts_lst[l], xA_L)

            bias_L += tmp_bias_L


        crown_L = torch.stack([_get_bound(xA_L[t], self.s_U[t], self.s_L[t], -1, t) for t in range(self.time_step_certify)], dim=0).sum(0) + bias_L 

        return crown_L, ibp_L



class MNIST_FC(torch.nn.Module):
    def __init__(self):
        super(MNIST_FC, self).__init__()

        self.flat   = sbl.BoundFlatten()

        self.fc1    = sbl.BoundLinear(784, 512)
        self.mem_f1 = sbl.BoundMemUpdate()

        self.fc2    = sbl.BoundLinear(512, 256)
        self.mem_f2 = sbl.BoundMemUpdate()

        self.fc3    = sbl.BoundLinear(256, 10)
        self.mem_f3 = sbl.BoundMemUpdate()

        self.classify = sbl.BoundLast()

        self.layer_lst = [self.flat, self.mem_f1, self.mem_f2, self.mem_f3, self.classify]
        self.opts_lst = [None, self.fc1, self.fc2, self.fc3, None]
        self.get_para()


    def get_para(self):
        self.time_step_certify = get_time_step_certify()

        for layer in self.layer_lst:
            layer.get_para()


    def forward(self, x_):
        x = torch.stack([x_ for _ in range(time_step)], dim=0)
        x = x > torch.rand(x.size()).cuda()
        x = x.float()

        s = self.flat(x)
        m, s = self.mem_f1(self.fc1, s)
        m, s = self.mem_f2(self.fc2, s)
        m, s = self.mem_f3(self.fc3, s)

        output = self.classify(s)

        return output


    def snn_bound(self, x_, eps, labels): 

        s_U = torch.stack([x_ + eps for _ in range(self.time_step_certify)], dim=0)
        s_L = torch.stack([x_ - eps for _ in range(self.time_step_certify)], dim=0)
        tmp = torch.rand(s_U.size()).cuda()
        s_U, s_L = (s_U > tmp), (s_L > tmp)

        s_U, s_L = s_U.float(), s_L.float()
        self.s_U = s_U.clone()
        self.s_L = s_L.clone()

        self.diff = (s_U - s_L).sum()

        for l in range(len(self.layer_lst)):

            if isinstance(self.layer_lst[l], sbl.BoundFlatten):
                s_U, s_L = self.layer_lst[l].bound(s_U, s_L)

            elif isinstance(self.layer_lst[l], sbl.BoundLast):
                output_U, output_L = self.layer_lst[l].bound(s_U, s_L, labels)  

            else:
                m_U, s_U, m_L, s_L = self.layer_lst[l].snn_bound(self.opts_lst[l], s_U, s_L)
            
        return output_U, output_L


    def snn_crown_ibp(self, x_, eps, labels):
        # apply forward IBP
        ibp_U, ibp_L = self.snn_bound(x_, eps, labels)

        xA_L, bias_L = 0, 0, 

        for l in range(len(self.layer_lst) - 1, -1, -1):

            if isinstance(self.layer_lst[l], sbl.BoundFlatten):
                xA_L, tmp_bias_L = self.layer_lst[l].bound_back(xA_L)

            elif isinstance(self.layer_lst[l], sbl.BoundLast):
                xA_L, tmp_bias_L = self.layer_lst[l].bound_back()  

            else:
                xA_L, tmp_bias_L = self.layer_lst[l].snn_bound_back(self.opts_lst[l], xA_L)

            bias_L += tmp_bias_L


        crown_L = torch.stack([_get_bound(xA_L[t], self.s_U[t], self.s_L[t], -1, t) for t in range(self.time_step_certify)], dim=0).sum(0) + bias_L 

        return crown_L, ibp_L


