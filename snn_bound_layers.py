import torch
import torch.nn as nn
from torch.nn import Conv2d, Linear
import torch.nn.functional as F

from snn_fire import m_th, alpha, time_step, get_time_step_certify, fire_func


class BoundEncodeConv2d(Conv2d):
    def __init__(self, cin, cout, kernel_size=3, stride=1, padding=1, bias=True):
        super(BoundEncodeConv2d, self).__init__(in_channels=cin, out_channels=cout, kernel_size=kernel_size, 
                stride=stride, padding=padding, dilation=1, groups=1, bias=bias)


    def forward(self, input):
        output = super(BoundEncodeConv2d, self).forward(input)
        return output


    def snn_bound(self, x_U, x_L):
        mid = (x_U + x_L) / 2.0
        diff = (x_U - x_L) / 2.0
        weight_abs = self.weight.abs()

        deviation = F.conv2d(diff, weight_abs, None, self.stride, self.padding, self.dilation, self.groups)
        center = F.conv2d(mid, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        self.output_shape = center.size()[1:]
        self.input_shape = x_L.size()[1:]

        m_U = center + deviation
        m_L = center - deviation

        return m_U, m_L


    def snn_bound_back(self, last_mA_L): 
        shape = last_mA_L.size()
        output_padding0 = int(self.input_shape[1]) - (int(self.output_shape[1]) - 1) * self.stride[0] + 2 * self.padding[0] - int(self.weight.size()[2])
        output_padding1 = int(self.input_shape[2]) - (int(self.output_shape[2]) - 1) * self.stride[1] + 2 * self.padding[1] - int(self.weight.size()[3]) 
        
        next_xA_L = F.conv_transpose2d(last_mA_L.view(shape[0] * shape[1], *shape[2:]), self.weight, None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, output_padding=(output_padding0, output_padding1))     
        next_xA_L = next_xA_L.view(shape[0], shape[1], *next_xA_L.shape[1:])

        # get bias
        bias_L = (last_mA_L.sum((3,4)) * self.bias).sum(2)

        return next_xA_L, bias_L



class BoundEncodeLinear(Linear):
    def __init__(self, cin, cout, bias=True):
        super(BoundEncodeLinear, self).__init__(cin, cout, bias)

    def forward(self, input):
        output = super(BoundEncodeLinear, self).forward(input)
        return output

    def snn_bound(self, x_U, x_L):
        mid = (x_U + x_L) / 2.0
        diff = (x_U - x_L) / 2.0
        weight_abs = self.weight.abs()

        center = torch.addmm(self.bias, mid, self.weight.t())
        deviation = diff.matmul(weight_abs.t())

        m_U = center + deviation
        m_L = center - deviation

        return m_U, m_L

    def snn_bound_back(self, last_mA_L):
        next_xA_L = last_mA_L.matmul(self.weight)
        bias_L = last_mA_L.matmul(self.bias)

        return next_xA_L, bias_L


class BoundFlatten(nn.Module):
    def __init__(self):
        super(BoundFlatten, self).__init__()
        self.get_para()


    def get_para(self):
        self.time_step_certify = get_time_step_certify()


    def forward(self, x):
        self.shape = x.size()[2:]
        return x.view(time_step, x.shape[1], -1)


    # x_U size: T, B, C, H, W -> T, B, CHW
    def bound(self, x_U, x_L):
        if isinstance(x_U, list):

            self.shape = x_U[0].size()[1:] # C, H, W
            x_U_lst, x_L_lst = [], []

            for i in range(len(x_U)):
                x_U_lst.append(x_U[i].view(x_U[i].size(0), -1))
                x_L_lst.append(x_L[i].view(x_L[i].size(0), -1))

            return x_U_lst, x_L_lst

        else:
            self.shape = x_U.size()[2:]
            return x_U.view(x_U.size(0), x_U.size(1), -1), x_L.view(x_L.size(0), x_L.size(1), -1)


    def bound_back(self, xA_L):
        # X_L size: T, 9, B, CHW -> T, 9, B, C, H, W
        if isinstance(xA_L, list):
            xA_L_lst = []
            for i in range(len(xA_L)):
                xA_L_lst.append(xA_L[i].view(xA_L[i].size(0), xA_L[i].size(1), *self.shape))

            return xA_L_lst, 0

        else:
            return xA_L.view(xA_L.size(0), xA_L.size(1), xA_L.size(2), *self.shape), 0


class BoundConv2d(Conv2d):
    def __init__(self, cin, cout, kernel_size=3, stride=1, padding=1, bias=True):
        super(BoundConv2d, self).__init__(in_channels=cin, out_channels=cout, kernel_size=kernel_size, 
                stride=stride, padding=padding, dilation=1, groups=1, bias=bias)


    def forward(self, input):
        output = super(BoundConv2d, self).forward(input)
        return output


    def snn_bound(self, s_U, s_L):
        # compute fix: a neuron must fire
        m_FIX = F.conv2d(s_L, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.output_shape = m_FIX.size()[1:]
        self.input_shape = s_L.size()[1:]

        pos_w, neg_w = self.weight.clamp(min=0), self.weight.clamp(max=0)
        s_xor = (s_U + s_L) % 2

        m_U = m_FIX + F.conv2d(s_xor, pos_w, None, self.stride, self.padding, self.dilation, self.groups)
        m_L = m_FIX + F.conv2d(s_xor, neg_w, None, self.stride, self.padding, self.dilation, self.groups)

        return m_U, m_L


    def snn_bound_back(self, last_mA_L): 
        shape = last_mA_L.size()
        output_padding0 = int(self.input_shape[1]) - (int(self.output_shape[1]) - 1) * self.stride[0] + 2 * self.padding[0] - int(self.weight.size()[2])
        output_padding1 = int(self.input_shape[2]) - (int(self.output_shape[2]) - 1) * self.stride[1] + 2 * self.padding[1] - int(self.weight.size()[3]) 
        
        # get sA
        next_sA_L = F.conv_transpose2d(last_mA_L.view(shape[0] * shape[1], *shape[2:]), self.weight, None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, output_padding=(output_padding0, output_padding1))     
        next_sA_L = next_sA_L.view(shape[0], shape[1], *next_sA_L.shape[1:])

        # get bias
        bias_L = (last_mA_L.sum((3,4)) * self.bias).sum(2)

        return next_sA_L, bias_L



class BoundLinear(Linear):
    def __init__(self, cin, cout, bias=True):
        super(BoundLinear, self).__init__(cin, cout, bias)


    def snn_bound(self, s_U, s_L):
        # compute fix: a neuron must fire
        m_FIX = torch.addmm(self.bias, s_L, self.weight.t())

        pos_w, neg_w = self.weight.clamp(min=0), self.weight.clamp(max=0)
        s_xor = (s_U + s_L) % 2

        m_U = m_FIX + torch.mm(s_xor, pos_w.t())
        m_L = m_FIX + torch.mm(s_xor, neg_w.t())

        return m_U, m_L


    def snn_bound_back(self, last_mA_L):
        next_sA_L = last_mA_L.matmul(self.weight)
        bias_L = last_mA_L.matmul(self.bias)

        return next_sA_L, bias_L
        


class BoundLast(nn.Module):
    def __init__(self, cin=None, num_class=10, linear=False):
        super(BoundLast, self).__init__()
        self.num_class = num_class
        self.linear = linear
        if linear:
            self.fc = nn.Linear(cin, num_class)

        self.eye_C = torch.eye(self.num_class).cuda()
        self.zeros_C = torch.zeros(self.num_class).cuda()

        self.get_para()

    def get_para(self):
        self.time_step_certify = get_time_step_certify()

    
    # x is the spike with size [T, B, C]
    def forward(self, x):
        x = torch.stack(x, dim=0)
        x = x.mean(dim=0)
        if self.linear:
            x = self.fc(x)

        return x


    # IBP interval 
    def bound(self, x_U, x_L, labels):
        x_U, x_L = torch.stack(x_U, dim=0), torch.stack(x_L, dim=0)
        x_U, x_L = x_U.mean(dim=0), x_L.mean(dim=0)

        # get C matrix
        labels = labels.long()
        C = torch.eye(self.num_class).type_as(x_U)[labels].unsqueeze(1) - torch.eye(self.num_class).type_as(x_U).unsqueeze(0) 
        I = (~(labels.data.unsqueeze(1) == torch.arange(self.num_class).type_as(labels.data).unsqueeze(0)))
        C = (C[I].view(x_U.size(0),self.num_class-1,self.num_class))
        self.C = C

        if self.linear:
            w = C.matmul(self.fc.weight)
            b = C.matmul(self.fc.bias)
        else:
            w = C.matmul(self.eye_C)
            b = C.matmul(self.zeros_C)

        mid, diff = (x_U + x_L) / 2.0, (x_U - x_L) / 2.0

        center = w.matmul(mid.unsqueeze(-1)) + b.unsqueeze(-1)
        deviation = w.abs().matmul(diff.unsqueeze(-1))

        center = center.squeeze(-1)
        deviation = deviation.squeeze(-1)

        o_U, o_L = center + deviation, center - deviation

        return o_U, o_L


    def bound_back(self):
        if self.linear:
            A = self.C.matmul(self.fc.weight)
            b = self.C.matmul(self.fc.bias)
        else:
            A = self.C.matmul(self.eye_C)
            b = self.C.matmul(self.zeros_C)
        
        A = [A / (1.0 * self.time_step_certify) for _ in range(self.time_step_certify)]

        return A, b



class BoundMemUpdate(nn.Module):
    def __init__(self):
        super(BoundMemUpdate, self).__init__()
        self.get_para()

    def get_para(self):
        self.time_step_certify = get_time_step_certify()


    def forward(self, opts, x):
        m, s = [], []

        for t in range(time_step):
            m.append(opts(x[t]))
            if t > 0:
                m[t] += (m[t - 1] * (1 - s[t - 1]) * alpha)
            s.append(fire_func(m[t]))

        return m, s


    def snn_bound(self, opts, x_U, x_L):
        m_U, m_L, s_U, s_L = [], [], [], []
        
        for t in range(self.time_step_certify):
            # spatial propagate
            spatial_bound = opts.snn_bound(x_U[t], x_L[t])
            m_U.append(spatial_bound[0])
            m_L.append(spatial_bound[1])

            # temporal propagate
            if t > 0:
                m_U_pre, m_L_pre = m_U[t - 1], m_L[t - 1]
                s_U_pre, s_L_pre = s_U[t - 1], s_L[t - 1]

                # initial & always fire
                m_U_tmp = torch.zeros_like(m_U_pre)
                m_L_tmp = torch.zeros_like(m_L_pre)

                # always not fire
                idx0 = (s_U_pre == 0)
                m_U_tmp[idx0], m_L_tmp[idx0] = m_U_pre[idx0], m_L_pre[idx0]

                # unstable
                idx2 = (s_U_pre == 1) * (s_L_pre == 0)
                m_U_tmp[idx2] = m_th

                idx3 = idx2 * (m_L_pre < 0)
                m_L_tmp[idx3] = m_L_pre[idx3]

                m_U[t] += (m_U_tmp * alpha)
                m_L[t] += (m_L_tmp * alpha)

            # fire propagate
            s_U.append(fire_func(m_U[t]))
            s_L.append(fire_func(m_L[t]))

        self.m_U, self.m_L = m_U, m_L
        self.s_U, self.s_L = s_U, s_L
        
        return m_U, s_U, m_L, s_L


    @staticmethod
    def _temporal_bound_backward1(last_mA_L, x_U, x_L, y_U, y_L): # treat s as x; m as y

        last_pos_mA_L, last_neg_mA_L = last_mA_L.clamp(min=0), last_mA_L.clamp(max=0)

        x_U, x_L = x_U.unsqueeze(1), x_L.unsqueeze(1)
        y_U, y_L = y_U.unsqueeze(1), y_L.unsqueeze(1)

        x_U = x_U.view(x_U.size(0), x_U.size(1), -1)
        x_L = x_L.view(x_U.shape)
        y_U = y_U.view(x_U.shape)
        y_L = y_L.view(x_U.shape)

        # x always 0: xy=0
        # x always 1: xy=y
        next_yA_L = x_L * last_mA_L

        # unstable x*y_l < xy < x*y_u
        idx2 = (x_U == 1) * (x_L == 0)
        y_L_tmp, y_U_tmp = torch.zeros_like(y_L), torch.zeros_like(y_U)
        y_L_tmp[idx2] = y_L[idx2]
        y_U_tmp[idx2] = y_U[idx2]
        next_xA_L = y_L_tmp * last_pos_mA_L + y_U_tmp * last_neg_mA_L

        shape_A = last_mA_L.shape

        next_xA_L = next_xA_L.view(shape_A[0], shape_A[1], -1) 
        next_yA_L = next_yA_L.view(shape_A[0], shape_A[1], -1) 

        return next_xA_L, next_yA_L, 0


    @staticmethod
    def _temporal_bound_backward2(last_mA_L, x_U, x_L, y_U, y_L): # treat s as x; m as y

        last_pos_mA_L, last_neg_mA_L = last_mA_L.clamp(min=0), last_mA_L.clamp(max=0)

        x_U, x_L = x_U.unsqueeze(1), x_L.unsqueeze(1)
        y_U, y_L = y_U.unsqueeze(1), y_L.unsqueeze(1)

        x_U = x_U.view(x_U.size(0), x_U.size(1), -1)
        x_L = x_L.view(x_U.shape)
        y_U = y_U.view(x_U.shape)
        y_L = y_L.view(x_U.shape)

        # x always 0: xy=0
        # x always 1: xy=y
        next_yA_L = x_L * last_mA_L

        # unstable 
        idx2 = (x_U == 1) * (x_L == 0)
        idx3 = idx2 * (y_L < 0)

        y_L_tmp, y_U_tmp = torch.zeros_like(y_L), torch.zeros_like(y_U)
        y_L_tmp[idx3] = y_L[idx3]
        y_U_tmp[idx3] = y_U[idx3]

        lower_d = y_L_tmp / (y_L_tmp - y_U_tmp - 1e-10)
        lower_b = (- lower_d * y_U_tmp)

        next_yA_L += (lower_d * last_pos_mA_L + idx2 * last_neg_mA_L)

        shape_A = last_mA_L.shape
        next_yA_L = next_yA_L.view(shape_A[0], shape_A[1], -1) 

        next_bias_L = last_pos_mA_L.matmul(lower_b.view(lower_b.size(0), -1, 1))
        next_bias_L = next_bias_L.squeeze(-1)

        return torch.zeros_like(next_yA_L), next_yA_L, next_bias_L


    @staticmethod
    def _temporal_bound_backward3(last_mA_L, x_U, x_L, y_U, y_L): # treat s as x; m as y

        last_pos_mA_L, last_neg_mA_L = last_mA_L.clamp(min=0), last_mA_L.clamp(max=0)

        x_U, x_L = x_U.unsqueeze(1), x_L.unsqueeze(1)
        y_U, y_L = y_U.unsqueeze(1), y_L.unsqueeze(1)

        x_U = x_U.view(x_U.size(0), x_U.size(1), -1)
        x_L = x_L.view(x_U.shape)
        y_U = y_U.view(x_U.shape)
        y_L = y_L.view(x_U.shape)

        # x always 0: xy=0
        # x always 1: xy=y
        next_yA_L = x_L * last_mA_L

        # unstable x*y_l < xy < x*m_th
        idx2 = (x_U == 1) * (x_L == 0)
        y_L_tmp, y_U_tmp = torch.zeros_like(y_L), torch.zeros_like(y_U)
        y_L_tmp[idx2] = y_L[idx2]
        y_U_tmp[idx2] = m_th
        next_xA_L = y_L_tmp * last_pos_mA_L + y_U_tmp * last_neg_mA_L

        shape_A = last_mA_L.shape

        next_xA_L = next_xA_L.view(shape_A[0], shape_A[1], -1) 
        next_yA_L = next_yA_L.view(shape_A[0], shape_A[1], -1) 

        return next_xA_L, next_yA_L, 0



    @staticmethod
    def _fire_bound_backward(last_sA_L, m_U, m_L, s_U, s_L):

        # initial settings, parallel bound type
        upper_d, upper_b = torch.zeros(m_U.shape).cuda(), torch.ones(m_U.shape).cuda()
        lower_d, lower_b = torch.zeros(m_U.shape).cuda(), torch.zeros(m_U.shape).cuda()

        # always fire
        idx1 = (s_L == 1)
        lower_b[idx1] = 1

        # always not fire
        idx0 = (s_U == 0)
        upper_b[idx0] = 0

        # unstable points
        idx2 = (s_U == 1) * (s_L == 0)
        idx3 = idx2 * ((m_U - m_th) >= (m_th - m_L))
        idx4 = idx2 * ((m_U - m_th) <  (m_th - m_L))

        # if abs(m_U - m_th) is larger: upper func not change
        lower_d[idx3] = (1 / (m_U[idx3] - m_th))
        lower_b[idx3] = (lower_d * (-m_th))[idx3]

        # if abs(m_L - m_th) is larger: lower func not change
        upper_d[idx4] = (1 / (m_th - m_L[idx4]))
        upper_b[idx4] = (upper_d * (-m_L))[idx4]

        lower_d = lower_d.unsqueeze(1)
        lower_b = lower_b.view(lower_b.size(0), -1, 1)
        upper_d = upper_d.unsqueeze(1)
        upper_b = upper_b.view(upper_b.size(0), -1, 1)

        last_pos_sA_L, last_neg_sA_L = last_sA_L.clamp(min=0), last_sA_L.clamp(max=0)

        next_mA_L = upper_d * last_neg_sA_L + lower_d * last_pos_sA_L
        next_mA_L = next_mA_L.view(last_sA_L.shape[0], last_sA_L.shape[1], -1)

        last_pos_sA_L = last_pos_sA_L.view(last_sA_L.size(0), last_sA_L.size(1), -1)
        last_neg_sA_L = last_neg_sA_L.view(last_sA_L.size(0), last_sA_L.size(1), -1)

        next_bias_L = last_neg_sA_L.matmul(upper_b) + last_pos_sA_L.matmul(lower_b)
        next_bias_L = next_bias_L.squeeze(-1)

        return next_mA_L, next_bias_L


    def snn_bound_back(self, opts, last_sA_L):

        next_mA_L = [0 for _ in range(self.time_step_certify)]
        next_sA_L = [0 for _ in range(self.time_step_certify)]
        bias_L = 0

        for t in range(self.time_step_certify - 1, - 1, -1):

            # fire propagation
            fp_mA_L, fp_bias_L = self._fire_bound_backward(last_sA_L[t], self.m_U[t], self.m_L[t], self.s_U[t], self.s_L[t])

            next_mA_L[t] += fp_mA_L

            # temporal propagation
            tp_bias_L, s_bias_L = 0, 0

            if t > 0:
                # alpha*u*(1-s) -> u*(1-s) 
                tmp_mA_L = alpha * next_mA_L[t]

                # upper and lower bound for 1-s  
                s_U, m_U = 1 - self.s_L[t - 1], self.m_U[t - 1]
                s_L, m_L = 1 - self.s_U[t - 1], self.m_L[t - 1]

                # back propagate of alpha*(1-s) treat s as x
                tp_sA_L, tp_mA_L, tp_bias_L = self._temporal_bound_backward3(tmp_mA_L, s_U, s_L, m_U, m_L)


                next_mA_L[t - 1] += tp_mA_L

                # back propagate of 1-s   
                s_bias_L = tp_sA_L.sum(-1)
                last_sA_L[t - 1] -= tp_sA_L.view(last_sA_L[t - 1].shape)


            # spatial propagate 
            next_sA_L[t], sp_bias_L = opts.snn_bound_back(next_mA_L[t].view(last_sA_L[t].shape))

            bias_L += (sp_bias_L + tp_bias_L + s_bias_L + fp_bias_L)

        return next_sA_L, bias_L

            



