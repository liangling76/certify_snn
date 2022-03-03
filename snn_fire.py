import torch

# define parameters here
m_th = 0.25
m_len = 0.5
alpha = 0.25
beta = 1.0

time_step = 10
time_step_certify = 3


def set_time_step_certify(value):
    global time_step_certify
    time_step_certify = value


def get_time_step_certify():
    global time_step_certify
    return time_step_certify


class FireFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(m_th).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        tmp = abs(input - m_th) < m_len
        return grad_input * tmp.float() * beta

fire_func = FireFunc.apply