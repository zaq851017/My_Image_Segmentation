import torch
import torch.nn as nn
from torch.autograd import Variable

# Batch x NumChannels x Height x Width
# UNET --> BatchSize x 1 (3?) x 240 x 240
# BDCLSTM --> BatchSize x 64 x 240 x240

''' Class CLSTMCell.
    This represents a single node in a CLSTM series.
    It produces just one time (spatial) step output.
'''


class CLSTMCell(nn.Module):

    # Constructor
    def __init__(self, input_channels, hidden_channels,
                 kernel_size, bias=True):
        super(CLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(self.input_channels + self.hidden_channels,
                              self.num_features * self.hidden_channels,
                              self.kernel_size,
                              1,
                              self.padding)

    # Forward propogation formulation
    def forward(self, x, h, c):
        # print('x: ', x.type)
        # print('h: ', h.type)
        if len(x.shape) == 3: # batch, H, W 
            x = x.unsqueeze(dim = 1)
        combined = torch.cat((x, h), dim=1)
        A = self.conv(combined)

        (Ai, Af, Ao, Ag) = torch.split(A,
                                       A.size()[1] // self.num_features,
                                       dim=1)

        i = torch.sigmoid(Ai)     # input gate
        f = torch.sigmoid(Af)     # forget gate
        o = torch.sigmoid(Ao)     # output gate
        g = torch.tanh(Ag)

        c = c * f + i * g           # cell activation state
        h = o * torch.tanh(c)     # cell hidden state

        return h, c

    @staticmethod
    def init_hidden(batch_size, hidden_c, shape):
        try:
            return(Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])).cuda(),
               Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])).cuda())
        except:
            return(Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])),
                    Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])))


''' Class CLSTM.
    This represents a series of CLSTM nodes (one direction)
'''


class CLSTM(nn.Module):
    # Constructor
    def __init__(self, input_channels=64, hidden_channels=[64],
                 kernel_size=5, bias=True):
        super(CLSTM, self).__init__()

        # store stuff
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)

        self.bias = bias
        self.all_layers = []

        # create a node for each layer in the CLSTM
        for layer in range(self.num_layers):
            name = 'cell{}'.format(layer)
            cell = CLSTMCell(self.input_channels[layer],
                             self.hidden_channels[layer],
                             self.kernel_size,
                             self.bias)
            setattr(self, name, cell)
            self.all_layers.append(cell)

    # Forward propogation
    # x --> BatchSize x NumSteps x NumChannels x Height x Width
    #       BatchSize x 2 x 64 x 240 x 240
    def forward(self, x):
        bsize, steps, _, height, width = x.size()
        internal_state = []
        outputs = []
        for step in range(steps):
            input = torch.squeeze(x[:, step, :, :, :], dim=1)
            for layer in range(self.num_layers):
                # populate hidden states for all layers
                if step == 0:
                    (h, c) = CLSTMCell.init_hidden(bsize,
                                                   self.hidden_channels[layer],
                                                   (height, width))
                    internal_state.append((h, c))
                # do forward
                name = 'cell{}'.format(layer)
                (h, c) = internal_state[layer]

                input, c = getattr(self, name)(
                    input, h, c)  # forward propogation call
                internal_state[layer] = (input, c)
            outputs.append(input)

        #for i in range(len(outputs)):
        #    print(outputs[i].size())
        return outputs


class BDCLSTM(nn.Module):
    # Constructor
    def __init__(self, input_channels=64, hidden_channels=[64],
                 kernel_size=5, bias=True, num_classes=1):

        super(BDCLSTM, self).__init__()
        self.forward_net = CLSTM(
            input_channels, hidden_channels, kernel_size, bias)
        self.reverse_net = CLSTM(
            input_channels, hidden_channels, kernel_size, bias)
        self.conv = nn.Conv2d(
            2 * hidden_channels[-1], num_classes, kernel_size=1)
        self.soft = nn.Softmax2d()

    # Forward propogation
    # x --> BatchSize x NumChannels x Height x Width
    #       BatchSize x 64 x 240 x 240
    def forward(self, previous_list, current_frame, next_list):
        previous_frame = torch.tensor([]).cuda()
        for i in range(len(previous_list)):
            previous_frame = torch.cat((previous_frame, previous_list[i].unsqueeze(dim = 1)), dim = 1)
        xforward = torch.cat( (previous_frame, current_frame.unsqueeze(dim = 1)), dim = 1)
        next_frame = torch.tensor([]).cuda()
        for i in range(len(next_list)):
            next_frame = torch.cat((next_frame, next_list[i].unsqueeze(dim = 1)), dim = 1)
        xreverse = torch.cat( (current_frame.unsqueeze(dim = 1),  next_frame), dim = 1)
        # x1 = torch.unsqueeze(x1, dim=1)
        # x2 = torch.unsqueeze(x2, dim=1)
        # x3 = torch.unsqueeze(x3, dim=1)
        # xforward = torch.cat((x1, x2), dim=1)
        # xreverse = torch.cat((x3, x2), dim=1)
        yforward = self.forward_net(xforward)
        yreverse = self.reverse_net(xreverse)
        
        ycat = torch.cat((yforward[-1], yreverse[-1]), dim=1)
        y = self.conv(ycat)
        return y

class New_BDCLSTM(nn.Module):
    # Constructor
    def __init__(self, length, input_channels=64, hidden_channels=[64],
                 kernel_size=5, bias=True, num_classes=1):

        super(New_BDCLSTM, self).__init__()
        self.len = length
        self.forward_net = CLSTM(
            input_channels, hidden_channels, kernel_size, bias)
        self.conv = []
        for i in range(self.len):
            self.conv.append(nn.Conv2d(hidden_channels[-1], num_classes, kernel_size=1).cuda())
        # self.final_conv = nn.Conv2d(self.len, num_classes, kernel_size=1)
        self.final_conv = nn.Conv3d(self.len, num_classes, kernel_size=1)
    def forward(self, continue_list):
        F_concanate_frame = torch.tensor([]).cuda()
        for i in range(len(continue_list)):
            F_concanate_frame = torch.cat((F_concanate_frame, continue_list[i].unsqueeze(dim = 1)), dim = 1)
        yforward = self.forward_net(F_concanate_frame)
        total_y = torch.tensor([]).cuda()
        for i in range(self.len):
            F_y = self.conv[i](yforward[i])
            total_y = torch.cat( (total_y, F_y), dim = 1)
        # current_y = self.final_conv(total_y)
        current_y = self.final_conv(total_y.unsqueeze(dim = 2)).squeeze(dim = 1)
        return current_y, total_y


class Temp_New_BDCLSTM(nn.Module):
    # Constructor
    def __init__(self, input_channels=64, hidden_channels=[64],
                 kernel_size=5, bias=True, num_classes=1):

        super(Temp_New_BDCLSTM, self).__init__()
        self.forward_net = CLSTM(
            input_channels, hidden_channels, kernel_size, bias)
        self.conv1 = nn.Conv2d(
            hidden_channels[-1], num_classes, kernel_size=1)
        self.conv2 = nn.Conv2d(
            hidden_channels[-1], num_classes, kernel_size=1)
        self.conv3 = nn.Conv2d(
            hidden_channels[-1], num_classes, kernel_size=1)
        self.conv4 = nn.Conv2d(
            hidden_channels[-1], num_classes, kernel_size=1)
        self.conv5 = nn.Conv2d(
            hidden_channels[-1], num_classes, kernel_size=1)
        self.final_conv = nn.Conv2d(5, num_classes, kernel_size=1)
    # Forward propogation
    # x --> BatchSize x NumChannels x Height x Width
    #       BatchSize x 64 x 240 x 240
    def forward(self, previous_list, current_frame, next_list):
        concanate_frame = torch.tensor([]).cuda()
        for i in range(len(previous_list)):
            concanate_frame = torch.cat((concanate_frame, previous_list[i].unsqueeze(dim = 1)), dim = 1)
        concanate_frame= torch.cat( (concanate_frame, current_frame.unsqueeze(dim = 1)), dim = 1)
        for i in range(len(next_list)):
            concanate_frame = torch.cat((concanate_frame, next_list[i].unsqueeze(dim = 1)), dim = 1)
        yforward = self.forward_net(concanate_frame)
        y1 = self.conv1(yforward[0])
        y2 = self.conv2(yforward[1])
        y3 = self.conv3(yforward[2])
        y4 = self.conv4(yforward[3])
        y5 = self.conv5(yforward[4])
        total_y = torch.cat((y1, y2, y3, y4, y5), dim = 1)
        current_y = self.final_conv(total_y)
        return current_y, total_y