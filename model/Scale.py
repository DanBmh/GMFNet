import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AveargeJoint(nn.Module):

    def __init__(self):
        super().__init__()

        self.left_leg_up = [0, 1]
        self.left_leg_down = [2, 3]
        self.right_leg_up = [4, 5]
        self.right_leg_down = [6, 7]
        self.torso = [8, 9]
        self.head = [10, 11]
        self.left_arm_up = [12, 13]
        self.left_arm_down = [14, 15, 16]
        self.right_arm_up = [17, 18]
        self.right_arm_down = [19, 20, 21]

    def forward(self, x):
        """
        x: [32, 66, 40]
        return: [32, 30, 40]
        """
        b, v, t = x.shape
        x = x.reshape(b, 3, v // 3, t)  # [32, 66, 40] -> [32, 3, 22, 40]

        x_leftlegup = F.avg_pool2d(x[:, :, self.left_leg_up, :], kernel_size=(2, 1))
        x_leftlegdown = F.avg_pool2d(x[:, :, self.left_leg_down, :], kernel_size=(2, 1))
        x_rightlegup = F.avg_pool2d(x[:, :, self.right_leg_up, :], kernel_size=(2, 1))
        x_rightlegdown = F.avg_pool2d(x[:, :, self.right_leg_down, :], kernel_size=(2, 1))
        x_torso = F.avg_pool2d(x[:, :, self.torso, :], kernel_size=(2, 1))
        x_head = F.avg_pool2d(x[:, :, self.head, :], kernel_size=(2, 1))
        x_leftarmup = F.avg_pool2d(x[:, :, self.left_arm_up, :], kernel_size=(2, 1))
        x_leftarmdown = F.avg_pool2d(x[:, :, self.left_arm_down, :], kernel_size=(3, 1))
        x_rightarmup = F.avg_pool2d(x[:, :, self.right_arm_up, :], kernel_size=(2, 1))
        x_rightarmdown = F.avg_pool2d(x[:, :, self.right_arm_down, :], kernel_size=(3, 1))
        x_part = torch.cat((x_leftlegup, x_leftlegdown, x_rightlegup, x_rightlegdown, x_torso, x_head, x_leftarmup,
                            x_leftarmdown, x_rightarmup, x_rightarmdown), dim=2)

        x_part = x_part.reshape(b, -1, t).contiguous()

        return x_part


class AveargePart(nn.Module):

    def __init__(self):
        super().__init__()

        self.left_leg = [0, 1, 2, 3]
        self.right_leg = [4, 5, 6, 7]
        self.torso = [8, 9, 10, 11]
        self.left_arm = [12, 13, 14, 15, 16]
        self.right_arm = [17, 18, 19, 20, 21]

    def forward(self, x):
        """
        x: [32, 66, 40]
        return: [32, 15, 40]
        """
        b, v, t = x.shape
        x = x.reshape(b, 3, v // 3, t)  # [32, 66, 40] -> [32, 3, 22, 40]

        x_leftleg = F.avg_pool2d(x[:, :, self.left_leg, :], kernel_size=(4, 1))
        x_rightleg = F.avg_pool2d(x[:, :, self.right_leg, :], kernel_size=(4, 1))
        x_torso = F.avg_pool2d(x[:, :, self.torso, :], kernel_size=(4, 1))
        x_leftarm = F.avg_pool2d(x[:, :, self.left_arm, :], kernel_size=(5, 1))
        x_rightarm = F.avg_pool2d(x[:, :, self.right_arm, :], kernel_size=(5, 1))
        x_body = torch.cat((x_leftleg, x_rightleg, x_torso, x_leftarm, x_rightarm), dim=2)

        x_body = x_body.reshape(b, -1, t).contiguous()

        return x_body


class PartLocalInform(nn.Module):

    def __init__(self):
        super().__init__()

        self.left_leg_up = [0, 1]
        self.left_leg_down = [2, 3]
        self.right_leg_up = [4, 5]
        self.right_leg_down = [6, 7]
        self.torso = [8, 9]
        self.head = [10, 11]
        self.left_arm_up = [12, 13]
        self.left_arm_down = [14, 15, 16]
        self.right_arm_up = [17, 18]
        self.right_arm_down = [19, 20, 21]

    def forward(self, part):
        b, v, t = part.shape
        part = part.reshape(b, 3, v // 3, t)

        # x = part.new_zeros((b, 3, 22, t))
        #
        # x[:, :, self.left_leg_up, :] = torch.cat((part[:, :, 0, :].unsqueeze(-2), part[:, :, 0, :].unsqueeze(-2)), 2)
        # x[:, :, self.left_leg_down, :] = torch.cat((part[:, :, 1, :].unsqueeze(-2), part[:, :, 1, :].unsqueeze(-2)), 2)
        # x[:, :, self.right_leg_up, :] = torch.cat((part[:, :, 2, :].unsqueeze(-2), part[:, :, 2, :].unsqueeze(-2)), 2)
        # x[:, :, self.right_leg_down, :] = torch.cat((part[:, :, 3, :].unsqueeze(-2), part[:, :, 3, :].unsqueeze(-2)), 2)
        # x[:, :, self.torso, :] = torch.cat((part[:, :, 4, :].unsqueeze(-2), part[:, :, 4, :].unsqueeze(-2)), 2)
        # x[:, :, self.head, :] = torch.cat((part[:, :, 5, :].unsqueeze(-2), part[:, :, 5, :].unsqueeze(-2)), 2)
        # x[:, :, self.left_arm_up, :] = torch.cat((part[:, :, 6, :].unsqueeze(-2), part[:, :, 6, :].unsqueeze(-2)), 2)
        # x[:, :, self.left_arm_down, :] = torch.cat((part[:, :, 7, :].unsqueeze(-2), part[:, :, 7, :].unsqueeze(-2), part[:, :, 7].unsqueeze(-2)), 2)
        # x[:, :, self.right_arm_up, :] = torch.cat((part[:, :, 8, :].unsqueeze(-2), part[:, :, 8, :].unsqueeze(-2)), 2)
        # x[:, :, self.right_arm_down, :] = torch.cat((part[:, :, 9, :].unsqueeze(-2), part[:, :, 9, :].unsqueeze(-2), part[:, :, 9].unsqueeze(-2)), 2)

        x0 = torch.cat((part[:, :, 0, :].unsqueeze(-2), part[:, :, 0, :].unsqueeze(-2)), 2)
        x1 = torch.cat((part[:, :, 1, :].unsqueeze(-2), part[:, :, 1, :].unsqueeze(-2)), 2)
        x2 = torch.cat((part[:, :, 2, :].unsqueeze(-2), part[:, :, 2, :].unsqueeze(-2)), 2)
        x3 = torch.cat((part[:, :, 3, :].unsqueeze(-2), part[:, :, 3, :].unsqueeze(-2)), 2)
        x4 = torch.cat((part[:, :, 4, :].unsqueeze(-2), part[:, :, 4, :].unsqueeze(-2)), 2)
        x5 = torch.cat((part[:, :, 5, :].unsqueeze(-2), part[:, :, 5, :].unsqueeze(-2)), 2)
        x6 = torch.cat((part[:, :, 6, :].unsqueeze(-2), part[:, :, 6, :].unsqueeze(-2)), 2)
        x7 = torch.cat((part[:, :, 7, :].unsqueeze(-2), part[:, :, 7, :].unsqueeze(-2), part[:, :, 7].unsqueeze(-2)), 2)
        x8 = torch.cat((part[:, :, 8, :].unsqueeze(-2), part[:, :, 8, :].unsqueeze(-2)), 2)
        x9 = torch.cat((part[:, :, 9, :].unsqueeze(-2), part[:, :, 9, :].unsqueeze(-2), part[:, :, 9].unsqueeze(-2)), 2)

        x = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8, x9), dim=2)

        x = x.reshape(b, -1, t).contiguous()

        return x


class BodyLocalInform(nn.Module):

    def __init__(self):
        super().__init__()

        self.left_leg = [0, 1, 2, 3]
        self.right_leg = [4, 5, 6, 7]
        self.torso = [8, 9, 10, 11]
        self.left_arm = [12, 13, 14, 15, 16]
        self.right_arm = [17, 18, 19, 20, 21]

    def forward(self, body):

        b, v, t = body.shape
        body = body.reshape(b, 3, v // 3, t)
        x = body.new_zeros((b, 3, 22, t))

        # x[:, :, self.left_leg, :] = torch.cat((body[:, :, 0:1, :], body[:, :, 0:1, :], body[:, :, 0:1, :], body[:, :, 0:1, :]), -2)
        # x[:, :, self.right_leg, :] = torch.cat((body[:, :, 1:2, :], body[:, :, 1:2, :], body[:, :, 1:2, :], body[:, :, 2:3, :]), -2)
        # x[:, :, self.torso, :] = torch.cat((body[:, :, 2:3, :], body[:, :, 2:3, :], body[:, :, 2:3, :], body[:, :, 1:2, :]), -2)
        # x[:, :, self.left_arm, :] = torch.cat((body[:, :, 3:4, :], body[:, :, 3:4, :], body[:, :, 3:4, :], body[:, :, 3:4, :], body[:, :, 3:4, :]), -2)
        # x[:, :, self.right_arm, :] = torch.cat((body[:, :, 4:5, :], body[:, :, 4:5, :], body[:, :, 4:5, :], body[:, :, 4:5, :], body[:, :, 4:5, :]), -2)

        x0 = torch.cat((body[:, :, 0:1, :], body[:, :, 0:1, :], body[:, :, 0:1, :], body[:, :, 0:1, :]), -2)
        x1 = torch.cat((body[:, :, 1:2, :], body[:, :, 1:2, :], body[:, :, 1:2, :], body[:, :, 2:3, :]), -2)
        x2 = torch.cat((body[:, :, 2:3, :], body[:, :, 2:3, :], body[:, :, 2:3, :], body[:, :, 1:2, :]), -2)
        x3 = torch.cat((body[:, :, 3:4, :], body[:, :, 3:4, :], body[:, :, 3:4, :], body[:, :, 3:4, :], body[:, :, 3:4, :]), -2)
        x4 = torch.cat((body[:, :, 4:5, :], body[:, :, 4:5, :], body[:, :, 4:5, :], body[:, :, 4:5, :], body[:, :, 4:5, :]), -2)

        x = torch.cat((x0, x1, x2, x3, x4), dim=2)

        x = x.reshape(b, -1, t).contiguous()

        return x
