import numpy as np
import copy
import torch


class BoxCreator(object):
    def __init__(self):
        self.box_list = []  # generated box list

    def reset(self):
        self.box_list.clear()

    def generate_box_size(self, **kwargs):
        pass

    def preview(self, length):
        """
        :param length:
        :return: list
        """
        while len(self.box_list) < length:
            self.generate_box_size()
        return copy.deepcopy(self.box_list[:length])

    def drop_box(self):
        assert len(self.box_list) >= 0
        self.box_list.pop(0)


class RandomBoxCreator(BoxCreator):
    default_box_set = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                default_box_set.append((2 + i, 2 + j, 2 + k))

    def __init__(self, box_size_set=None):
        super().__init__()
        self.box_set = box_size_set
        if self.box_set is None:
            self.box_set = RandomBoxCreator.default_box_set
        # print(self.box_set)

    def generate_box_size(self, **kwargs):
        idx = np.random.randint(0, len(self.box_set))
        # print("box size: ", self.box_set[idx])
        self.box_list.append(self.box_set[idx])
        # print("box list: ", self.box_list)

class LoadOrderCreator(BoxCreator):
    def __init__(self, box_size_set=None):
        super().__init__()
        self.box_set = box_size_set
        if self.box_set is None:
            self.box_set = RandomBoxCreator.default_box_set
        # print("box set: ", self.box_set)
        self.index = 0
        self.box_index = 0

    def generate_box_size(self, **kwargs):
        # if self.box_index == 0:
        #     self.box_list=self.box_set
        #     self.box_index += 1
        
        if self.box_index < len(self.box_set):
            self.box_list.append(self.box_set[self.box_index])
            self.box_index += 1
        else:
            self.box_list.append((550, 550, 500))
            self.box_index += 1
# load data

class LoadCutCreator(BoxCreator):
    def __init__(self, data_name=None):
        super().__init__()
        self.data_name = data_name
        self.cut_trajs = torch.load(self.data_name)
        print("load data set successfully, data name: ", self.data_name)
        self.index = 0
        self.cut_index = 0
        self.traj_nums = len(self.cut_trajs)
class LoadBoxCreator(BoxCreator):
    def __init__(self, data_name=None):  # data url
        super().__init__()  
        self.data_name = data_name
        self.index = 0
        self.box_index = 0
        self.traj_nums = len(torch.load(self.data_name))  
        print("load data set successfully, data name: ", self.data_name)

    def reset(self, index=None):
        self.box_list.clear()
        box_trajs = torch.load(self.data_name)
        self.recorder = []
        if index is None:
            self.index += 1
        else:
            self.index = index
        self.boxes = box_trajs[self.index]
        self.box_index = 0
        self.box_set = self.boxes
        self.box_set.append([10, 10, 10])

    def generate_box_size(self, **kwargs):
        if self.box_index < len(self.box_set):
            self.box_list.append(self.box_set[self.box_index])
            self.recorder.append(self.box_set[self.box_index])
            self.box_index += 1
        else:
            self.box_list.append((10, 10, 10))
            self.recorder.append((10, 10, 10))
            self.box_index += 1
