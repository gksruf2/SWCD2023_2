import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import Image
from util import pose_utils
import pandas as pd
import numpy as np
import torch

class FashionDistillationDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.set_defaults(load_size=256)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(old_size=(256, 176))
        parser.set_defaults(structure_nc=18)
        parser.set_defaults(image_nc=3)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.phase = opt.phase

        # prepare for image (image_dir), image_pair (name_pairs) and bone annotation (annotation_file)
        self.image_dir = os.path.join(self.root, self.phase)
        self.bone_file = os.path.join(self.root, 'fasion-resize-annotation-%s.csv' % self.phase)
        pairLst = os.path.join(self.root, 'fasion-resize-pairs-%s.csv' % self.phase)
        self.name_pairs = self.init_categories(pairLst)
        self.annotation_file = pd.read_csv(self.bone_file, sep=':')
        self.annotation_file = self.annotation_file.set_index('name')

        # load image size
        if isinstance(opt.loadSize, int):
            self.load_size = (opt.loadSize, opt.loadSize)
        else:
            self.load_size = opt.loadSize

        # prepare for transformation
        transform_list=[]
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list)

        teacher_transform_list=[]
        teacher_transform_list.append(transforms.ToTensor())
        self.teacher_trans = transforms.Compose(teacher_transform_list)

    def __getitem__(self, index):
        # prepare for source image Xs and target image Xt
        Xs_name, Xt_name = self.name_pairs[index]
        Xs_path = os.path.join(self.image_dir, Xs_name)
        Xt_path = os.path.join(self.image_dir, Xt_name)

        Xs = Image.open(Xs_path).convert('RGB')
        Xt = Image.open(Xt_path).convert('RGB')

        Xs = F.resize(Xs, self.load_size)
        Xt = F.resize(Xt, self.load_size)

        Ps = self.obtain_bone(Xs_name)
        Xs = self.trans(Xs)
        Pt = self.obtain_bone(Xt_name)
        Xt = self.trans(Xt)

        image_dir = '/data/gksruf293/swcd2/results/DPTN_fashion/smallnet_train_image/'
        image_s_Teacher = Image.open(image_dir+Xs_name).convert('RGB')
        image_t_Teacher = Image.open(image_dir+os.path.splitext(Xs_name)[0]+ '_2_' + Xt_name).convert('RGB')

        image_s_Teacher = F.resize(image_s_Teacher, self.load_size)
        image_t_Teacher = F.resize(image_t_Teacher, self.load_size)

        image_s_Teacher = self.teacher_trans(image_s_Teacher)
        image_t_Teacher = self.teacher_trans(image_t_Teacher)

        return {'Xs': Xs, 'Ps': Ps, 'Xt': Xt, 'Pt': Pt,
                'Xs_path': Xs_name, 'Xt_path': Xt_name,
                'Ts': image_s_Teacher, 'Tt': image_t_Teacher}

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        size = len(pairs_file_train)
        pairs = []
        print('Loading data pairs ...')
        for i in range(size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            pairs.append(pair)

        print('Loading data pairs finished ...')
        return pairs

    def getRandomAffineParam(self):
        if self.opt.angle is not False:
            angle = np.random.uniform(low=self.opt.angle[0], high=self.opt.angle[1])
        else:
            angle = 0
        if self.opt.scale is not False:
            scale = np.random.uniform(low=self.opt.scale[0], high=self.opt.scale[1])
        else:
            scale = 1
        if self.opt.shift is not False:
            shift_x = np.random.uniform(low=self.opt.shift[0], high=self.opt.shift[1])
            shift_y = np.random.uniform(low=self.opt.shift[0], high=self.opt.shift[1])
        else:
            shift_x = 0
            shift_y = 0
        return angle, (shift_x, shift_y), scale

    def obtain_bone(self, name):
        string = self.annotation_file.loc[name]
        array = pose_utils.load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
        pose = pose_utils.cords_to_map(array, self.load_size, self.opt.old_size)
        pose = np.transpose(pose,(2, 0, 1))
        pose = torch.Tensor(pose)
        return pose

    def obtain_bone_affine(self, name, affine_matrix):
        string = self.annotation_file.loc[name]
        array = pose_utils.load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
        pose = pose_utils.cords_to_map(array, self.load_size, self.opt.old_size, affine_matrix)
        pose = np.transpose(pose,(2, 0, 1))
        pose = torch.Tensor(pose)
        return pose

    def __len__(self):
        return len(self.name_pairs) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'FashionDistillationDataset'