import numpy as np
import csv
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import time_frequence as tf


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.gan_loss = opt.gan_loss
        self.isTrain = opt.isTrain

        #set table and csv file to write
        self.table = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']
        self.sub_name = os.path.join(opt.checkpoints_dir, opt.name, 'submission.csv')
        with open(self.sub_name, 'w') as f:
            header = ['fname', 'label']
            writer = csv.writer(f)
            writer.writerow(header)

        # define tensors self.Tensor has been reloaded
        self.inputAudio = self.Tensor(opt.batchSize, opt.len).cuda(device=self.gpu_ids[0])
        self.inputLabel = torch.LongTensor(opt.batchSize, opt.nClasses).cuda(device=self.gpu_ids[0])
        # load/define networks
        self.netG = networks.define_G(opt.nClasses, self.gpu_ids)


        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
            #                              opt.which_model_netD,
            #                              opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            # if self.isTrain:
            #    self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            # self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            # self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterion = torch.nn.CrossEntropyLoss()

            # initialize optimizers

            self.TrainableParam = list()
            param = self.netG.named_parameters()
            IgnoredParam = [id(P) for name, P in param if 'stft' in name]

            if self.opt.optimizer == 'Adam':
                self.optimizer_G = torch.optim.Adam(
                    filter(lambda P: id(P) not in IgnoredParam,
                       self.netG.parameters()),
                        lr=opt.lr,
                        betas=(opt.beta1, 0.999))

            if self.opt.optimizer == 'sgd':
                self.optimizer_G = torch.optim.SGD(
                    filter(lambda P: id(P) not in IgnoredParam,
                       self.netG.parameters()),
                        lr=opt.lr)




            print('---------- Networks initialized ---------------')
            networks.print_network(self.netG)
            # networks.print_network(self.netD)
            print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        inputAudio = input['Audio']
        inputLabel = input['Label']
        self.inputFname = input['Fname']

        self.inputAudio.resize_(inputAudio.size()).copy_(inputAudio)
        self.inputLabel.resize_(inputLabel.size()).copy_(inputLabel)

        self.image_paths = 'NOTIMPLEMENT'

    def forward(self):
        self.input = Variable(self.inputAudio)
        output = self.netG.forward(self.input)

        #import pdb; pdb.set_trace()
        self.predLogits = output['logits']


    # no backprop gradients
    def test(self):
        self.netG.eval()
        self.forward()
        # self.input = Variable(self.inputAudio, volatile=True)
        # self.predLogits = self.netG.forward(self.input)['logits']
        #self.netG.train()

        prediction = np.argmax(self.predLogits.cpu().data.numpy(), axis=1).astype(int)
        print(np.sum(self.inputLabel.cpu().numpy() == prediction) / max(prediction.shape))
        predictLabel = [self.table[i] for i in prediction]
        # import pdb; pdb.set_trace()

        with open(self.sub_name, 'a') as f:
            message = [m for m in zip(self.inputFname, predictLabel)]
            writer = csv.writer(f)
            for row in message:
                writer.writerow(row)
                # print(row)
        f.close()
        self.netG.train()

        return prediction


    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):
        Label = Variable(self.inputLabel, requires_grad=False)
        self.loss_G = self.criterion(self.predLogits, Label)

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        if self.gan_loss:
            return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                                ('G_L1', self.loss_G_L1.data[0]),
                                ('D_real', self.loss_D_real.data[0]),
                                ('D_fake', self.loss_D_fake.data[0]),
                                ('Logits', self.predLogits)
                                ])
        else:
            # print("#############clean sample mean#########")
            # sample_data = self.input_B.cpu().numpy()

            # print("max value", np.max(sample_data))
            # print("mean value", np.mean(np.abs(sample_data)))
            return OrderedDict([('Logits', self.predLogits.data.cpu().numpy()),
                                ('G_LOSS', self.loss_G.data.cpu().numpy())])
            # return self.loss_G.data[0]

    def get_current_visuals(self):
        real_A = self.real_A.data.cpu().numpy()
        fakeB = self.fakeB.data.cpu().numpy()
        realB = self.realB.data.cpu().numpy()
        clean = self.clean.cpu().numpy()
        noise = self.noise.cpu().numpy()
        return OrderedDict([
            ('est_ratio', fakeB),
            ('clean', clean),
            ('ratio', realB),
            ('noise', noise),
        ])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        if self.gan_loss:
            self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        # lrd = self.opt.lr / self.opt.niter_decay
        # lr = self.old_lr - lrd
        lr = self.old_lr * 0.6
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
