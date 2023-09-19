import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse
import glob
import torch.utils.data
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import time
from torchvision.models import resnet18, ResNet18_Weights, alexnet, AlexNet_Weights, regnet_y_400mf
from torch.optim.lr_scheduler import *
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import timm
from torchinfo import summary
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class SingleImgDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=20):
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*.png'))
            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        im = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            im = self.transform(im)
        return (class_id, im, path)


class MultiviewImgDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=20, shuffle=True):
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*.png'))
            stride = int(20/self.num_views) # 12 6 4 3 2 1
            all_files = all_files[::stride]

            if num_models == 0:
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        if shuffle==True:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths)/num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i]*num_views:(rand_idx[i]+1)*num_views])
            self.filepaths = filepaths_new

        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return int(len(self.filepaths)/self.num_views)

    def __getitem__(self, idx):
        path = self.filepaths[idx*self.num_views]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        return (class_id, torch.stack(imgs), self.filepaths[idx*self.num_views:(idx+1)*self.num_views])


   
class Stage1AlexNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.regnety = timm.create_model('regnety_004.tv2_in1k', pretrained=True, num_classes=40)
        self.fc2 = nn.Linear(1000, 40).cuda()
    def forward(self, x):
        return self.regnety(x)


class SE(nn.Module):
    def __init__(self, input_dim: int = 1024):
        super(SE, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, input_dim//4),
            nn.BatchNorm1d(input_dim//4),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(input_dim//4, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Sigmoid()
        )
        self.Excitation = nn.Sequential(
            self.fc1,
            self.fc2
        )
    
    def forward(self, x):
        return torch.mul(x, self.Excitation(x))


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.se = SE(input_dim=input_dim).cuda()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        ).cuda()
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        ).cuda()
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        ).cuda()
        self.fc4 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        ).cuda()
        #self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.se(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x) + x
        x = self.fc4(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int):
        super(AttentionBlock, self).__init__()
        self.attention  = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.5).cuda()
        self.ln1 = nn.LayerNorm(embed_dim).cuda()
        self.ln2 = nn.LayerNorm(embed_dim).cuda()
        self.dropout = nn.Dropout(0.5)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*mlp_ratio),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim*mlp_ratio, embed_dim)
        ).cuda()
        
    def forward(self, x):
        Z = self.ln1(x)
        Zhat, _ = self.attention(Z, Z, Z)
        Zhat = Zhat + Z
        Zhat2 = self.ln2(Zhat)
        Zhat2 = self.mlp(Zhat2)
        Zhat2 = self.dropout(Zhat2) + Zhat
        return Zhat2

class Encoder(nn.Module):
    def __init__(self, num_blocks, num_heads, embed_dim, mlp_ratio):
        super(Encoder, self).__init__()
        # self.blocks = [AttentionBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for i in range(num_blocks)]
        self.block1 = AttentionBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.block2 = AttentionBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
    def forward(self, x):
        return self.block2(self.block1(x))


class AlirezaNet(nn.Module):
    def __init__(self, stage1: Stage1AlexNet, num_views=20, nclasses=40):
        super().__init__()
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        self.nclasses = nclasses
        self.mean = torch.tensor([0.485, 0.456, 0.406],dtype=torch.float, requires_grad=False)
        self.std = torch.tensor([0.229, 0.224, 0.225],dtype=torch.float, requires_grad=False)
        self.init_net = stage1
        self.init_net.regnety.reset_classifier(0)
        self.attention_pool = nn.Linear(368, 1).cuda()
        self.encoder = Encoder(num_blocks=4, num_heads=8, embed_dim=368, mlp_ratio=2).cuda()
        self.decoder = Decoder(368, 2048, 40).cuda()

    def forward(self, x):
        Z0 = self.init_net(x).squeeze().cuda()
        Z0 = Z0.view((int(Z0.shape[0] / num_views), num_views, -1))
        Z1 = self.encoder(Z0)
        Z2 = torch.matmul(F.softmax(self.attention_pool(Z1), dim=1).transpose(-1, -2), Z1).squeeze(-2)
        Y = self.decoder(Z2)
        return Y
 

class Stage1():
    def __init__(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.net = Stage1AlexNet().cuda()
        weight_decay = 0.00002
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, weight_decay=weight_decay, momentum=0.9)
        self.scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=17, eta_min=0.00001)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, num_epochs=30):
        best_acc_epoch = 0
        best_acc = 0
        for epoch in range(num_epochs): 
            running_loss = 0.0
            epoch_start_time = time.time()
            iters = len(train_loader)
            for i, data in enumerate(train_loader, 0):
                start_time = time.time()
                in_data = Variable(data[1]).cuda()
                target = Variable(data[0]).long().cuda()
                self.optimizer.zero_grad()
                outputs = self.net(in_data).cuda()
                loss = self.criterion(outputs, target)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 100 == 0:
                    progress_percent = int(((i * batch_size) / (197000))*100)
                    end_time = time.time()
                    batch_time = (end_time - start_time) * 100
                    print(f'S1: [{epoch + 1}, {i + 1:4d}][{progress_percent}%][{batch_time:.2f}s] loss: {running_loss / 100:.5f}')
                    running_loss = 0.0
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            print(f"Learning rates: ", self.scheduler.get_last_lr())
            self.scheduler.step()
            print(f"Epoch took {epoch_time:.3f} seconds")
            with torch.no_grad():
                print('testing ...')
                self.net.eval()
                all_correct_points = 0
                all_points = 0
                count = 0
                wrong_class = np.zeros(40)
                samples_class = np.zeros(40)
                all_loss = 0
                for _, data in enumerate(val_loader):
                    in_data = Variable(data[1]).cuda()
                    target = Variable(data[0]).cuda()
                    out_data = self.net(in_data)
                    pred = torch.max(out_data, 1)[1]
                    all_loss += self.criterion(out_data, target).cpu().data.numpy()
                    results = pred == target
                    for i in range(results.size()[0]):
                        if not bool(results[i].cpu().data.numpy()):
                            wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                        samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
                    correct_points = torch.sum(results.long())

                    all_correct_points += correct_points
                    all_points += results.size()[0]
                print('Total # of test models: ', all_points)
                class_acc = (samples_class - wrong_class) / samples_class
                val_mean_class_acc = np.mean(class_acc)
                acc = all_correct_points.float() / all_points
                val_overall_acc = acc.cpu().data.numpy()
                loss = all_loss / len(val_loader)

                print('val mean class acc. : ', val_mean_class_acc)
                print('val overall acc. : ', val_overall_acc)
                print('val loss : ', loss)
                # print(class_acc)
                self.net.train()
                if val_overall_acc > best_acc:
                    best_acc = val_overall_acc
                    best_acc_epoch = epoch
                print('best acc: ', best_acc, ' at ', best_acc_epoch)
            

if __name__ == '__main__':
    seed_torch()
    colorama_init()
    num_views = 20
    num_models = 0
    num_epochs_stage_1 = 0
    num_epochs_stage_2 = 200
    n_models_train = num_models*num_views
    train_path = 'data/modelnet40v2png_ori4/*/train'
    val_path = 'data/modelnet40v2png_ori4/*/test'
    weight_decay = 0.00002
    lr = 1e-3
    bs = 5
    # 1060
    #batch_size = 480
    #mv_batchsize = 30
    # 3080
    batch_size = 100
    mv_batchsize = 16
    print('Stage 1 loading data ...')
    train_dataset = SingleImgDataset(train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataset = SingleImgDataset(val_path, scale_aug=False, rot_aug=False, test_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    stage1 = Stage1(train_loader=train_loader, test_loader=val_loader)


    USE_PRE_TRAINED_STAGE1 = True
    if not USE_PRE_TRAINED_STAGE1:
        print('training stage1 from scratch...')
        # print model summary before train
        stage1.train(num_epochs_stage_1)
        print('model summary after train')
        #summary_input_data = torch.randn(1, 3, 224, 224).cuda()
        #summary(stage1.net, input_data=summary_input_data, depth=4)
        #del summary_input_data
        code_version = os.path.basename(__file__).split('.')[0]
        stage1_file_name = f"stage1_{code_version}.pth"
        print(f'saving stage1 to {stage1_file_name}')
        torch.save(stage1, stage1_file_name)
    else:
        code_version = os.path.basename(__file__).split('.')[0]
        stage1_file_name = f"stage1_{code_version}.pth"
        print(f'loading pre-trained stage1 from {stage1_file_name}')
        stage1 = torch.load('stage1_mv13_se_plus_stage1_mf200_93_12.pth')

    train_dataset = MultiviewImgDataset(train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=num_views,test_mode=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=mv_batchsize, shuffle=False, num_workers=4)
    val_dataset = MultiviewImgDataset(val_path, scale_aug=False, rot_aug=False, num_views=num_views,test_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=mv_batchsize, shuffle=False, num_workers=4)
    print('data load done')
    net = AlirezaNet(stage1=stage1.net).cuda()
    #optimizer = optim.SGD(net.parameters(), lr=1e-2, weight_decay=weight_decay, momentum=0.9)
    optimizer = optim.AdamW(net.parameters(), lr=0.5*1e-3)
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100)
    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=60, cycle_mult=1.0, max_lr=0.5*1e-4, min_lr=0.000001, warmup_steps=7, gamma=0.5)
    #optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    #print('alirezanet ', AlirezaNet)

    print('summary stage2:' )
    summary_input_data = torch.randn(20, 3, 224, 224).cuda()
    summary(net, input_data=summary_input_data, depth=5)

    print('depth 2')
    summary(net, input_data=summary_input_data, depth=2)
    del summary_input_data

    best_acc_epoch = 0
    best_acc = 0
    for epoch in range(num_epochs_stage_2): 
        running_loss = 0.0
        running_loss_count = 0
        epoch_start_time = time.time()
        iters = len(train_loader)
        #print("iters : ", iters)
        for i, data in enumerate(train_loader):
            start_time = time.time()
            #print('X ', data[1].shape)
            N, V, C, H, W = data[1].size()
            #print('shape' , data[1].shape)
            in_data = Variable(data[1]).view(-1, C, H, W).cuda()
            #print('shape ', in_data.shape)
            target = Variable(data[0]).long().cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(in_data).cuda()
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_loss_count += 1
            if i % 100 == 0:
                progress_percent = int(((i * batch_size) / (197000))*100)
                end_time = time.time()
                batch_time = (end_time - start_time) * 100
                #print(f'S2[{epoch + 1}, {i + 1:4d}][{progress_percent}%][{batch_time:.2f}s] loss: {running_loss / 100:.6f}')
                #running_loss = 0.0
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(f'S2[{epoch}] train-loss: {running_loss/running_loss_count:.6f}, time: {epoch_time:.3f}s, lr: {scheduler.get_lr()[0]:.9f}')
        scheduler.step(epoch)
        with torch.no_grad():
            #print('testing ...')
            net.eval()
            test_start_time = time.time()
            all_correct_points = 0
            all_points = 0
            count = 0
            wrong_class = np.zeros(40)
            samples_class = np.zeros(40)
            all_loss = 0
            for _, data in enumerate(val_loader):
                N, V, C, H, W = data[1].size()
                in_data = Variable(data[1]).view(-1, C, H, W).cuda()
                target = Variable(data[0]).cuda()
                out_data = net(in_data).cuda()
                pred = torch.max(out_data, 1)[1]
                all_loss += criterion(out_data, target).cpu().data.numpy()
                results = pred == target
                for i in range(results.size()[0]):
                    if not bool(results[i].cpu().data.numpy()):
                        wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                    samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
                correct_points = torch.sum(results.long())

                all_correct_points += correct_points
                all_points += results.size()[0]
            test_end_time = time.time() - test_start_time
            class_acc = (samples_class - wrong_class) / samples_class
            val_mean_class_acc = np.mean(class_acc)
            acc = all_correct_points.float() / all_points
            val_overall_acc = acc.cpu().data.numpy()
            loss = all_loss / len(val_loader)
            print(f'S2-test[{epoch}] loss: {loss:.6f}, mcacc: {val_mean_class_acc*100:.2f}%, acc: {Fore.GREEN}{val_overall_acc*100:.2f}%{Style.RESET_ALL}, time: {int(test_end_time)}s')
            net.train()
            if val_overall_acc > best_acc:
                best_acc = val_overall_acc
                best_acc_epoch = epoch
                code_version = os.path.basename(__file__).split('.')[0]
                best_acc_str = str(round(best_acc*100,3)).replace('.','-')
                save_best_acc_path = f"acc_stage2_{code_version}_e{best_acc_epoch}_{best_acc_str}%.pth"
                print(f'-> found new best acc, saving to {save_best_acc_path}')
                torch.save(net, save_best_acc_path)
                print(f'-> save done')
            print(f'best acc: {Fore.YELLOW}{round(best_acc*100,2)}%{Style.RESET_ALL} at {best_acc_epoch}')
        

    PATH = 'net.pth'
    torch.save(net.state_dict(), PATH)
