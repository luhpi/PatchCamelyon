import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torch.utils import data
import pandas as pd
import numpy as np
from PIL import Image


class MyDataset(data.Dataset):
    '''Classe responsável por carregar os dados do computador e alimentar
    o algoritmo que irá realizar o treinamento da rede neural através
    de batches.'''

    def __init__(self, labels_csv, image_folder):

        '''Função responsável por inicializar a classe do dataset, aqui serão
        configurados os diretórios e arquivos necessários para carregar os
        dados utilizados no treinamento.

        :param labels_csv: endereço para arquivo csv com informações sobre
        os dados
        :param image_folder: pasta contendo as imagens que serão carregadas
        '''

        raw_labels = pd.read_csv(labels_csv)

        self.image_folder = image_folder
        self.labels = np.asarray(raw_labels.iloc[:, 2:]).argmax(axis=1)
        self.filenames = np.asarray(raw_labels.iloc[:, 1] + '.jpg')

        self.count = len(self.filenames)

        # Variável que armazena os diferentes tipos de transformação que
        # iremos aplicar nas nossas imagens de entrada antes de alimentar
        # a rede neural.
        # Repare que aqui iremos realizar um recorte na área central da
        # imagem.

        self.transformations = transforms.Compose([transforms.CenterCrop(32),
                                                   transforms.ToTensor()])

    def __len__(self):

        '''Função que retorna a quantidade de imagens existente no conjunto
        de dados.'''

        return self.count

    def __getitem__(self, index):
        '''Função que carrega uma imagem do disco e sua classe verdadeira
        utilizando como base os dados informados na inicialização da classe.

        :param index: índice que define os dados que serão carregados'''

        single_image_path = self.image_folder + '/' + self.filenames[index]

        image_data = Image.open(single_image_path).resize((64, 64))
        image_data = self.transformations(image_data)

        image_label = self.labels[index]

        return image_data, image_label


'''Setting the labels and image folder location'''

labels_csv = '~/projects/PatchCamelyon/labels.csv'
image_folder = '~/projects/PatchCamelyon/imagens'

dataset = MyDataset(labels_csv, image_folder)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,  7)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = LeNet()

EPOCH = 50
BATCH_SIZE = 32
cudaopt = True
torch.cuda.set_device(0)


# training parameters
LR = 0.001
WD = 0
MN = .9
step_size_schd = 15
gamma_schd = .2

model = LeNet()
model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MN,
                            weight_decay=WD)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=step_size_schd,
                                            gamma=gamma_schd)


train_loader = DataLoader(dataset=dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False)

Loss_train = np.zeros((EPOCH,))
Acc_train = np.zeros((EPOCH,))

for epoch in range(EPOCH):

    scheduler.step()

    # train 1 epoch
    model.train()
    correct = 0
    train_loss = 0
    rec_error = 0

    for step, (x, y) in enumerate(train_loader):

        b_x = torch.autograd.Variable(x)  # batch x,
        b_y = torch.autograd.Variable(y)  # batch label

    if cudaopt:
        b_y, b_x = b_y.cuda(), b_x.cuda()

        optimizer.zero_grad()  # clear gradients for this training step
        scores = model(b_x)
        loss = F.cross_entropy(scores, b_y)

        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        # computing training accuracy

        pred = scores.data.max(1, keepdim=True)[1]
        correct += pred.eq(b_y.data.view_as(pred)).long().cpu().sum()
        train_loss += F.cross_entropy(scores, b_y, reduction='sum').data.item()

    Acc_train[epoch] = 100 * float(correct) / float(len(train_loader.dataset))
    Loss_train[epoch] = train_loss / len(train_loader.dataset)

    print('Epoch: ', epoch,
          '| train loss: ', round(Loss_train[epoch], 3),
          '| train acc: ',  round(Acc_train[epoch],  3), '%')
