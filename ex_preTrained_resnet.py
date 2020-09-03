### Data download & dataloader 구현
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import (Dataset,
                               DataLoader,
                               TensorDataset)

myPath = "/home/jsh/PycharmProjects/Torch_Exam/data/taco_and_burrito"
train_imgs = ImageFolder(myPath+"/train/",
                         transform=transforms.Compose([
                             transforms.RandomCrop(224),
                             transforms.ToTensor()]
                         ))
test_imgs = ImageFolder(myPath+"/test/",
                         transform=transforms.Compose([
                             transforms.CenterCrop(224),
                             transforms.ToTensor()]
                         ))

### DataLoader 구현
train_loader = DataLoader(train_imgs,batch_size=32,shuffle=True)
test_loader = DataLoader(test_imgs,batch_size=100,shuffle=False)

print(train_imgs.classes)
print(train_imgs.class_to_idx)


### pre-trained된 모델을 가져온다.
import torch
from torch import nn, optim
import tqdm
from torchvision.datasets import FashionMNIST
from torchvision import transforms, models

net = models.resnet18(pretrained=True)

# 모든 파라미터를 미분대상에서 제외
for p in net.parameters():
    p.requires_grad = False

fc_input_dim = net.fc.in_features
net.fc = nn.Linear(fc_input_dim,2) # output은 2밖에 안돼?

### eval, train function 자체 설계
def eval_net(net,test_loader,device="cpu"):
    net.eval()
    ypreds=[] # 매 batch마다 y_pred모아서 나중에 accuracy계산용
    ys=[] # 매 batch마다 y모아서 나중에 acc 계산

    for x,y in test_loader:
        x=x.to(device)
        y=y.to(device)

        with torch.no_grad():
            _, y_pred = net(x).max(1) # 가장 확률 높은 y를 택해서 그걸 output으로 인정한다.
        ypreds.append(y_pred)
        ys.append(y)

    ypreds = torch.cat(ypreds)
    ys = torch.cat(ys)

    acc = (ys==ypreds).float().sum()/len(ys) # 정확도를 계산한다.
    return acc.item()

def train_net(net,train_loader,test_loader,only_fc = True,
              optimizer_cls = optim.Adam, loss_fn=nn.CrossEntropyLoss(), n_iter=10,device="cpu"):
    train_loss=[]
    train_acc=[]
    test_acc=[]

    if only_fc:
        optimizer = optimizer_cls(net.fc.parameters())
    else:
        optimizer = optimizer_cls(net.parameters())

    for epoch in range(n_iter):
        net.train()
        cnt = 0
        model_accuracy = 0.0
        running_loss = 0.0

        for i, (xx,yy) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            xx=xx.to(device)
            yy=yy.to(device)
            print(type(xx))
            h=net(xx)
            loss = loss_fn(h,yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, y_pred = h.max(1)
            cnt += len(xx)
            model_accuracy += (yy==y_pred).float().sum().item()
            running_loss += loss

        train_loss.append(running_loss/i)
        train_acc.append(model_accuracy/cnt)
        test_acc.append(eval_net(net,test_loader,device=device))

        print('epoch : ', epoch, 'train_loss[-1]: ', train_loss[-1],
              'train_acc[-1] : ', train_acc[-1], 'test_acc[-1] : ',test_acc[-1])


### 전체 실행
# net 가져온다
# net_train을 실행한다.
net.to("cuda:0")
# 모델을 학습한다.
train_net(net,train_loader,test_loader,n_iter=20, device="cuda:0")




