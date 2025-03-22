import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为 Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 使用 MNIST 数据集的均值和标准差进行标准化
])
train_dataset = datasets.MNIST(
    root=r"D:\srt\workspace\pytorch_learning",
    train = True,
    download = True,
    transform = transform 
)
test_dataset = datasets.MNIST(
    root=r"D:\srt\workspace\pytorch_learning",
    train = False,
    download = True,
    transform = transform 
)  
batchsize = 64
train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle = False)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(10816, 128)  # 全连接层输入尺寸需要根据特征图计算
        self.fc2 = nn.Linear(128, 10)
    def forward(self,x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
def train(model, device, train_dataloader,optimizer,epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataloader.dataset)}'
                  f' ({100. * batch_idx / len(train_dataloader):.0f}%)]\tLoss: {loss.item():.6f}')
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({accuracy:.2f}%)\n')
    return accuracy
epochs=10
best_accuracy = 0.0
for epoch in range(1, epochs + 1):
    train(model, device, train_dataloader, optimizer, epoch)
    current_accuracy = test(model, device, test_loader)
    
    # 保存最佳模型
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        torch.save(model.state_dict(), "mnist_cnn_best.pth")

print(f"Best accuracy: {best_accuracy:.2f}%")