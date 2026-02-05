import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
import random
import torch.nn.init as init

# 自定义的动态注意力模块
class DynamicAttentionBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(DynamicAttentionBlock, self).__init__()
        # 使用卷积层来生成注意力权重
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数将权重限制在0到1之间

    def forward(self, x):
        # 通过卷积层计算注意力权重
        attn_map = self.conv(x)  # 获取卷积后的输出作为注意力图
        attn_map = self.sigmoid(attn_map)  # 将注意力值限制在0到1之间

        # 将注意力图应用到特征图
        return x * attn_map  # 将原始特征图和注意力图相乘


# 固定全局随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 初始化模型权重
def initialize_weights(model, seed=42):
    torch.manual_seed(seed)  # 确保权重初始化时的随机种子固定
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 使用Kaiming初始化
            if m.bias is not None:
                init.zeros_(m.bias)  # 将偏置初始化为0
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)  # 使用Xavier初始化
            if m.bias is not None:
                init.zeros_(m.bias)  # 将偏置初始化为0
        elif isinstance(m, nn.BatchNorm2d):
            init.ones_(m.weight)  # BatchNorm中的权重初始化为1
            init.zeros_(m.bias)  # BatchNorm中的偏置初始化为0

# 设置随机种子
seed = 42  # 固定种子值
set_seed(seed)

# 数据转换，将图片转换为Tensor，并进行归一化处理，修改resize大小为448x448
transform = transforms.Compose([
    transforms.Resize((448, 448)),  # 调整图片大小为448x448
    transforms.ToTensor(),  # 将PIL图片转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化，使用ImageNet的均值和标准差
])

# 加载数据集，假设所有图片都在 'data' 文件夹下
data_dir = r'datapath'  # 你的数据所在目录

dataset = datasets.ImageFolder(data_dir, transform=transform)

# 数据划分：80%用于训练，20%用于验证
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# 随机划分数据集，使用相同的种子生成器
generator = torch.Generator().manual_seed(seed)
train_data, val_data = random_split(dataset, [train_size, val_size], generator=generator)

# 使用DataLoader加载数据集，并且设置shuffle=True进行数据的打乱
batch_size = 32

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# 加载非预训练的AlexNet
model = models.alexnet(pretrained=False)

# 修改最后的全连接层 (将输出的类别数改为2)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)  # 2是类别数目

# 添加动态Attention模块
model.features = nn.Sequential(
    model.features,
    DynamicAttentionBlock(in_channels=256, kernel_size=3)  # 假设AlexNet的中间层特征图通道数为256
)

# 使用固定种子初始化权重
initialize_weights(model, seed=seed)

# 将模型移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# 训练模型
num_epochs = 50
best_val_accuracy = 0.0
patience = 5  # 早停的耐心次数
epochs_without_improvement = 0  # 没有改进的epoch计数器

# 保存最优模型和最终模型
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        # 将数据移到GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # 清零优化器的梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 优化器更新参数
        optimizer.step()

        running_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 计算每个epoch的损失和准确率
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total

    # 在验证集上评估模型
    model.eval()  # 设置模型为评估模式
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():  # 在验证时不计算梯度
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = val_correct / val_total
    val_loss = val_loss / len(val_loader)

    # 输出训练集和验证集的损失和准确率
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # 保存最优模型
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'Model_best.pth')  # 保存最优模型
        epochs_without_improvement = 0  # 重置计数器
    else:
        epochs_without_improvement += 1

    # 如果验证准确率在多个epoch内没有改进，则提前停止训练
    if epochs_without_improvement >= patience:
        print("Early stopping triggered. Training stopped.")
        break

# 保存最终模型
torch.save(model.state_dict(), 'Model_final.pth')
print("Model saved to 'Model_final.pth'")