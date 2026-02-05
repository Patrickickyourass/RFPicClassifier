import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
import os
import time  # ✅ 新增：导入时间模块

# ✅ 开始计时
start_time = time.time()

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载非预训练的AlexNet并修改最后一层
model = models.alexnet(pretrained=False)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)

# 添加动态Attention模块
class DynamicAttentionBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(DynamicAttentionBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        attn_map = self.conv(x)
        attn_map = self.sigmoid(attn_map)
        return x * attn_map

model.features = torch.nn.Sequential(
    model.features,
    DynamicAttentionBlock(in_channels=256, kernel_size=3)
)

# 加载模型参数
model_path = 'Model_best.pth'
model.load_state_dict(torch.load(model_path))

# 将模型移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# 定义输入文件夹路径
input_folder = r'C:\Users\UserName\Desktop\RF_PICs'
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 预测结果列表
results = []

# 对每张图片进行预测
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = datasets.folder.default_loader(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()

    results.append((image_file, predicted_class))

# 保存结果到文本文件
output_file = r'C:\Users\UserName\Desktop\Classify_result.txt'
with open(output_file, 'w') as f:
    for image_file, predicted_class in results:
        f.write(f'{image_file}\t{predicted_class}\n')

# ✅ 结束计时并打印耗时
end_time = time.time()
elapsed_time = end_time - start_time
print(f"预测结果已保存到 {output_file}")
print(f"总耗时: {elapsed_time:.2f} 秒")
