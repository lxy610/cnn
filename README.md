生成的onnx文件在 https://netron.app/ 打开

<details>
  <summary>#0 origin 74.5%</summary>
  
```Bash 
  A[Input] --> B[conv1: Conv2d(3, 32, 3, padding=1)]
  B --> C[ReLU]
  C --> D[pool: MaxPool2d(2, 2)]
  D --> E[conv2: Conv2d(32, 64, 3, padding=1)]
  E --> F[ReLU]
  F --> G[pool]
  G --> H[conv3: Conv2d(64, 64, 3, padding=1)]
  H --> I[ReLU]
  I --> J[View(-1, 64*8*8)]
  J --> K[fc1: Linear(64*8*8, 64)]
  K --> L[ReLU]
  L --> M[fc2: Linear(64, 10)]
  M --> N[Output]
```

</details>

<details>
  <summary>#1 三层卷积 78.95%</summary>

  ```bash
  ConvNet
  |
  |-- Input Layer (3 channels)
  |
  |-- conv1: Conv2d(3, 64, 3, padding=1)
  |   |-- BatchNorm2d(64)
  |   |-- ReLU Activation
  |   |-- pool: MaxPool2d(2, 2)
  |
  |-- conv2: Conv2d(64, 128, 3, padding=1)
  |   |-- BatchNorm2d(128)
  |   |-- ReLU Activation
  |   |-- pool: MaxPool2d(2, 2)
  |
  |-- conv3: Conv2d(128, 256, 3, padding=1)
  |   |-- BatchNorm2d(256)
  |   |-- ReLU Activation
  |   |-- pool: MaxPool2d(2, 2)
  |
  |-- Flatten Layer (256 * 4 * 4)
  |
  |-- dropout: Dropout(0.5)
  |
  |-- fc1: Linear(256 * 4 * 4, 512)
  |   |-- ReLU Activation
  |
  |-- fc2: Linear(512, 10)
  |
  |-- Output Layer (10 classes)
```
</details>

<details>
  <summary>#2 加入残差 81.82%</summary>
  
  ```bash
  ConvNet
  |
  |-- Input Layer (3 channels)
  |
  |-- conv1: Conv2d(3, 64, 3, padding=1)
  |   |-- bn1: BatchNorm2d(64)
  |   |-- ReLU Activation
  |
  |-- resblock1: ResidualBlock(64, 64)
  |   |-- conv1: Conv2d(64, 64, 3, stride=1, padding=1)
  |   |-- bn1: BatchNorm2d(64)
  |   |-- ReLU Activation
  |   |-- conv2: Conv2d(64, 64, 3, stride=1, padding=1)
  |   |-- bn2: BatchNorm2d(64)
  |   |-- shortcut: (Conditional)
  |       |-- Conv2d(64, 64, 1, stride=1) (if stride != 1 or in_channels != out_channels)
  |       |-- BatchNorm2d(64)
  |   |-- ReLU Activation (after addition)
  |
  |-- resblock2: ResidualBlock(64, 128, stride=2)
  |   |-- conv1: Conv2d(64, 128, 3, stride=2, padding=1)
  |   |-- bn1: BatchNorm2d(128)
  |   |-- ReLU Activation
  |   |-- conv2: Conv2d(128, 128, 3, stride=1, padding=1)
  |   |-- bn2: BatchNorm2d(128)
  |   |-- shortcut: (Conditional)
  |       |-- Conv2d(64, 128, 1, stride=2) (if stride != 1 or in_channels != out_channels)
  |       |-- BatchNorm2d(128)
  |   |-- ReLU Activation (after addition)
  |
  |-- resblock3: ResidualBlock(128, 256, stride=2)
  |   |-- conv1: Conv2d(128, 256, 3, stride=2, padding=1)
  |   |-- bn1: BatchNorm2d(256)
  |   |-- ReLU Activation
  |   |-- conv2: Conv2d(256, 256, 3, stride=1, padding=1)
  |   |-- bn2: BatchNorm2d(256)
  |   |-- shortcut: (Conditional)
  |       |-- Conv2d(128, 256, 1, stride=2) (if stride != 1 or in_channels != out_channels)
  |       |-- BatchNorm2d(256)
  |   |-- ReLU Activation (after addition)
  |
  |-- pool: MaxPool2d(2, 2)
  |
  |-- Flatten Layer (256 * 4 * 4)
  |
  |-- dropout: Dropout(0.5)
  |
  |-- fc1: Linear(256 * 4 * 4, 512)
  |   |-- ReLU Activation
  |
  |-- fc2: Linear(512, 10)
  |
  |-- Output Layer (10 classes)
```
</details>

<details>
  <summary>#3 relu换gelu 82.17%</summary>
  null
</details>
