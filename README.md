# Pose-Guided Image Generation 모델의 간소화

## 프로젝트 개요
이 프로젝트는 Pose-Guided Image Generation 모델을 간소화하는 것을 목표로 하며, 더 낮은 컴퓨팅 자원으로도 효율적인 이미지 생성을 가능하게 하고자 합니다. 이를 통해 모델의 복잡성을 줄이면서도 성능은 유지하는 방법을 탐구하였습니다. 이 프로젝트는 "[Exploring Dual-task Correlation for Pose Guided Person Image Generation](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Exploring_Dual-Task_Correlation_for_Pose_Guided_Person_Image_Generation_CVPR_2022_paper.pdf)“ 논문의 모델을 base로 사용합니다. 

## 모델의 경량화 방법
* 1) 모델의 구조를 일부 수정
     기존 모델의 경우 Encoder block 에서 추출한 특징을  ResBlock 에서 개선하는 방식이 사용됩니다. 여기서 ResBlock 대신, EfficientNet(2019)에서 사용된 MBConvBlock을 사용하면 경량화 과정인 Depthwise separable convolution과 Squeeze-and-excitation을 수행하기 때문에 적은 수의 파라메터로 높은 성능을 유지할 것이라고 판단되었습니다.
* 2) Knowledge distillation 기법을 활용한 Loss 함수를 변경하는 기법
     기존 모델에서 사용되었던 Pose Transformer Modul(이하 PTM)은 각각 2개의 Context Augment Block(CAB)과 Texture Transfer Block(TTB)이 사용되었습니다. 이 부분에서 CAB와 TTB의 수를 1개로 감소시키는 방식을 통해 기존 모델보다 더 작은 모델을 구상할 수 있었고, Knowledge distillation을 적용시키는 Student model로 선정하게 되었습니다, 학습에서는 Teacher model과 Student model의 output을 L1 loss를 사용하여 비교하는 distillation loss를 적용시켜 Teacher model의 지식을 Student model로 이식을 수행하도록 하였습니다.

## 과제 수행 방법
경희대학교의 KHU Seraph 서버와 Colab을 활용하여 cloud 상에서 Dual task Pose Transformer Network 모델에 대한 수정과 학습을 수행합니다. Dataset으로 DeepFashin의 101,966개의 Train data pairs와 8,570개의 Test data pairs를 사용하였고, 각 pairs는 1개 객체에 대한 [ front, back, side, full, additional ] 중 2개의 pose에 대한 이미지로 구성되었습니다. 각 이미지로부터 OpenPose를 사용하여 Pose features를 추출하였고, memory 관리 차원에서 batch size를 기존 32에서 16으로 변화함에 따라 learning rate를 기존 0.0002에서 0.0001로 수정하여 학습을 수행하였습니다. 각 학습은 200epoch씩 수행한 뒤 Test 과정과 분석 과정을 수행하였습니다. Test 과정에서는 각 이미지와 각 pose에 대해 Test pair에 존재하는 그 이미지의 다른 포즈들을 모두 생성하는 과정(SourceImage_to_TargetPose)를 수행하였습니다.

## 수행결과
### 과제수행 결과
기존 모델 (1)의 ResBlock을 MBConvBlock으로 교체한 Teacher 모델 (2)과 (2)의 PTM을 축소한 Student 모델 (3)을 작성하고 학습하였습니다. 각 모델의 비교 분석 결과는 다음과 같습니다.

| 모델           | 파라메터 수 | PSNR↑  | SSIM↑  | FID↓   | LPIPS↓ |
|----------------|------------|-------|-------|-------|-------|
| (1)       | 9.794 M      | 19.1500  | 0.7674  | 11.4575  | 0.1956  |
| (2)        | 8.520 M       | 19.2210  | 0.7700  | 11.4965  | 0.1948  |
| (3) | 7.464 M       | 19.1752  | 0.7660  | 12.0234  | 0.1981  |

### 최종결과물 주요특징 및 설명
위 결과에 따르면, Teacher 모델과 Student 모델의 파라메터 수가 베이스 모델에 비하여 각각 약 1.2M(13%), 2.3M(23%) 감소하였고, PSNR과 같은 성능 측정 결과는 약간 증가하거나 감소하지만 크게 성능이 변화되지는 않았다는 사실을 설명할 수 있습니다. 따라서, 베이스 모델을 충분히 경량화에 성공하였음을 확인할 수 있었습니다.

## 설치 방법
본 섹션에서는 프로젝트를 설치하고 실행하는 단계별 방법을 제공합니다. 필요한 소프트웨어, 라이브러리 및 기타 의존성을 명시합니다. 

### 필요 조건

* Python 3.7.9
* Pytorch 1.7.1
* torchvision 0.8.2
* CUDA 11.1
* NVIDIA A100 40GB PCIe
