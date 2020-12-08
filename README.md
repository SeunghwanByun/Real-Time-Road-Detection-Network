# Real-Time-Road-Detection-Network
##### Real-time road area detection algorithm using a super-resolution network. This network focus on on-road features like boundary, coordinate in image tensors.

이 레포지토리는 제안한 [논문](https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE10492637)의 결과를 담고있다.

본 연구는 LiDAR와 Camera 센서를 융합하여 생성한 입력 텐서를 사용하여 Super Resolution 연구에 사용되었던 여러 네트워크들을 참조하여 고안한 네트워크를 제안하였다.
기존 SOTA와 비교하였을 때 성능이 비교적 낮지만 실시간 검출을 보여주고 있어 실제 자율 주행 자동차에 사용되기에 적합한 결과이다. 전체 알고리즘은 PyTorch로 작성되었다.

### Network Architecture
Typical Encoder-Decoder 사이에 Boundary Reinforcement 모듈을 추가하여 학습되는 feature들이 도로의 경계면을 정밀하게 학습하도록 고안하였다.
<p align="center">
<img src="https://user-images.githubusercontent.com/49049277/101449294-d99a8b00-396b-11eb-89c0-18ded34775f4.jpg" width="90%">
</p>


### Sensor Fusion Input Tensor
LiDAR와 Camera를 구면좌표계에 투영하여 융합한 입력 데이터를 생성하였다.
<p align="center">
<img src="https://user-images.githubusercontent.com/49049277/101480050-7a05a500-3996-11eb-9e98-015d9a97fd94.jpg" width="40%">
</p>


### Result of KITTI road benchmark Test sets
> 위에서부터 UM, UMM, UU 도로 타입에 대한 결과 이미지
<p align="center">
<img src="https://user-images.githubusercontent.com/49049277/101450146-68f46e00-396d-11eb-8c01-a0110a948df1.jpg" width="60%">
</p>

> Bird's Eye View로 변환한 결과 이미지
<p align="center">
<img src="https://user-images.githubusercontent.com/49049277/101450056-3c405680-396d-11eb-90cd-fa53dce857dd.jpg" width="60%">
</p>


### Table comparing with SOTA
<p align="center">
<img src="https://user-images.githubusercontent.com/49049277/101456757-d907f180-3977-11eb-9503-d248c15f6e90.jpg" width="60%">
</p>

### Setup
```
main.py
  |---- image_2 
  |---- calib
  |---- velodyne
  |---- gt_txt
```
### Train
```
python main.py
```
