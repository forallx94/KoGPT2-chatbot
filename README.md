# Simple Chit-Chat based on KoGPT2

## Purpose

- [공개된 한글 챗봇 데이터](https://github.com/songys/Chatbot_data)와 pre-trained [KoGPT2](https://github.com/SKT-AI/KoGPT2)를 이용한 간단한 챗봇 실험
- `KoGPT2`의 다양한 활용 가능성을 타진하고 성능을 정성적으로 평가한다.

## Fork Purpose

- 해당 코드 이해 및 추가적인 Fine tune 을 위해 train_torch.py 에 대한 주석 작성
- 기존의 input 입력 방식에서 하나의 함수로 작동하도록 설정 변경 (test_torch.py)
- [Merged Data for Korean Emotion Classification](https://github.com/forallx94/EC_data)을 감정 섞인 질문으로 가정하여 답변 생성

## Architecture

학습 데이터에 적합하게 [Hello! GPT-2](https://arxiv.org/abs/1907.05774)를 응용, 아래와 같은 아키텍처를 설계함

<img src="imgs/simsimi.png" alt="drawing" style="width:600px;"/>


- 데이터의 Q 필드를 `<usr>`발화, A 필드를 `<sys>`발화 그리고 감정 레이블을 `<sent>`로 매핑해 `P(<sys>|<usr>, <sent>)`를 최대화 할 수 있는 모델을 학습한다. 감정 레이블은 [이곳](https://github.com/songys/Chatbot_data)의 정의를 따른다(일상다반사 0, 이별(부정) 1, 사랑(긍정) 2).

## Install


```bash
git clone --recurse-submodules https://github.com/haven-jeon/KoGPT2-chatbot.git
cd KoGPT2-chatbot
pip3 install -r requirements.txt 
```

## How to Train

```bash
CUDA_VISIBLE_DEVICES=0 python train_torch.py --gpus 1 --train --max_epochs 2
```

## How to Chat!

```bash
# CUDA_VISIBLE_DEVICES=0 python train.py --chat
CUDA_VISIBLE_DEVICES=0 python train_torch.py --gpus 1 --chat
user > 대박 
Simsimi > 잘 될 거예요.

user > 너 이름이 뭐니?
Simsimi > 저는 위로봇입니다.

user > 어버이날엔 뭘 하죠?
Simsimi > 저랑 놀아요.

user > 비가와
Simsimi > 우산 챙기세요.

user > 눈이와
Simsimi > 눈사람 만들어보세요.

user > 천둥이 쳐
Simsimi > 저도 보고 싶어요.

user > 봄이왔어.
Simsimi > 봄은 항상 두근거려요.

user > 배탈났어.
Simsimi > 얼른 집에 가서 쉬세요.

user > 너를 만든 사람은?
Simsimi > 당신이 만든 사람이에요.
```

## How to Chat! ver module


```python
# 모듈로 로드 하여 진행하는 과정
from test_torch import KoGPT2Chat
model = KoGPT2Chat.load_from_checkpoint('model_chp/model_-last.ckpt')
question = '내가 그의 이름을 불러 주었을 때'
answer = model.chat(question)
print(answer)

'''그게 당신의 운이었을 거예요.'''
```

## How to Chat! ver make excel result

```bash
python chatboy.py
```

## Result

![결과](/imgs/result.png)

전반적으로 괜찮은 모습을 보여주나 간혹 `<pad>`, 진짜 등 마무리 되지 못하고 계속 반복되는 문제 발생. KoGPT2가 `<pad>`, 진짜 와 같은 단어 다음에 나올 단어를 동일한 단어로 훈련하여 이를 계속 반복하여 생긴 문제로 예상.

## Colab tutorials

- PyTorch
  - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/forallx94/KoGPT2-chatbot/blob/master/KoGPT2_chatbot_pytorch.ipynb)

