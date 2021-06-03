# -*- coding: utf-8 -*-
import argparse

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel


# tokenizer 토큰 정보 설정
U_TKN = '<usr>' # Q token
S_TKN = '<sys>' # A token
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

# KoGPT2 tokrnizer 설정
TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 


class KoGPT2Chat(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.hparams = hparams
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output = self.kogpt2(inputs, return_dict=True)
        return output.logits

    def chat(self, question, sent='0'):
        tok = TOKENIZER
        sent_tokens = tok.tokenize(sent) # 중립 감정 '0' 를 토큰화
        with torch.no_grad():
            # 질문 입력
            q = question.strip()

            # 답변 a 설정
            a = ''

            # 답변 생성 과정 기존 while 문에서 for 문으로 변경
            for i in range(200):
                # tok.encode(U_TKN + q + SENT + sent + S_TKN + a) = f'<usr>{질문}<unused1>0<sys>{생성된 답변}'
                input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                pred = self(input_ids) # KoGPT2 forward 진행 pred 에는 모든 토큰에 대한 가능성

                # 11개의 각 최고 값을 추출 후 마지막 하나만을 선택
                gen = tok.convert_ids_to_tokens(
                    torch.argmax(
                        pred,
                        dim=-1).squeeze().numpy().tolist())[-1]

                # 종료 단어일 경우 종료
                if gen == EOS:
                    break

                # 종료가 아닐경우 답변 추가 _ 일경우 띄어쓰기로 변경
                a += gen.replace('▁', ' ')

            return a.strip()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')
    # 감정 상태 확인
    parser.add_argument('--sentiment',
                        type=str,
                        default='0',
                        help='sentiment for system. 0 is neutral, 1 is negative, 2 is positive.')

    # 모델 weight 위치
    parser.add_argument('--model_params',
                        type=str,
                        default='model_chp/model_-last.ckpt',
                        help='model binary for starting chat')
    args = parser.parse_args()

    model = KoGPT2Chat.load_from_checkpoint(args.model_params)
    model.chat()
