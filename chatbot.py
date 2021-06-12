import sys
import os

from test_torch import KoGPT2Chat

import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    # import 를 위한 경로 설정

    model = KoGPT2Chat.load_from_checkpoint('./model_chp/model_-last.ckpt')
    question = '내가 그의 이름을 불러 주었을 때'
    answer = model.chat(question)
    print(f'Q : {question} | A : {answer}')

    df = pd.read_excel('./EC_data/감정_분류를_위한_대화_음성_데이터셋_4-5.xlsx',index_col=0)
    df = df.dropna()

    tqdm.pandas()

    Emotion_dict = {
        'Neutral':'0',
        'Surprise':'0',
        'Angry':'1',
        'Sadness':'1',
        'Fear':'1',
        'Disgust':'1',
        'Happiness':'2',
    }

    df['Answer'] = df.progress_apply(lambda x : model.chat(x[0],Emotion_dict[x[1]]), axis=1)

    df.to_excel('./EC_data/df4_5_excel.xlsx')