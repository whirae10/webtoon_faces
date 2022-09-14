from pickle import GET
from flask import Flask, render_template, request, url_for, redirect, after_this_request, g, session
from model import Encoder, Generator
import cv2
from werkzeug.utils import secure_filename
import os, glob
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
from cachelib import SimpleCache
import face2webtoon as fw

cache = SimpleCache()

# 이거는 코사인 유사도를 위한 모듈
module = hub.load("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2")

# app 선언
app = Flask(__name__)

# 웹툰화 불러오기
'''
model=pickle.load(open(''.'rb')

'''


# 메인 템플렛 불러오기
@app.route('/')
def home():
    return render_template('main.html')


# 파일 업로드, 리사이징, 원본이미지 삭제
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        # 저장할 경로 + 파일명
        f.save('./uploads/' + secure_filename(f.filename))
        upload_dir = r'D:\cp1\cp1_webtoonize\uploads'  # 업로드된 파일 디렉토리
        resize_dir = r'/cp1_webtoonize/static/ml_img'  # 리사이징 된 파일 디렉토리
        files = os.listdir(upload_dir)
        for file in files:
            path = os.path.join(upload_dir, file)
            new_image = Image.open(path).resize((256, 256))
            new_image.save(f"{'./static/ml_img'}/001.png")  # 파일 저장
            files = os.listdir(upload_dir)
            for file in files:
                file_path = os.path.join(upload_dir, file)
                os.remove(file_path)

    return redirect('/results')


# 이미지 웹툰화 및 가장 '비슷한' 캐릭터가 존재하는 웹툰 추천 html 결과창
@app.route('/results', methods=['GET', 'POST'])
def total_process():
    fw.run(r'D:\cp1\cp1_webtoonize\static\ml_img\001.png')
    # cos_sim
    '''
    tmp_df 피클 파일 열람후 전처리된 이미지를 백터화
    
    output array 에 저장후 백터화된 검증 데이터 이미지 들 과 cosine similarity 비교 
    
    이후 상위 3 가지의 웹툰 이미지 및 유사도 추출
    '''

    tmp_df = pd.read_pickle('tmp_dfs.pkl')
    path = r'./static/ml_img/002.png'  # 웹툰화된 이미지 경로
    image = Image.open(path).convert('RGB')
    img_input = tf.keras.preprocessing.image.img_to_array(image)
    img_input = np.array([img_input])
    output = np.array(module(img_input))
    cos_sim_array = []

    for i in range(len(tmp_df['output'])):
        cos_sim_array.append(cosine_similarity(output, tmp_df['output'][i]))

    tmp_df['cos_sim'] = cos_sim_array
    tmp_df = tmp_df.sort_values(by='cos_sim', ascending=False)

    top3list = tmp_df[:3].reset_index(drop=True)
    top3 = tmp_df[:3]['cos_sim'].reset_index(drop=True)

    first = round(top3[0][0][0] * 100, 2)  # 1순위 코사인 유사도
    second = round(top3[1][0][0] * 100, 2)  # 2순위 코사인 유사도
    third = round(top3[2][0][0] * 100, 2)  # 3순위 코사인 유사도

    # 웹 크롤링한 메타데이터 정리
    meta_df = pd.read_csv("Verif_metadata.csv", index_col=0)
    meta_df['Story'] = meta_df['Story'].replace(r'\n', ' ', regex=True)

    # 가장 유사한 3웹툰 메타데이터 추출 
    tmp = []
    tmps = []
    for i in range(len(top3list['filename'])):
        a = top3list['filename'][i].split('_')
        print(a)
        tmp.append(a[0])
        tmps.append(a)
    for i in range(len(tmp)):
        tmp[i] = int(tmp[i])
    metas = pd.DataFrame()
    for i in range(3):
        metas = metas.append(meta_df[meta_df['id'].isin([tmp[i]])])
    metas = metas.reset_index().drop(['index'], axis=1)
    metas.to_csv(os.path.join('static', 'metas.csv'))

    # img_show(tmps) 이미지 파일 경로
    path_top1 = f'static\\{tmps[0][0]}\\{tmps[0][1]}\\{tmps[0][2]}'
    path_top2 = f'static\\{tmps[1][0]}\\{tmps[1][1]}\\{tmps[1][2]}'
    path_top3 = f'static\\{tmps[2][0]}\\{tmps[2][1]}\\{tmps[2][2]}'
    webtoonimage = './static/ml_img/002.png'
    # 다른 탬플릿의 변수를 넘기기 위해서 사용한 cache 기능
    cache.set('path_top1', f'static\\{tmps[0][0]}\\{tmps[0][1]}\\{tmps[0][2]}', timeout=5 * 60)
    cache.set('path_top2', f'static\\{tmps[1][0]}\\{tmps[1][1]}\\{tmps[1][2]}', timeout=5 * 60)
    cache.set('path_top3', f'static\\{tmps[2][0]}\\{tmps[2][1]}\\{tmps[2][2]}', timeout=5 * 60)
    return render_template('results.html', f_per=first, s_per=second, t_per=third, aa=path_top1, bb=path_top2,
                           cc=path_top3, metas=metas, webmyface=webtoonimage)


# 1순위 유사한 웹툰 캐릭터 개별 페이지
@app.route('/webtoons1.html', methods=['GET', 'POST'])
def movepage1():
    metas = pd.read_csv(r'D:\cp1\cp1_webtoonize\static\metas.csv')
    webtoonimage = r'D:\cp1\cp1_webtoonize\static\ml_img\002.png'
    testvar = cache.get('path_top1')
    return render_template('webtoons1.html', metas=metas, aa=testvar, webmyface=webtoonimage)


# 2순위 유사한 웹툰 캐릭터 개별 페이지
@app.route('/webtoons2.html', methods=['GET', 'POST'])
def movepage2():
    metas = pd.read_csv(r'D:\cp1\cp1_webtoonize\static\metas.csv')
    webtoonimage = r'D:\cp1\cp1_webtoonize\static\ml_img\002.png'
    testvar = cache.get('path_top2')
    return render_template('webtoons2.html', metas=metas, bb=testvar, webmyface=webtoonimage)


# 3순위 유사한 웹툰 캐릭터 개별 페이지
@app.route('/webtoons3.html', methods=['GET', 'POST'])
def movepage3():
    metas = pd.read_csv(r'D:\cp1\cp1_webtoonize\static\metas.csv')
    webtoonimage = r'D:\cp1\cp1_webtoonize\static\ml_img\002.png'
    testvar = cache.get('path_top3')
    return render_template('webtoons3.html', metas=metas, cc=testvar, webmyface=webtoonimage)


# 다시하기 버튼 및 데이터 삭제
@app.route('/homeb', methods=['GET', 'POST'])
def homeb():
    # resize_dir = './static/ml_img'
    # files = os.listdir(resize_dir)
    # for file in files:
    #     file_path = os.path.join(resize_dir, file)
    #     os.remove(file_path)
    print("--------")
    return render_template('main.html')


if __name__ == '__main__':
    app.run(debug=True, host=('0.0.0.0'))
