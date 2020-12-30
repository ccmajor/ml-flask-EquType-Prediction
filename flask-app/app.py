from flask import Flask, request, render_template, jsonify
#import pickle
import joblib
import numpy as np
import re
import jieba.posseg as pseg  # 词性标注
import jieba
import os
from sklearn.feature_extraction.text import CountVectorizer
import csv
import numpy as np
import pandas as pd


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False #实现中文显示




#  在请求进入视图函数之前 做出响应，只执行一次，用于生成模型
@app.before_first_request
def model_creation():
    # 声明全局变量
    global vec
    # 指定数据集路径
    dataset_path = '.\data'
    # 读取语料库
    datafile_corpus = os.path.join(dataset_path, 'Corpus_TF8_wuqita.csv')
    # 加载数据
    corpus_raw_data = pd.read_csv(datafile_corpus)
    #数据清洗
    corpus_cln_data = corpus_raw_data.dropna().copy()
    #设置特种设备类型标签
    label_mapping = {"锅炉": 1, "压力容器": 2, "压力管道": 3, "电梯": 4, "起重机械": 5, "客运索道": 6, "大型游乐设施": 7, "场（厂）内专用机动车辆": 8,
                     "其他": 9}
    corpus_cln_data['EquType'] = corpus_cln_data['EquType'].map(label_mapping)
    # 创建训练、测试集
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(corpus_cln_data['Words'].values,
                                                        corpus_cln_data['EquType'].values, random_state=1)
    words = []  # 为训练集创建list对象
    for line_index in range(len(x_train)):
        try:
            # x_train[line_index][word_index] = str(x_train[line_index][word_index])
            words.append(' '.join(x_train[line_index]))  # x_train转为list对象
        except:
            print(line_index)
    test_words = []  # 创建测试集的list对象
    for line_index in range(len(x_test)):
        try:
            # x_train[line_index][word_index] = str(x_train[line_index][word_index])
            test_words.append(' '.join(x_test[line_index]))
        except:
            print(line_index)
    # 构建词袋
    # CountVectorizer：将文本中的词语转换为词频矩阵，只考虑词频
    from sklearn.feature_extraction.text import CountVectorizer
    # max_features：对所有关键词的term frequency进行降序排序，取前max_features个作为关键词集
    vec = CountVectorizer(analyzer='char', max_features=3000, lowercase=False)
    vec.fit(words)
    # 构建贝叶斯模型
    from sklearn.naive_bayes import MultinomialNB  # MultinomialNB：Naive Bayes classifier for multinomial models
    classifier = MultinomialNB()
    classifier.fit(vec.transform(words), y_train)
    #查看预测准确率
    print('test_words_sorce_all:', classifier.score(vec.transform(test_words), y_test))  # 评估预测正确率平均值
    # 使用joblib保存模型
    joblib.dump(classifier, 'joblib_classifier.pkl')


@app.route('/')
def home():
	return render_template('home.html')

#测试json
equtypes = [
    {
        '锅炉': 1, "压力容器": 2, "压力管道": 3, "电梯": 4, "起重机械":5, "客运索道": 6,"大型游乐设施": 7,"场（厂）内专用机动车辆": 8,"其他": 9
    }
]
@app.route('/json',methods=['GET'])
def get_json():
    # 返回json格式特种设备类型名称
    return jsonify({'equtypes': equtypes})


@app.route('/getEquType',methods=['POST','GET'])
def get_EquType():
    if request.method=='POST':
        result=request.form
        pubwords = result['publicwords']

        # 停用词表路径
        stop_words_path = '.\stop_words'
        # 加载停用词表
        stopwords1 = [line.rstrip() for line in
                      open(os.path.join(stop_words_path, '中文停用词库.txt'), 'r', encoding='utf-8')]
        stopwords2 = [line.rstrip() for line in
                      open(os.path.join(stop_words_path, '哈工大停用词表.txt'), 'r', encoding='utf-8')]
        stopwords3 = [line.rstrip() for line in
                      open(os.path.join(stop_words_path, '四川大学机器智能实验室停用词库.txt'), 'r', encoding='utf-8')]
        stopwords4 = [line.rstrip() for line in
                      open(os.path.join(stop_words_path, '自建停用词库.txt'), 'r', encoding='utf-8')]

        stopwords = stopwords1 + stopwords2 + stopwords3 + stopwords4

        # 处理文本数据
        def proc_text(raw_line):
            """
                处理文本数据
                返回分词结果
            """

            # 1. 使用正则表达式去除非中文字符
            filter_pattern = re.compile('[^\u4E00-\u9FD5]+')
            chinese_only = filter_pattern.sub('', raw_line)

            ## 2. 结巴分词+词性标注
            # word_list = pseg.cut(chinese_only)

            # jieba分词
            # seg_list = jieba.cut("这是一句话", cut_all=True) # 全模式
            # print("全模式: " + "/ ".join(seg_list))  # 全模式

            seg_list = jieba.cut(chinese_only, cut_all=False)  # 精确模式
            # print("精确模式: " + "/ ".join(seg_list))  # 精确模式

            # 3. 去除停用词，保留有意义的词性
            # 动词，形容词，副词
            # used_flags = ['v', 'a', 'ad']
            meaninful_words = []
            # for word, flag in word_list:
            for word in seg_list:
                # if (word not in stopwords) and (flag in used_flags):
                if (word not in stopwords):
                    meaninful_words.append(word)
            return ' '.join(meaninful_words)
            return ' '.join(seg_list)  # 返回内容，而非地址，如果直接 return seg_list 返回的是地址

        pub_test = proc_text(pubwords)
        pub_test_words = [pub_test]  # 为训练集创建list对象
        # for line_index in range(len(x_test)):
        #     try:
        #         # x_train[line_index][word_index] = str(x_train[line_index][word_index])
        #         words.append(' '.join(x_test[line_index]))  # x_train转为list对象
        #     except:
        #         print(line_index)
        print(pub_test_words[0])

        #x_test_words=['早 上   点   广 州   番 禺   兴 南   大 道   一 间   名 为   新 丰   小 食 店   街 坊   回 忆   当 时   正 常   做 生 意   突 然   听 到   一 声   巨 响   小 食 店   出   浓 烟   店 内   大 量   物 品   炸 毁   现 场   一 片 狼 藉']
        vec_test_word = vec.transform(pub_test_words)
        print(vec_test_word)
        print(vec_test_word.shape)


        #pkl_file = open('classifier.pkl', 'rb')
        new_classifier = joblib.load('joblib_classifier.pkl')
        prediction = new_classifier.predict(vec_test_word[0:1])

        return render_template('result.html',prediction=prediction)

@app.route('/getEquTypeAPI',methods=['POST'])
def get_EquTypeAPI():
    if request.method=='POST':
        result=request.json
        result_df=pd.DataFrame(result)

        #pubwords = result_df.loc[1,'words']

        # 停用词表路径
        stop_words_path = '.\stop_words'
        # 加载停用词表
        stopwords1 = [line.rstrip() for line in
                      open(os.path.join(stop_words_path, '中文停用词库.txt'), 'r', encoding='utf-8')]
        stopwords2 = [line.rstrip() for line in
                      open(os.path.join(stop_words_path, '哈工大停用词表.txt'), 'r', encoding='utf-8')]
        stopwords3 = [line.rstrip() for line in
                      open(os.path.join(stop_words_path, '四川大学机器智能实验室停用词库.txt'), 'r', encoding='utf-8')]
        stopwords4 = [line.rstrip() for line in
                      open(os.path.join(stop_words_path, '自建停用词库.txt'), 'r', encoding='utf-8')]

        stopwords = stopwords1 + stopwords2 + stopwords3 + stopwords4

        # 处理文本数据
        def proc_text(raw_line):
            """
                处理文本数据
                返回分词结果
            """

            # 1. 使用正则表达式去除非中文字符
            filter_pattern = re.compile('[^\u4E00-\u9FD5]+')
            chinese_only = filter_pattern.sub('', raw_line)

            ## 2. 结巴分词+词性标注
            # word_list = pseg.cut(chinese_only)

            # jieba分词
            # seg_list = jieba.cut("这是一句话", cut_all=True) # 全模式
            # print("全模式: " + "/ ".join(seg_list))  # 全模式

            seg_list = jieba.cut(chinese_only, cut_all=False)  # 精确模式
            # print("精确模式: " + "/ ".join(seg_list))  # 精确模式

            # 3. 去除停用词，保留有意义的词性
            # 动词，形容词，副词
            # used_flags = ['v', 'a', 'ad']
            meaninful_words = []
            # for word, flag in word_list:
            for word in seg_list:
                # if (word not in stopwords) and (flag in used_flags):
                if (word not in stopwords):
                    meaninful_words.append(word)
            return ' '.join(meaninful_words)
            return ' '.join(seg_list)  # 返回内容，而非地址，如果直接 return seg_list 返回的是地址

        #pub_test = proc_text(pubwords)
        #pub_test_words = [pub_test]  # 为训练集创建list对象
        result_df['words_'] = result_df['words'].apply(proc_text)

        pubwords = []
        for line_index in range(len(result_df['words_'])):
            try:
                # x_train[line_index][word_index] = str(x_train[line_index][word_index])
                pubwords.append(' '.join(result_df.loc[line_index,'words_']))  # 转为list对象
            except:
                print(line_index)

        #x_test_words=['早 上   点   广 州   番 禺   兴 南   大 道   一 间   名 为   新 丰   小 食 店   街 坊   回 忆   当 时   正 常   做 生 意   突 然   听 到   一 声   巨 响   小 食 店   出   浓 烟   店 内   大 量   物 品   炸 毁   现 场   一 片 狼 藉']

        # vec_test_word = vec.transform(pub_test_words)
        # print(vec_test_word)
        # print(vec_test_word.shape)


        #pkl_file = open('classifier.pkl', 'rb')
        new_classifier = joblib.load('joblib_classifier.pkl')
        prediction = list(new_classifier.predict(vec.transform(pubwords)))

        #return render_template('result.html',prediction=prediction)
        return jsonify({'prediction': str(prediction)})
    
if __name__ == '__main__':
    app.debug = True
    #app.run(host='192.168.5.206', port=5632)
    app.run()