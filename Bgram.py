# -*- coding: UTF-8 -*-

from collections import Counter
import numpy as np
import math
import time
import os
import re


def _word_ngrams(tokens, stop_words=None, ngram_range=(1, 1), separator=' '):  # from sklearn import CountVectorizer
    """Turn tokens into a sequence of n-grams after stop words filtering"""
    # handle stop words
    if stop_words is not None:
        tokens = [w for w in tokens if w not in stop_words]

    # handle token n-grams
    min_n, max_n = ngram_range
    if max_n != 1:
        original_tokens = tokens
        tokens = []
        n_original_tokens = len(original_tokens)
        for n in range(min_n,
                        min(max_n + 1, n_original_tokens + 1)):
            for i in range(n_original_tokens - n + 1):
                tokens.append(separator.join(original_tokens[i: i + n]))
    return tokens


class _NGramModel:
    def __init__(self, cdict, train, smooth=0):
        """
        模型初始化
        :param cdict: 词典资源
        :param train: 训练集
        :param smooth: 选取平滑方式序号
        :return: None
        """
        time_start = time.time()
        self._smooths = [self._probf__0, self._probf__1]
        # 初始化词典
        word_dict = dict()
        for lineno, line in enumerate(cdict):
            try:
                content = line.strip().split()
                word_dict.setdefault(content[1], int(content[2]))
            except ValueError:
                raise ValueError(
                    'invalid dictionary entry in %s at Line %s: %s' % (cdict.name, lineno, line))  # 错误处理 文件 行号 行
        # 初始化前缀词典
        pre_dict = dict()
        train_dict = dict()
        for lineno, line in enumerate(train):
            try:
                line = line.split('/m')[1]
                line = line.strip('\t').strip()
                #line = line.replace('。', '。\tEND\tBOS\t')
                line = re.sub(r'([。？！])', r'\1\tEND\tBOS\t', line).strip('\tEND\tBOS\t')
                line = 'BOS\t' + line + '\tEND'
                line = line.replace(' ', '')
                words = line.split('\t')
                bgramwords = _word_ngrams(tokens=words, ngram_range=(2,2))
                for word in bgramwords:
                    pre, wd = word.split(' ')
                    pre_dict.setdefault(wd, {})
                    pre_dict[wd].setdefault(pre, 0)
                    pre_dict[wd][pre] += 1
                    train_dict.setdefault(pre, 0)
                    train_dict[pre] += 1
            except ValueError:
                raise ValueError(
                    'invalid dictionary entry in %s at Line %s: %s \n word: %s' % (train.name, lineno, line, word))  # 错误处理 文件 行号 行
        pre_dict.pop('BOS')              # 'END BOS' 不需要列入统计
        train_dict.pop('BOS')
        train_dict.pop('END')

        # 类成员定义
        self._word_dict = word_dict               # 词典
        self._train_dict = train_dict             # 训练集词典
        self._pre_dict = pre_dict                 # 前缀词典
        self._V = len(train_dict)                 # 训练集词汇数
        self._probf = self._smooths[smooth]       # 平滑后概率值公式

        # 计时
        time_end = time.time()
        print("模型训练时长:", time_end - time_start, 's\n')

    def split(self, text, word_max_len = 4):
        """
        中文文本切分
        :param text:待切分文本
        :param word_max_len:最大词长
        :return:切分文本
        """
        text = re.sub(r'([。？！])', r'\1<>', text).strip('<>')
        sts = text.split('<>')
        for n, s in enumerate(sts):
            sts[n], ps = self.split_sentence(sentence=s, word_max_len=word_max_len)
        st = '/'.join(sts)
        return st

    def split_sentence(self, sentence, word_max_len=4):
        """
        单句分词
        :param sentence:待切分文本
        :param word_max_len:最大词长
        :return:最终切分文本，切分结果各自的概率值
        """
        ps = [int() for i in range(word_max_len-1)]
        listcut = [list() for i in range(word_max_len-1)]
        sts = list()
        for i in range(word_max_len - 1, 0, -1):
            st = sentence
            words = _word_ngrams(tokens=sentence, ngram_range=(i+1, i+1), separator='')
            ps[i-1] = 0
            psn = 0
            temp = -word_max_len
            for flag, word in enumerate(words):
                if word in self._word_dict:
                    if self._word_dict[word] > ps[i-1]:
                        ps[i-1] = self._word_dict[word]
                    if flag - temp > i:
                        listcut[i - 1].append(word)
                        temp = flag
                        st = st.replace(word, '<>', 1)
                        psn = self._word_dict[word]
                    elif self._word_dict[word] > psn:
                        st = (st[::-1].replace('<>'[::-1], words[temp][::-1], 1))[::-1]
                        temp = flag
                        listcut[i - 1].pop()
                        listcut[i - 1].append(word)
                        st = st.replace(word, '<>', 1)
                        psn = self._word_dict[word]
            stt = st.split('<>')
            for seg in range(len(stt)):
                reg = self._spst(stt[seg], word_max_len - 1)
                stt[seg] = reg
            st = '<>'.join(stt)
            for word in listcut[i-1]:
                st = st.replace('<>', '/'+word+'/', 1)
            st = re.sub(r'/+', '/', st).strip('/')          # 整理格式
            '''print(st, stt)'''
            sts.append(st)
        psst = 0
        for n, s in enumerate(sts):
            ns = self._prob(s)
            ps[n] = ns
            if ns > psst:
                psst = ns
                st = s
        return st, ps

    def _spst(self, sentence, word_max_len=3):
        """
        用于递归的分词函数
        :param sentence:待切分句子
        :param word_max_len:最大词长
        :return:切分后的句子
        """
        st = sentence
        if word_max_len <= 0:
            return st
        if len(st) <= word_max_len:
            if st in self._word_dict:
                return st
            else:
                return self._spst(st, word_max_len - 1)
        psn = 0
        listcut = list()
        words = _word_ngrams(tokens=st, ngram_range=(word_max_len, word_max_len), separator='')
        temp = -word_max_len
        for flag, word in enumerate(words):
            if word in self._word_dict:
                if flag - temp > word_max_len - 1:
                    listcut.append(word)
                    temp = flag
                    st = st.replace(word, '<>', 1)
                    psn = self._word_dict[word]
                elif self._word_dict[word] > psn:
                    st = (st[::-1].replace('<>'[::-1], words[temp][::-1], 1))[::-1]
                    temp = flag
                    listcut.pop()
                    listcut.append(word)
                    st = st.replace(word, '<>', 1)
                    psn = self._word_dict[word]
        stt = st.split('<>')
        for seg in range(len(stt)):
            reg = self._spst(stt[seg], word_max_len-1)
            stt[seg] = reg
        st = '<>'.join(stt)
        for word in listcut:
            st = st.replace('<>', '/' + word + '/', 1)
        st = re.sub(r'/+', '/', st).strip('/')  # 整理格式
        return st

    def _prob(self, sentence):
        """
        计算切分概率值
        :param sentence:切分后的句子
        :return:单句概率值
        """
        st = 'BOS/' + sentence + '/END'
        st = st.replace(' ', '')
        words = st.split('/')
        bgramwords = _word_ngrams(tokens=words, ngram_range=(2, 2))
        probs = list()
        for word in bgramwords:
            pre, wd = word.split(' ')
            if wd in self._pre_dict and pre in self._pre_dict[wd]:
                probs.append(self._probf(pre, wd))
            elif pre != 'BOS':
                probs.append(self._probf())
        return np.prod(probs)

    def _probf__0(self, word_pre='', word=''):
        """
        无平滑
        :param word_pre:前缀词
        :param word:词
        :return: 无平滑结果
        """
        return self._pre_dict.get(word, dict()).get(word_pre, 0) / self._train_dict.get(word, 1)

    def _probf__1(self, word_pre='', word=''):
        """
        加一平滑
        :param word_pre:前缀词
        :param word:词
        :return: 加一平滑后的结果
        """
        return (self._pre_dict.get(word, dict()).get(word_pre, 0) + 1) / (self._train_dict.get(word, 0) + self._V)

    def findwords(self, text, word_len=2):
        """
        未登录词识别（二字词）
        :param text:语料
        :return:
        """
        # original_txt = text
        text = text.read()
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        #text = re.sub(r'\W+', '', text.replace('_', ''))    # 去除符号
        #text = re.sub(r'[a-zA-Z0-9]', '', text)             # 去除数字和字母
        total = len(text)
        character = dict(Counter(text))
        original_words = _word_ngrams(tokens=text, ngram_range=(word_len, word_len), separator='')

        words = {}
        for word in original_words:
            if word not in self._word_dict:
                words[word] = words.get(word, 0) + 1
        words = dict(Counter(words))
        words = {k: v for k, v in words.items() if v >= 3}      # 去除词频小于3的二字词
        original_words = words.copy()                           # 保留词频
        for word in words:
            # 点互信值(PMI)
            # log ((词频/总字数) / ( (单字1频数/总字数) * (单字2频数/总字数) )
            words[word] = math.log2(words[word] / character[word[0]] / character[word[1]] * total)
        words = sorted(words.items(), key=lambda item: item[1], reverse=True)   # 根据PMI从大到小排序
        words = dict(words[:len(words)//500 if len(words) >= 500 else 1])       # 保留点互信值排名前0.2%的未登陆词
        temp = words.keys()
        LE = dict.fromkeys(temp, 0)   # 左邻接熵初始化
        RE = dict.fromkeys(temp, 0)   # 右邻接熵初始化
        for word in LE:
            ch = re.findall(('(.)(?:%s)') % word, text)
            ed = len(ch)
            ch = Counter(ch).values()
            for v in ch:
                LE[word] -= v/ed * math.log2(v/ed)
        for word in RE:
            ch = re.findall(('(?:%s)(.)') % word, text)
            ed = len(ch)
            ch = Counter(ch).values()
            for v in ch:
                RE[word] -= v / ed * math.log2(v / ed)
        LE = sorted(LE.items(), key=lambda item: item[1], reverse=True)        # 左邻接熵从大到小排序
        RE = sorted(RE.items(), key=lambda item: item[1], reverse=True)        # 右邻接熵从大到小排序
        LE = dict(LE[:len(LE) // 5 if len(words) >= 5 else 1])                # 保留左邻接熵排名前20%的未登陆词
        RE = dict(RE[:len(RE) // 5 if len(words) >= 5 else 1])                # 保留右邻接熵排名前20%的未登陆词
        LE = set(LE.keys())
        RE = set(RE.keys())
        words = LE & RE
        words = {k: original_words[k] for k in words}
        return words

    def reworddict(self, cdict):
        """
        根据本地词典资源重载词典
        :param cdict: 词典资源
        :return: None
        """
        word_dict = dict()
        for lineno, line in enumerate(cdict):
            try:
                content = line.strip().split()
                word_dict.setdefault(content[1], int(content[2]))
            except ValueError:
                raise ValueError(
                    'invalid dictionary entry in %s at Line %s: %s' % (cdict.name, lineno, line))  # 错误处理 文件 行号 行
        self._word_dict = word_dict

    def repredict(self, train):
        """
        根据输入的训练集重新训练2-gram模型
        :param train: 训练集
        :return: None
        """
        pre_dict = dict()
        train_dict = dict()
        for lineno, line in enumerate(train):
            try:
                line = line.split('/m')[1]
                line = line.strip('\t').strip()
                # line = line.replace('。', '。\tEND\tBOS\t')
                line = re.sub(r'([。？！])', r'\1\tEND\tBOS\t', line).strip('\tEND\tBOS\t')
                line = 'BOS\t' + line + '\tEND'
                line = line.replace(' ', '')
                words = line.split('\t')
                bgramwords = _word_ngrams(tokens=words, ngram_range=(2, 2))
                for word in bgramwords:
                    pre, wd = word.split(' ')
                    pre_dict.setdefault(wd, {})
                    pre_dict[wd].setdefault(pre, 0)
                    pre_dict[wd][pre] += 1
                    train_dict.setdefault(pre, 0)
                    train_dict[pre] += 1
            except ValueError:
                raise ValueError(
                    'invalid dictionary entry in %s at Line %s: %s \n word: %s' % (train.name, lineno, line, word))  # 错误处理 文件 行号 行
        pre_dict.pop('BOS')  # 'END BOS' 不需要列入统计
        train_dict.pop('BOS')
        train_dict.pop('END')

        self._train_dict = train_dict   # 训练集词典
        self._pre_dict = pre_dict       # 前缀词典
        self._V = len(train_dict)       # 训练集词汇数

    def resmooth(self, smooth):
        """
        重新选择概率值公式（选择平滑方式）
        :param smooth:
        :return: None
        """
        self._probf = self._smooths[smooth]

    def test(self, test, answer):
        """
        测试集验证模型精度
        :param test: 测试集
        :param answer: 测试集理想分词结果
        :return:准确率、召回率、F-测度值
        """
        testline = list()
        answerline = list()
        for lineno, line in enumerate(test):
            try:
                line = line.split('/m')[1].strip('\t').strip()
                testline.append(line)
            except ValueError:
                raise ValueError(
                    'invalid dictionary entry in %s at Line %s: %s' % (test.name, lineno, line))  # 错误处理 文件 行号 行
        for lineno, line in enumerate(answer):
            try:
                line = line.split('/m')[1].strip('\t').strip()
                answerline.append(line)
            except ValueError:
                raise ValueError(
                    'invalid dictionary entry in %s at Line %s: %s' % (test.name, lineno, line))  # 错误处理 文件 行号 行
        pm, pd = list(), list()     # 正确答案个数、总输出数
        rd = 0                      # 标准答案个数
        for lineno, line in enumerate(testline):    # 使用区间表示分词情况
            st1 = self.split(line)
            words1 = st1.split('/')
            words2 = answerline[lineno].split('\t')
            pm.append(0)        # 初始化单句准确率分子
            pd.append(0)        # 初始化单句准确率分母
            rd += len(words2)   # 召回率分母
            pd[lineno] += len(words1)
            word1 = set()
            flag = 1
            for word in words1:
                word1.add((flag, len(word)))
                flag += len(word)
            word2 = set()
            flag = 1
            for word in words2:
                word2.add((flag, len(word)))
                flag += len(word)
            pm[lineno] += len(word1 & word2)
        precision = sum(pm) / sum(pd)
        recall = sum(pm) / rd
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def show(self):
        print('V:' + str(self._V))
        print(self._probf)


def main():
    # 训练模型
    dictname = '30wChinsesSeqDic.txt'   # 词典资源
    train = 'train.txt'                 # 训练集
    with open(os.getcwd() + '/' + dictname, encoding='UTF-8-sig') as fp:
        with open(os.getcwd() + '/' + train, encoding='UTF-8-sig') as f:
            model = _NGramModel(cdict=fp, train=f, smooth=1)             # 生成n-Gram模型
    # 测试
    time_start = time.time()
    test = 'test.txt'
    answer = 'answer.txt'
    with open(os.getcwd() + '/' + test, encoding='UTF-8-sig') as fp:
        with open(os.getcwd() + '/' + answer, encoding='UTF-8-sig') as f:
            precision, recall, f1 = model.test(test=fp, answer=f)    # 通过测试集测试模型精度
    time_end = time.time()
    print('Correct ratio/Precision - {:%}'.format(precision))                  # 准确率输出
    print('Recall ratio - {:%}'.format(recall))                                # 召回率输出
    print('F-Measure - {:%}'.format(f1))                                       # F-测度值输出
    print("time cost:", time_end - time_start, 's\n')
    # 分词演示
    time_start = time.time()
    text = "什么是金融资本呢？金融资本就是在实现剩余商品到货币的转换以后，在如何分配这些货币资本的问题上纠缠不休的资本。也就是说，金融资本是在工业资本完成了由货币到商品（即购买生产资料和雇工）和再由商品到货币（即产品市场出卖）的两个转换以后，在蛋糕造好了以后，就如何分蛋糕的抢夺中，通过贷款利息、股权和期货交易等等手段大显身手的资本。金融资本本身和商品价值的创造毫无关系，因而它是寄生性的。"
    st = model.split(text)
    time_end = time.time()
    print('被切分文本： ' + text)
    print('最终切分结果： ' + st)
    print("time cost:", time_end - time_start, 's\n')
    # 未登录词识别
    time_start = time.time()
    with open(os.getcwd() + '/COAE2015微博观点句识别语料.txt', encoding='UTF-8-sig') as fp:
        words = model.findwords(fp)
    time_end = time.time()
    print('未登录词识别结果：', words)
    print("time cost:", time_end - time_start, 's')
    pass


if __name__ == "__main__":
    main()
