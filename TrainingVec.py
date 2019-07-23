import re
import codecs
import logging
import jieba
from typing import List
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import multiprocessing
import time
import warnings

warnings.filterwarnings("ignore")

# log基本设置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Step0：提取数据
def extract_data(data_path: str) -> List[str]:
    file_train_read = []
    with codecs.open(data_path, 'r', encoding='gbk') as dataTrainRaw:
        resource_data = dataTrainRaw.read()

        # 新闻内容的pattern
        content_pattern = re.compile(r'<content>\n(.*)\n</content>')
        # 新闻题目的pattern
        title_pattern = re.compile(r'<contenttitle>(.*)</contenttitle>')

        # 正则表达式匹配
        file_train_read.extend(content_pattern.findall(resource_data))
        file_train_read.extend(title_pattern.findall(resource_data))

        logger.info('提取内容共{}行'.format(len(file_train_read)))
        # 内容去重(因为发现有重复的新闻内容)
        file_train_read = list(set(file_train_read))
        logger.info('去重后内容共{}行'.format(len(file_train_read)))
        return file_train_read


# Step1：清洗 + 分词
def cut_data(file_train_read: List[str], seg_path: str) -> None:
    # 加载自定义词典
    jieba.load_userdict('./dic/dict.txt.big')
    # 输出文件路径
    seg_result_file = open(seg_path, 'a', encoding='utf-8')
    for sentence in file_train_read:
        # 清洗掉部分无用文字
        pattern = '\((责任编辑.*)\)'
        sentence = re.sub(pattern, '', sentence).replace('搜狐体育讯', '')

        # 去除标点符号和空格
        filter_r = "[\s+\.\!\/_,$%^*():：+\"\']+|[+——！，。？、~@#￥%……&*（）-]+|[“”\[\];?《》’【】■]+"
        sentence = re.sub(filter_r, '', sentence)

        # Jieba分词
        seg_list = jieba.cut(sentence, cut_all=False)
        seg_text = ' '.join(seg_list)
        seg_result_file.write(seg_text)
        seg_result_file.write('\n')
    seg_result_file.close()
    return


# Step2：训练Vec
def train_vec(input_file: str, out_dir: str, train_type: int) -> None:
    # outp_model 为输出模型, outp_vec2为原始c版本word2vec的vector格式的模型
    outp_model = out_dir + 'text_Type_{}.model'.format(train_type)
    outp_vec = out_dir + 'text_Type_{}.vector'.format(train_type)

    model = None
    # 选择训练的方式
    if train_type == 1:
        # 1.hs-SkipGram
        model = Word2Vec(LineSentence(input_file), size=50, window=5, sg=1, hs=1, workers=multiprocessing.cpu_count())
    elif train_type == 2:
        # 2.负采样-CBOW
        model = Word2Vec(LineSentence(input_file), size=50, window=5, sg=0, hs=0, negative=3,
                         workers=multiprocessing.cpu_count())
    elif train_type == 3:
        # 3.负采样-SkipGram
        model = Word2Vec(LineSentence(input_file), size=50, window=5, sg=1, hs=0, negative=3,
                         workers=multiprocessing.cpu_count())
    # 保存模型
    model.save(outp_model)
    model.wv.save_word2vec_format(outp_vec, binary=False)
    return


if __name__ == '__main__':
    # 语料路径
    data_inPath = './data/Sohu_sports.txt'
    data_outPath = './data/seg_data.txt'
    # 训练模型输出路径
    model_outdir = './word2vec/'

    # Step0:抽取数据
    file_read_data = extract_data(data_inPath)
    # Step1：清洗分词
    cut_data(file_read_data, data_outPath)

    # Step2：分模式训练
    time_1 = time.clock()
    train_vec(data_outPath, model_outdir, 1)
    time_2 = time.clock()
    train_vec(data_outPath, model_outdir, 2)
    time_3 = time.clock()
    train_vec(data_outPath, model_outdir, 3)
    end_time = time.clock()

    logger.info("hs-SkipGram训练时间为：{}\n负采样-CBOW训练时间为：{}\n负采样-SkipGram训练时间为：{}"
                .format(time_2 - time_1, time_3 - time_2, end_time - time_3))
