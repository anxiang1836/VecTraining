import gensim
import warnings
warnings.filterwarnings(action='ignore')# 忽略警告

if __name__ == '__main__':
    fdir = './word2vec/'
    train_type = {'hs-SkipGram': 1, '负采样-CBOW': 2, '负采样-SkipGram': 3}
    x = '火箭'
    print('输出与【{}】最相近的10个词'.format(x))
    print('*********************')
    for t in train_type.keys():
        # 加载模型
        model = gensim.models.Word2Vec.load(fdir + 'text_Type_{}.model'.format(train_type[t]))
        print('采用{}方式训练'.format(t))
        word = model.most_similar(x)
        for text in word:
            print(text[0], text[1])
        print('*********************')
