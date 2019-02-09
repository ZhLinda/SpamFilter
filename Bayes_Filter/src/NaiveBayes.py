# -- coding: utf-8 --
from numpy import *
from Feature_extraction import *
from Ngram import *


def trainNB(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix) #the number of training examples
    numWords = len(trainMatrix[0]) #the number of words in one email
    p_spam = sum(trainCategory)/float(numTrainDocs)
    p0num = ones(numWords); p1num = ones(numWords)
    p0total = 2.0; p1total = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1num += trainMatrix[i]
            p1total += sum(trainMatrix[i])
        else:
            p0num += trainMatrix[i]
            p0total += sum(trainMatrix[i])
    p1vec = np.log(p1num / p1total)
    p0vec = np.log(p0num / p0total)
    return p0vec,p1vec,p_spam



def classify(vec2classify, p_words_ham, p_words_spam, p_spam):
    ps = sum(vec2classify * p_words_spam) + np.log(p_spam)
    ph = sum(vec2classify * p_words_ham) + np.log(1 - p_spam)
    if ps > ph:
        return 1,ps,ph
    else:
        return 0,ps,ph




def Bayesian_classifier(traindata, testdata, trainlabel):

    method = ['IG','CHI','MI']
    m = method[2]

    flag = 1
    if flag == 1: #重新选择特征，训练模型
        print "----------正在进行特征选择----------"
        feature_set = feature_selection(traindata, trainlabel, m)
        print len(feature_set)
        print "----------特征选择完成----------\n"
        # size = 0.1 * len(feature_set)
        size = 3500
        dictionary = feature_set[:int(size)]  # 选取前1000个词词作为特征
        # for term in dictionary:
        # print term

        print "----------正在进行word-to-vector转化----------"
        train_vectors = word2vec_setOfWords(traindata, dictionary)
        print "----------word-to-vector转化完成----------\n"

        print "----------正在进行模型训练----------"
        p_words_ham, p_words_spam, p_spam = trainNB(train_vectors, trainlabel)
        save_NB_model(p_words_ham, p_words_spam, p_spam, dictionary)

        print "----------模型训练完成----------\n"
        del feature_set,train_vectors

    else: #载入储存的模型
        p_words_ham,p_words_spam,p_spam,dictionary = load_NB_model()

    print "----------正在进行testing----------"

    test_vector = word2vec_setOfWords(testdata,dictionary)

    classify_result = []
    for vec2classify in test_vector:
        label,t1,t2 = classify(vec2classify,p_words_ham,p_words_spam,p_spam)
        classify_result.append(label)
    print "----------testing完成----------\n"

    del dictionary,test_vector
    gc.collect()
    return classify_result


def evaluate(pre_result, test_label):
    TP = 0; FN = 0; FP = 0; TN = 0
    length = len(pre_result)
    print pre_result
    print test_label
    for i in range(length):
        if pre_result[i] == 1 and test_label[i] == 1:
            TP += 1
        elif pre_result[i] == 1 and test_label[i] == 0:
            FP += 1
        elif pre_result[i] == 0 and test_label[i] == 1:
            FN += 1
        else:
            TN += 1
    print TP,FN,FP,TN
    Precise = float(TP) / float(TP + FP)
    Recall = float(TP) / float(TP + FN)
    F1 = float(2 * TP) / float(2 * TP + FP + FN)
    print "精确率为： ",Precise
    print "召回率为： ",Recall
    print "F1值为： ",F1

def save_NB_model(p_words_ham,p_words_spam,p_spam,dictionary):
    print "----------正在保存模型----------\n"
    fpSpam = open("../model/pSpam.txt","w")
    spam = p_spam.__str__()
    fpSpam.write(spam)
    fpSpam.close()
    fw = open("../model/dictionary.txt","w")
    for i in range(len(dictionary)):
        fw.write(dictionary[i].encode("utf-8") + "\n")
    fw.flush()
    fw.close()
    np.savetxt("../model/p_words_spam.txt",p_words_spam,delimiter="\t")
    np.savetxt("../model/p_words_ham.txt",p_words_ham,delimiter="\t")

    print "----------模型保存完成----------\n"

def load_NB_model():
    fpd = open("../model/dictionary.txt","r")
    dictionary = [line.strip().decode('utf-8') for line in fpd.readlines()]
    fpd.close()
    p_words_spam = np.loadtxt("../model/p_words_spam.txt",delimiter="\t")
    p_words_ham = np.loadtxt("../model/p_words_ham.txt",delimiter="\t")
    fr = open("../model/pSpam.txt","r")
    pSpam = float(fr.readline().strip())
    fr.close()
    return p_words_ham,p_words_spam,pSpam,dictionary


'''
if __name__ == "__main__":

    class_label = load_class_data('../data/derived/classLabel_100w.txt')
    train_corpus = load_train_data('../data/derived/train_seg_100w.txt')

    #交叉验证，验证集占总训练集的20%
    train_data, test_data, train_label, test_label = train_test_split(train_corpus, class_label,test_size=0.1)
    predict_result = Bayesian_classifier(train_data, test_data, train_label, test_label)
    evaluate(predict_result,test_label)'''