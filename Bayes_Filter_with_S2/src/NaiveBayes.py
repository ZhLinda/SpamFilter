# -- coding: utf-8 --
from numpy import *
from Feature_extraction import *
from Ngram import *
from Handle_S2 import *


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



def classify(vec_s2, vec2classify, p_words_ham, p_words_spam, p_spam, s2_theta, s2_sigma):

    p_f_ham = - 0.5 * np.log(2. * np.pi * s2_sigma[0, :])
    p_f_ham -= 0.5 * ((vec_s2 - s2_theta[0, :]) ** 2) / (s2_sigma[0, :])

    p_f_spam = - 0.5 * np.log(2. * np.pi * s2_sigma[1, :])
    p_f_spam -= 0.5 * ((vec_s2 - s2_theta[1, :]) ** 2) / (s2_sigma[1, :])

    '''
    vec2classify = list(vec2classify)
    vec2classify.extend(list(vec_s2))
    p_words_spam = list(p_words_spam)
    p_words_spam.extend(p_f_spam)
    p_words_ham = list(p_words_ham)
    p_words_ham.extend(p_f_ham)
    '''

    ps = sum(vec2classify * np.array(p_words_spam)) + sum(np.array(p_f_spam)) + np.log(p_spam)
    ph = sum(vec2classify * np.array(p_words_ham)) + sum(np.array(p_f_ham)) + np.log(1 - p_spam)


    if ps > ph:
        return 1,ps,ph
    else:
        return 0,ps,ph

def save_dictionary(dictionary,filename):
    print "----------正在保存字典----------\n"
    fw = open(filename,"w")
    for i in range(len(dictionary)):
        fw.write(dictionary[i].encode("utf-8") + "\n")
    fw.flush()
    fw.close()
    print "----------字典保存完成----------\n"

def load_dictionary(filename):
    print "----------正在加载字典----------\n"
    fpd = open(filename,"r")
    dictionary = [line.strip().decode('utf-8') for line in fpd.readlines()]
    fpd.close()
    print "----------字典加载完成----------\n"
    return dictionary

def save_train_vectors(train_vectors,filename):
    print "----------正在保存特征向量----------\n"
    fp = open(filename, "w")  # 存储setOfWords特征向量
    for i in range(len(train_vectors)):
        for j in range(len(train_vectors[0])):
            fp.write(train_vectors[i][j].__str__() + " ")
        fp.write("\n")
    print "----------特征向量保存完成----------\n"

def load_train_vectors(len_doc,len_dictionary,filename):
    print "----------正在加载特征向量----------\n"
    train_vectors = np.zeros((len_doc, len_dictionary), np.float64)
    fp = open(filename, "r")
    lines = fp.readlines()
    l = len(lines)
    for i in range(l):
        line = lines[i].strip().split()
        size = len(line)
        for j in range(size):
            train_vectors[i][j] = float(line[j])
    print "----------特征向量加载完成----------\n"
    return train_vectors






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
    fpSpam = open("../model/pSpam.txt","w+")
    spam = p_spam.__str__()
    fpSpam.write(spam)
    fpSpam.close()
    fw = open("../model/dictionary.txt","w+")
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