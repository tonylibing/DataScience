import codecs
import os
import glob
import re
import pickle
import random
import time
from pprint import pprint
import gensim
import jieba
import jieba.analyse
import jieba.analyse
import jieba.posseg as psg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim import corpora, models
from gensim import matutils
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from functools import lru_cache
import numpy.linalg
import scipy.optimize
from six.moves import xrange

""" Module to compute projections on the positive simplex or the L1-ball

A positive simplex is a set X = { \mathbf{x} | \sum_i x_i = s, x_i \geq 0 }

The (unit) L1-ball is the set X = { \mathbf{x} | || x ||_1 \leq 1 }

Adrien Gaidon - INRIA - 2011
"""



def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the simplex

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.

    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def euclidean_proj_l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the L1-ball

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s

    Notes
    -----
    Solves the problem by a reduction to the positive simplex case

    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w


e = 1e-100
error_diff = 1.0


class CollaborativeTopicModel():
    """
    Wang, Chong, and David M. Blei. "Collaborative topic modeling for recommending scientific articles."
    Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2011.
    Attributes
    ----------
    n_item: int
        number of items
    n_user: int
        number of users
    R: ndarray, shape (n_user, n_item)
        user x item rating matrix
    """

    def __init__(self,lda, n_topic, n_voca, n_user, n_item, doc_ids, doc_cnt, ratings):
        self.lda_model = lda
        self.verbose=True
        self.lambda_u = 0.01
        self.lambda_v = 0.01
        self.alpha = 1
        self.eta = 0.01
        self.a = 1
        self.b = 0.01

        self.n_topic = n_topic
        self.n_voca = n_voca
        self.n_user = n_user
        self.n_item = n_item

        # U = user_topic matrix, U x K
        self.U = np.random.multivariate_normal(np.zeros(n_topic), np.identity(n_topic) * (1. / self.lambda_u),
                                               size=self.n_user)
        # V = item(doc)_topic matrix, V x K
        self.V = np.random.multivariate_normal(np.zeros(n_topic), np.identity(n_topic) * (1. / self.lambda_u),
                                               size=self.n_item)
        self.theta = self.load_lda_distribution()
        # self.theta = np.random.random([n_item, n_topic])
        # self.theta = self.theta / self.theta.sum(1)[:, np.newaxis]  # normalize
        # self.beta = np.random.random([n_voca, n_topic])
        # self.beta = self.beta / self.beta.sum(0)  # normalize

        self.doc_ids = doc_ids
        self.doc_cnt = doc_cnt

        self.C = np.zeros([n_user, n_item]) + self.b
        self.R = np.zeros([n_user, n_item])  # user_size x item_size

        for di in xrange(len(ratings)):
            rate = ratings[di]
            for user in rate:
                self.C[di,user] += self.a - self.b
                self.R[di,user] = 1

        self.phi_sum = np.zeros([n_voca, n_topic]) + self.eta

    def load_lda_distribution(self):
        return self.lda_model.get_topic_distribution()

    def fit(self, doc_ids, doc_cnt, rating_matrix, max_iter=1000):
        old_err = 0
        for iteration in xrange(max_iter):
            tic = time.clock()
            self.do_e_step()
            self.do_m_step()
            err = self.sqr_error()
            if self.verbose:
                print('[ITER] {0},\tElapsed time:{1},\tReconstruction error:{2}'.format(iteration, time.clock() - tic, err))
            if abs(old_err - err) < error_diff:
                break

    # reconstructing matrix for prediction
    def predict_item(self):
        return np.dot(self.U, self.V.T)

    # reconstruction error
    def sqr_error(self):
        err = (self.R - self.predict_item()) ** 2
        err = err.sum()

        return err

    def do_e_step(self):
        self.update_u()
        self.update_v()
        # self.update_theta()

    def update_theta(self):
        def func(x, v, phi, beta, lambda_v):
            return 0.5 * lambda_v * np.dot((v - x).T, v - x) - np.sum(np.sum(phi * (np.log(x * beta) - np.log(phi))))

        for vi in xrange(self.n_item):
            W = np.array(self.doc_ids[vi])
            word_beta = self.beta[W, :]
            phi = self.theta[vi, :] * word_beta + e  # W x K
            phi = phi / phi.sum(1)[:, np.newaxis]
            result = scipy.optimize.minimize(func, self.theta[vi, :], method='nelder-mead',
                                             args=(self.V[vi, :], phi, word_beta, self.lambda_v))
            self.theta[vi, :] = euclidean_proj_simplex(result.x)
            self.phi_sum[W, :] += np.array(self.doc_cnt[vi])[:, np.newaxis] * phi

    def update_u(self):
        for ui in xrange(self.n_user):
            left = np.dot(self.V.T * self.C[ui, :], self.V) + self.lambda_u * np.identity(self.n_topic)
            self.U[ui, :] = numpy.linalg.solve(left, np.dot(self.V.T * self.C[ui, :], self.R[ui, :]))

    def update_v(self):
        for vi in xrange(self.n_item):
            left = np.dot(self.U.T * self.C[:, vi], self.U) + self.lambda_v * np.identity(self.n_topic)
            self.V[vi, :] = numpy.linalg.solve(left, np.dot(self.U.T * self.C[:, vi],self.R[:, vi]) + self.lambda_v * self.theta[vi, :])

    def do_m_step(self):
        self.beta = self.phi_sum / self.phi_sum.sum(0)
        self.phi_sum = np.zeros([self.n_voca, self.n_topic]) + self.eta

    def in_matrix_test(self):
        pass

    def outof_matrix_test(self):
        pass


class WordSeg1():
    def __init__(self,stopwords_path = ''):
        self.stop_words = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]

    def seg_stopword_sentence(self,sentence):
        sentence = sentence
        sentence_seged = jieba.cut(sentence)
        outstr = ''
        for word in sentence_seged:
            if word not in self.stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        return outstr

    def cut(self,df,col='content'):
        return df[col].apply(self.seg_stopword_sentence)

class WordSeg():
    def __init__(self,stopwords_path = '',user_dict=None):
        jieba.initialize()
        jieba.enable_parallel()
        jieba.analyse.set_stop_words(stopwords_path)
        if user_dict is not None:
            jieba.load_userdict(user_dict)

    def remove_illegal(self,input):
        line = input.strip().replace("<br>", "")
        line, _ = re.subn('【', '', line)
        line, _ = re.subn('】', '', line)
        line, _ = re.subn(r'/:[a-z]+', '', line)
        line, _ = re.subn(r'%[0-9A-Z]+', '', line)
        line, _ = re.subn(r' +', ' ', line)
        line, _ = re.subn(r'nan', '', line)
        line, _ = re.subn("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：]+",' ',line)
        return line

    def seg_stopword_sentence(self,sentence):
        line = self.remove_illegal(sentence)
        res = [x.word  for x in psg.cut(line) if x.flag.startswith('n')]
        return res
        # res = [(x.word, x.flag) for x in psg.cut(line) if x.flag.startswith('n')]
        # if len(res)>0:
        #     return res
        # else:
        #     return None
        # return list(jieba.cut(line))

    def cut_df(self,df,col='content'):
        return df[col].apply(self.seg_stopword_sentence)

    def cut(self,texts):
        return [self.seg_stopword_sentence(doc) for doc in tqdm(texts,desc='Seg documents') ]


class LDA_by_sklearn():
    def __init__(self, stopwords_path = '',texts=None):
        self.stopwords = codecs.open(stopwords_path, 'r', encoding='utf-8')
        self.stopwords = [w.strip() for w in self.stopwords]

        self.lda = None
        self.tf_vectorizer = None

    def run_lda(self, n_topics=5):
        self.tf_vectorizer = CountVectorizer(strip_accents='unicode', stop_words=self.stopwords)
        tf = self.tf_vectorizer.fit_transform(open("/home/tanglek/dataset/corpus_seg_100k.txt", 'r', encoding='utf-8'))
        self.lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=100,learning_method='online', learning_offset=50, random_state=999)
        self.lda.fit(tf)

    def print_top_words(self, n_top_words=10):
        tf_feature_names = self.tf_vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(self.lda.components_):
            print('Topic #{0}:'.format(str(topic_idx)))
            print(' '.join([tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
            print()

class LDA_by_gensim():
    def __init__(self, stopwords_path = '',texts=None):
        self.stopwords = codecs.open(stopwords_path, 'r', encoding='utf-8')
        self.stopwords = [w.strip() for w in self.stopwords]

        self.train_set = texts

    def run_lda(self, n_topics=10):
        self.n_topics = n_topics
        dictionary = Dictionary(self.train_set)
        corpus = [dictionary.doc2bow(text) for text in self.train_set]

        self.lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics)

    def print_top_words(self, n_top_words=10):
        pprint(self.lda.show_topics(num_topics=self.n_topics, num_words=n_top_words, formatted=False))
        # for topic in self.lda.print_topics(20):
        #     print(topic)

class ChnTfidfLDAModel():
    def __init__(self,model_name,train_texts,test_texts):
        self.model_name = model_name
        self.train_texts = train_texts
        self.test_texts = test_texts
        # self.num_topics = [5]
        self.num_topics = [5, 10, 15, 20]
        # topicnums = [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.no_below_this_number = 50
        self.no_above_fraction_of_doc = 0.2
        self.corpus = None
        self.model = None
        self.ldamodels_tfidf = {}
        self.ldamodels_eval = {}
        self.all_topics = None

    # http://radimrehurek.com/topic_modeling_tutorial/2%20-%20Topic%20Modeling.html
    def intra_inter_tfidf(self,lda_model, dictionary, test_docs, num_pairs=10000):
        # Split each test document into two halves and compute topics for each half
        part1 = [lda_model[self.tfidf[dictionary.doc2bow(tokens[:int(list(tokens) / 2)])]] for tokens in test_docs]
        part2 = [lda_model[self.tfidf[dictionary.doc2bow(tokens[int(list(tokens) / 2):])]] for tokens in test_docs]
        # Compute topic distribution similarities using cosine similarity
        # print("Average cosine similarity between corresponding parts (higher is better):")
        corresp_parts = np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip(part1, part2)])
        # print("Average cosine similarity between 10,000 random parts (lower is better):")
        random.seed(42)
        random_pairs = np.random.randint(0, len(test_docs), size=(num_pairs, 2))
        random_parts = np.mean([gensim.matutils.cossim(part1[i[0]], part2[i[1]]) for i in random_pairs])
        return corresp_parts, random_parts

    def fit(self,df,col="content"):
        pass

    def print_features(clf, vocab, n=10):
        """ Print sorted list of non-zero features/weights. """
        coef = clf.coef_[0]
        print('positive features: %s' % (
        ' '.join(['%s/%.2f' % (vocab[j], coef[j]) for j in np.argsort(coef)[::-1][:n] if coef[j] > 0])))
        print('negative features: %s' % (
        ' '.join(['%s/%.2f' % (vocab[j], coef[j]) for j in np.argsort(coef)[:n] if coef[j] < 0])))

    def fit_classifier(X, y, C=0.1):
        """ Fit L1 Logistic Regression classifier. """
        # Smaller C means fewer features selected.
        clf = LogisticRegression(penalty='l1', C=C)
        clf.fit(X, y)
        return clf

    def fit_lda(X, vocab, num_topics=5, passes=20):
        """ Fit LDA from a scipy CSR matrix (X). """
        print('fitting lda...')
        return LdaModel(matutils.Sparse2Corpus(X), num_topics=num_topics,
                        passes=passes,
                        id2word=dict([(i, s) for i, s in enumerate(vocab)]))

    def jaccard_similarity(query, document):
        intersection = set(query).intersection(set(document))
        union = set(query).union(set(document))
        return float(len(intersection)) / float(len(union))

    def eval(self):
        dictionary = corpora.Dictionary(self.train_texts)
        dictionary.filter_extremes(no_below=self.no_below_this_number, no_above=self.no_above_fraction_of_doc)
        self.corpus = [dictionary.doc2bow(text) for text in self.train_texts]
        self.tfidf = models.TfidfModel(self.corpus)
        corpus_tfidf = self.tfidf[self.corpus]
        for i in tqdm(self.num_topics, desc='num of topics'):
            random.seed(42)
            self.ldamodels_tfidf[i] =LdaModel(corpus_tfidf, num_topics=i, id2word=dictionary)
            self.ldamodels_tfidf[i].save('./data/{0}_ldamodels_tfidf_{1}.lda'.format(self.model_name,str(i)))
            for j in range(i):
                print('Topic {} : {}'.format (str(j) , self.ldamodels_tfidf[i].print_topic(j)))

        self.n_topic = self.num_topics[1]
        self.model = self.ldamodels_tfidf[self.num_topics[1]]


        # for i in tqdm(self.topicnums, desc='num of topics'):
        #     lda_model =LdaModel.load('./data/ldamodels_tfidf_' + str(i) + '.lda')
        #     self.ldamodels_eval[i] = self.intra_inter_tfidf(lda_model, dictionary, self.test_texts)
        #
        # pickle.dump(self.ldamodels_eval, open('./data/pub_ldamodels_tfidf_eval.pkl', 'wb'))

    def visualize_eval(self):
        # ldamodels_eval = pickle.load(open('./data/pub_ldamodels_tfidf_eval.pkl', 'rb'))
        corresp_parts = [self.ldamodels_eval[i][0] for i in self.num_topics]
        random_parts = [self.ldamodels_eval[i][1] for i in self.num_topics]
        sns.set_context("poster")
        with sns.axes_style("darkgrid"):
            x = self.num_topics
            y1 = corresp_parts
            y2 = random_parts
            plt.plot(x, y1, label='Parts from same article')
            plt.plot(x, y2, label='Parts from random articles')
            plt.ylim([0.0, 1.0])
            plt.xlabel('Number of topics')
            plt.ylabel('Average cosine similarity')
            plt.legend()
            plt.show()

    def visualize_stability(self):
        lda_stability = pickle.load(open('./data/pub_lda_tfidf_stability.pkl', 'rb'))
        mean_stability = [np.array(lda_stability[i]).mean() for i in self.num_topics[:-1]]

        with sns.axes_style("darkgrid"):
            x = self.num_topics[:-1]
            y = mean_stability
            plt.plot(x, y, label='Mean overlap')
            plt.xlim([1, 100])
            plt.ylim([0, 1])
            plt.xlabel('Number of topics')
            plt.ylabel('Average Jaccard similarity')
            # plt.legend()
            plt.show()

    def print_top_words(self, feature_names, n_top_words):
        for topic_idx, topic in enumerate(self.model.components_):
            print("Topic #{}:".format(topic_idx))
            print(" ".join([feature_names[i]  for i in topic.argsort()[:-n_top_words - 1:-1]]))

    def print_topics(self,n=10):
        """ Print the top words for each topic. """
        for k,lda in self.ldamodels_tfidf.items():
            topics = lda.show_topics(num_topics=k,num_words=n, log=True,formatted=False)
            for ti, topic in enumerate(topics):
                print('{}:{}'.format(ti,topic))
                print('-'*60)
            #     print('topic %d: %s' % (ti, ' '.join('%s/%.2f' % (t[1], t[0]) for t in topic)))
            print("{}{} topics{}".format("="*25,k,"="*25))

    def get_topic_distribution(self):
        topics = self.model.get_document_topics(self.corpus, per_word_topics=True)
        self.all_topics = [(doc_topics, word_topics, word_phis) for doc_topics, word_topics, word_phis in topics]
        self.topic_dist = np.zeros([len(self.corpus), self.n_topic])
        for i,doc in enumerate(self.all_topics):
            for probs in doc[0]:
                (topic_idx,proba) = probs
                self.topic_dist[i,topic_idx] = proba

        return self.topic_dist

    def load_pretrained_model(self,model_path):
        return LdaModel.load(model_path, mmap='r')

def thucnews_lda():
    wordseg = WordSeg("/home/tanglek/opensource/stopwords/all_stopwords.txt")
    texts=[]
    i =0
    for filename in tqdm(glob.iglob("/home/tanglek/dataset/THUCNews/**/*.txt", recursive=True),'processing THUCNews'):
        # print(filename)
        with codecs.open(filename, 'r', encoding='utf-8') as f:
            line = f.read()
            texts.append(line)

    texts = wordseg.cut(texts)
    print("texts len:{}".format(len(texts)))
    # print(texts)
    model = ChnTfidfLDAModel('thucnews',texts,None)
    model.eval()

def ccfnews():
    data = pd.read_csv("~/dataset/ccf_news_rec/train.txt", sep='\t', header=None)
    data.columns = ['user_id', 'news_id', 'browse_time', 'title', 'content', 'published_at']
    data['title'] = data['title'].astype(str)
    data['content'] = data['content'].astype(str)
    wordseg = WordSeg("/home/tanglek/opensource/stopwords/all_stopwords.txt")
    df = data.drop_duplicates(['news_id'])
    texts=df['content'].tolist()
    texts = wordseg.cut(texts)
    # print(texts[0])
    print("texts size:{}".format(len(texts)))
    random.seed(42)
    train_set = random.sample(list(range(0, len(texts))), len(texts) - 1000)
    test_set = [x for x in list(range(0, len(texts))) if x not in train_set]

    train_texts = [texts[i] for i in train_set]
    test_texts = [texts[i] for i in test_set]
    model = ChnTfidfLDAModel('ccfnews',texts,texts)
    # model = ChnTfidfLDAModel('ccfnews',train_texts,test_texts)
    model.eval()
    topic_dist = model.get_topic_distribution()
    print(topic_dist)
    # model.print_topics()

def large_corpus_test():
    stop_words = [line.strip() for line in open("/home/tanglek/workspace/funlp/data/stop_words.txt", 'r', encoding='utf-8').readlines()]
    texts = []
    with codecs.open("/home/tanglek/dataset/corpus_seg_100k.txt",'r',encoding='utf-8') as f:
        for line in f.readlines():
            outstr = []
            for word in line.strip("\n").split(" "):
                if word not in stop_words:
                    if word != '\t':
                        outstr.append(word)
            texts.append(outstr)

    print("texts size:{}".format(len(texts)))
    # texts = wordseg.cut(texts)
    model = LDA_by_sklearn("/home/tanglek/workspace/funlp/data/stop_words.txt",texts)
    model.run_lda()
    model.print_top_words()

def ccfnews_cf_topic_regression():
    data = pd.read_csv("~/dataset/ccf_news_rec/train.txt", sep='\t', header=None)
    data.columns = ['user_id', 'news_id', 'browse_time', 'title', 'content', 'published_at']
    data['title'] = data['title'].astype(str)
    data['content'] = data['content'].astype(str)
    #pre trained lda model
    wordseg = WordSeg("/home/tanglek/opensource/stopwords/all_stopwords.txt")
    df = data.drop_duplicates(['news_id'])
    texts=df['content'].tolist()
    texts = wordseg.cut(texts)
    # print(texts[0])
    print("texts size:{}".format(len(texts)))
    random.seed(42)
    train_set = random.sample(list(range(0, len(texts))), len(texts) - 1000)
    test_set = [x for x in list(range(0, len(texts))) if x not in train_set]

    train_texts = [texts[i] for i in train_set]
    test_texts = [texts[i] for i in test_set]
    model = ChnTfidfLDAModel('ccfnews',texts,texts)
    # model = ChnTfidfLDAModel(train_texts,test_texts)
    model.eval()
    #train ctr model
    num_topics = 10
    n_voca = 100000
    n_users= len(data['user_id'].unique())
    n_items = len(data['news_id'].unique())
    doc_ids = range(len(data['news_id'].unique()))
    doc_cnts = n_items
    data['rating'] = int(1)
    data = data.sort_values(['browse_time'], ascending=True).groupby(["user_id", "news_id"]).last().reset_index()
    ratings = data.pivot(index='user_id', columns='news_id', values='rating')
    ratings.fillna(0,inplace=True)
    rt=ratings.values.astype(int)
    print('Start creating model...')
    CTR = CollaborativeTopicModel(lda=model,n_topic = num_topics,n_voca= n_voca, n_user=n_users, n_item=n_items, doc_ids=doc_ids, doc_cnt=doc_cnts,ratings=rt)
    print('Start fiting model...')
    CTR.fit(doc_ids=None, doc_cnt=None, rating_matrix=None,max_iter=100)
    print ('Testing')

if __name__ == '__main__':
    thucnews_lda()
    # ccfnews_cf_topic_regression()
    # ccfnews()