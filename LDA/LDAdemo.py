import jieba
from docx import Document
import os
import numpy as np
from gensim import corpora, models
from gensim.models.ldamodel import LdaModel

### 分词 ###
# 读取本地停用词表
stopwords = set(open('cn_stopwords.txt', 'r', encoding='utf-8').read().splitlines())


### 文档读取和处理 ###
def read_text_from_docx(file_path):
    """读取.docx文件并返回文本内容。"""
    document = Document(file_path)
    return "\n".join(para.text for para in document.paragraphs)


def preprocess_chinese_text(text):
    """使用jieba进行中文分词，并过滤停用词。"""
    words = jieba.cut_for_search(text)
    return [word for word in words if word not in stopwords]


def process_documents(directory):
    """处理目录下的所有.docx文件，并返回分词结果列表。"""
    documents = []
    documents_name = []
    for filename in os.listdir(directory):
        if filename.endswith(".docx"):
            file_path = os.path.join(directory, filename)
            text = read_text_from_docx(file_path)
            segmented_text = preprocess_chinese_text(text)
            documents.append(segmented_text)
            documents_name.append(filename)
    return documents, documents_name


### LDA建模 ###
def prepare_corpus(documents):
    """准备语料库和词典，用于LDA模型。"""
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(text) for text in documents]
    return dictionary, corpus


def lda_model(corpus, dictionary, num_topics=5, random_state=42):
    """训练LDA模型。"""
    return LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=30, random_state=random_state)


### 文档主题分布获取 ###
def get_topic_distribution(lda, corpus):
    """获取每个文档的主题分布。"""
    return [lda.get_document_topics(doc, minimum_probability=0.0) for doc in corpus]


### 处理用户查询并获取查询的主题分布 ###
def preprocess_query(query):
    """对用户查询进行预处理并获取主题分布。"""
    words = jieba.cut_for_search(query)
    query_bow = dictionary.doc2bow([word for word in words if word not in stopwords])
    return lda.get_document_topics(query_bow, minimum_probability=0.0)


### 计算查询与文档之间的主题距离 ###
def calculate_query_document_distance(query_topics, doc_topics):
    """计算查询与一个文档之间的主题距离。"""
    query_dist = dict(query_topics)
    doc_dist = dict(doc_topics)
    all_topics = set(query_dist.keys()).union(set(doc_dist.keys()))
    return sum(abs(query_dist.get(topic, 0) - doc_dist.get(topic, 0)) for topic in all_topics)


### 归一化距离 ###
def normalize_distances(distances):
    """将一系列距离归一化到0-1的区间。"""
    max_distance = max(distances) if distances else 0
    if max_distance == 0:  # 防止除以零
        return [0] * len(distances)
    return [1 - (dist / max_distance) for dist in distances]  # 距离越小，相似度越高


def coherence(num_topics, documents):
    lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=30, random_state=42)
    coherence_model_lda = models.CoherenceModel(model=lda, texts=documents, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    return coherence_lda


def combine_scores(similarities, ctr, alpha=0.7):
    """结合主题相似度和点击率来计算综合得分。
    :param similarities: 文档与查询的相似度列表。
    :param ctr: 每个文档的点击率列表。
    :param alpha: 相似度与点击率的权重系数。
    :return: 综合得分列表。
    """
    if len(similarities) != len(ctr):
        raise ValueError("The length of similarities and ctr must be the same.")
    return [alpha * sim + (1 - alpha) * click for sim, click in zip(similarities, ctr)]


### 主执行逻辑 ###
if __name__ == '__main__':
    directory = "技术1"
    documents, document_names = process_documents(directory)
    dictionary, corpus = prepare_corpus(documents)
    # x = range(10, 15)
    # y = [coherence(i, documents) for i in x]
    # lda = lda_model(corpus, dictionary, num_topics=max(zip(x,y),key=lambda pair:pair[1]) [0])
    # lda.save("lda_model1")
    lda = LdaModel.load("lda_model1")
    doc_topics = get_topic_distribution(lda, corpus)

    # 假设点击数据
    clicks = np.array([5, 10, 15, 20, 25, 5, 10, 15, 20, 25, 5, 10, 15, 20, 25, 5, 10, 15, 20, 25])
    max_clicks = np.max(clicks)
    ctr = clicks / max_clicks  # 归一化点击率

    # 假设有用户查询，获取用户查询主题分布
    user_query = "技术"
    query_topics = preprocess_query(user_query)

    # 计算查询与每个文档之间的主题距离
    doc_distances = [calculate_query_document_distance(query_topics, doc) for doc in doc_topics]

    # 归一化距离
    normalized_distances = normalize_distances(doc_distances)

    # 结合点击率和主题相似度
    combined_scores = combine_scores(normalized_distances, ctr, alpha=0.7)

    # 排序文档基于综合得分
    sorted_doc_distances = sorted(zip(document_names, combined_scores), key=lambda x: x[1], reverse=True)

    # 打印排序后的文档和相似度
    for doc_name, similarity in sorted_doc_distances:
        print(f"Document: {doc_name}, Similarity: {similarity:.4f}")
