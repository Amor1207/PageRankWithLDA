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


# 计算两个两个主题分布之间的曼哈顿距离
def calculate_topic_distance_abs_diff(doc_topics_i, doc_topics_j):
    # 保证每个主题的概率被考虑到，即使某些主题在某文档中的概率为0
    # 将主题分布转换为字典形式
    topic_dist_i = dict(doc_topics_i)
    topic_dist_j = dict(doc_topics_j)

    # 获取所有主题的并集
    all_topics = set(topic_dist_i.keys()).union(set(topic_dist_j.keys()))
    distance = sum(abs(topic_dist_i.get(topic, 0) - topic_dist_j.get(topic, 0)) for topic in all_topics)
    return distance


# 计算所有文档之间的主题距离
def calculate_all_topic_distances(doc_topics):
    num_docs = len(doc_topics)
    distances = [[0] * num_docs for _ in range(num_docs)]
    for i in range(num_docs):
        for j in range(i+1, num_docs):
            distance = calculate_topic_distance_abs_diff(doc_topics[i], doc_topics[j])
            distances[i][j] = distances[j][i] = distance
    return distances


### 处理用户查询并获取查询的主题分布 ###
def preprocess_query(query, lda, dictionary):
    query_bow = dictionary.doc2bow(jieba.cut_for_search(query))
    query_topics = lda.get_document_topics(query_bow, minimum_probability=0.0)
    return dict(query_topics)


# 将主题距离转换为相似度
# 阈值threshold用于确定两个文档之间是否存在链接，按需调整
def convert_distance_to_similarity(distances, threshold=0.1):
    max_distance = max(max(row) for row in distances if row)
    similarity_matrix = []
    links = []
    for i, row in enumerate(distances):
        new_row = []
        link_row = []
        for j, dist in enumerate(row):
            similarity = 1 - (dist / max_distance) if max_distance else 1
            new_row.append(similarity)
            if similarity > threshold and i != j:
                link_row.append(j)
        similarity_matrix.append(new_row)
        links.append(link_row)
    return similarity_matrix, links


def calculate_query_similarity(query_topics, doc_topics):
    num_docs = len(doc_topics)
    distances = [[0] for _ in range(num_docs)]
    for i in range(num_docs):
        distance = calculate_topic_distance_abs_diff(query_topics, doc_topics[i])
        distances[i] = distance
    max_distance = max(distances) if distances else 0
    query_similarity = [1 - (dist / max_distance) if max_distance else 1 for dist in distances]
    return query_similarity


def adjust_link_weights(doc_topics, query_similarity, links, base_weight=0.01):
    adjusted_weights = []
    for i, topics in enumerate(doc_topics):
        doc_similarity = query_similarity[i]
        # 将基础权重添加到与查询相关的文档上
        row_weights = [base_weight + doc_similarity if j in links[i] else 0 for j in range(len(doc_topics))]
        adjusted_weights.append(row_weights)
    return adjusted_weights


def coherence(num_topics, documents):
    lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=30, random_state=42)
    coherence_model_lda = models.CoherenceModel(model=lda, texts=documents, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    return coherence_lda


# 计算PageRank
def page_rank_with_ctr(links, similarities, adjusted_weights, ctr, alpha=0.85, beta = 0.95, convergence_threshold=0.0001):
    N = len(links)
    pr = np.ones(N) / N  # 初始化PR值，总和为1
    change = 1
    w1 = 0.6
    w2 = 0.4
    # 进行迭代直到收敛
    while change > convergence_threshold:
        new_pr = np.zeros(N)
        for i in range(N):
            link_contributions = 0
            for j in links[i]:  # 遍历节点i的所有出链节点j
                if len(links[j]) > 0:  # 避免除以零
                    # 结合 adjusted_weights 和 similarities 作为权重
                    link_weight = w1 * adjusted_weights[j][i] + w2 * similarities[j][i]
                    link_contributions += pr[j] * link_weight / len(links[j])
            new_pr[i] = (1 - alpha) / N + alpha * (beta*link_contributions + (1-beta)*ctr[i])
        # 归一化新的PageRank值，确保它们的总和为1
        new_pr /= np.sum(new_pr)  # 归一化步骤
        change = np.linalg.norm(new_pr - pr)
        pr = new_pr
    return pr



### 主执行逻辑 ###
if __name__ == '__main__':
    directory = "技术1"
    documents, document_names = process_documents(directory)
    dictionary, corpus = prepare_corpus(documents)

    # 第一次运行需要训练LDA模型
    # x = range(5, 15)
    # y = [coherence(i, documents) for i in x]
    # lda = lda_model(corpus, dictionary, num_topics=max(zip(x,y),key=lambda pair:pair[1]) [0])
    # lda.save("lda_model1")

    # 之后可以直接加载已经训练好的模型
    lda = LdaModel.load("lda_model1")
    doc_topics = get_topic_distribution(lda, corpus)

    # 假设点击数据
    clicks = np.array([5, 10, 15, 20, 25, 5, 10, 15, 20, 25, 5, 10, 15, 20, 25, 5, 10, 15, 20, 25])
    max_clicks = np.max(clicks)
    ctr = clicks / max_clicks  # 归一化点击率

    # 假设有用户查询，获取用户查询主题分布
    query = "技术"
    query_topics = preprocess_query(query, lda, dictionary)

    query_similarity = calculate_query_similarity(query_topics, doc_topics)

    # 计算查询与每个文档之间的主题距离
    distances = calculate_all_topic_distances(doc_topics)

    similarities, links = convert_distance_to_similarity(distances)

    adjusted_weights = adjust_link_weights(doc_topics, query_similarity, links)

    pr = page_rank_with_ctr(links, similarities, adjusted_weights, ctr)

    # 按PageRank值对文档进行排序
    pagerank_score = pr
    doc_pagerank = list(zip(document_names, pagerank_score))

    # 按PageRank分数降序排序
    sorted_doc_pagerank = sorted(doc_pagerank, key=lambda x: x[1], reverse=True)

    # 打印排序后的结果
    for doc_name, pr_score in sorted_doc_pagerank:
        print(f"Document: {doc_name}, PageRank Score: {pr_score:.5f}")
