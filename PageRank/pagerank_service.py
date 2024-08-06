import grpc
from concurrent import futures
import pagerank_service_pb2
import pagerank_service_pb2_grpc

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


def coherence(num_topics, documents, corpus, dictionary):
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


class DocumentProcessorServicer(pagerank_service_pb2_grpc.DocumentProcessorServicer):
    def ProcessDocuments(self, request, context):
        directory = request.directory
        query = request.query  # 获取查询词
        documents, document_names = process_documents(directory)
        dictionary, corpus = prepare_corpus(documents)
        lda = LdaModel.load("lda_model1")
        doc_topics = get_topic_distribution(lda, corpus)
        distances = calculate_all_topic_distances(doc_topics)
        similarities, links = convert_distance_to_similarity(distances)
        query_topics = preprocess_query(query, lda, dictionary)
        query_similarity = calculate_query_similarity(query_topics, doc_topics)

        # 使用查询相关性和文档相似性调整权重
        adjusted_weights = adjust_link_weights(doc_topics, query_similarity, links)

        clicks = np.array(request.clicks)  # 使用从客户端接收的点击数组
        max_clicks = np.max(clicks)
        ctr = clicks / max_clicks
        pr_with_ctr = page_rank_with_ctr(links, similarities, adjusted_weights, ctr)
        doc_pagerank = list(zip(document_names, pr_with_ctr))
        sorted_doc_pagerank = sorted(doc_pagerank, key=lambda x: x[1], reverse=True)

        response_list = pagerank_service_pb2.DocumentList()
        for doc_name, pr_score in sorted_doc_pagerank:
            response_list.documents.add(documentName=doc_name, pageRankScore=pr_score)
        return response_list


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pagerank_service_pb2_grpc.add_DocumentProcessorServicer_to_server(DocumentProcessorServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("gRPC starting")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()