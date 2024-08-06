import grpc
from concurrent import futures
import document_service_pb2
import document_service_pb2_grpc
import jieba
from docx import Document
import os
import numpy as np
from gensim import corpora
from gensim.models.ldamodel import LdaModel

# 导入先前的文档处理代码
# 假设你的文档处理代码已经被定义在一个名为 document_analysis 的模块中
### 分词 ###
# 读取本地停用词表
stopwords = set(open('C:/pythonProject/PageRank/cn_stopwords.txt', 'r', encoding='utf-8').read().splitlines())

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

### 文档之间的主题距离计算 ###
def calculate_topic_distance_abs_diff(doc_topics_i, doc_topics_j):
    """计算两个文档之间的主题绝对距离。"""
    topic_dist_i = dict(doc_topics_i)
    topic_dist_j = dict(doc_topics_j)
    all_topics = set(topic_dist_i.keys()).union(set(topic_dist_j.keys()))
    return sum(abs(topic_dist_i.get(topic, 0) - topic_dist_j.get(topic, 0)) for topic in all_topics)

def calculate_all_topic_distances(doc_topics):
    """计算所有文档之间的主题距离。"""
    num_docs = len(doc_topics)
    distances = [[0] * num_docs for _ in range(num_docs)]
    for i in range(num_docs):
        for j in range(i + 1, num_docs):
            distance = calculate_topic_distance_abs_diff(doc_topics[i], doc_topics[j])
            distances[i][j] = distances[j][i] = distance
    return distances

### PageRank和点击率分析 ###
def convert_distance_to_similarity(distances, threshold=0.1):
    """将距离转换为相似度，并基于相似度构建链接。"""
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

def page_rank_with_ctr(links, similarities, ctr, alpha=0.85, beta=0.7, convergence_threshold=0.0001):
    """结合PageRank和点击率（CTR）计算文档排序。"""
    N = len(links)
    pr = np.ones(N) / N
    change = 1
    while change > convergence_threshold:
        new_pr = np.zeros(N)
        for i in range(N):
            link_contributions = sum(pr[j] * similarities[i][j] / len(links[j]) for j in links[i] if links[j])
            new_pr[i] = (1 - alpha) / N + alpha * (beta * link_contributions + (1 - beta) * ctr[i])
        change = np.linalg.norm(new_pr - pr)
        pr = new_pr
    return pr

class DocumentProcessorServicer(document_service_pb2_grpc.DocumentProcessorServicer):
    def ProcessDocuments(self, request, context):
        directory = request.directory
        documents, document_names = process_documents(directory)
        dictionary, corpus = prepare_corpus(documents)
        lda = lda_model(corpus, dictionary, num_topics=5)
        doc_topics = get_topic_distribution(lda, corpus)
        distances = calculate_all_topic_distances(doc_topics)
        similarities, links = convert_distance_to_similarity(distances)
        clicks = np.array(request.clicks)  # 使用从客户端接收的点击数组
        max_clicks = np.max(clicks)
        ctr = clicks / max_clicks
        pr_with_ctr = page_rank_with_ctr(links, similarities, ctr)
        doc_pagerank = list(zip(document_names, pr_with_ctr))
        sorted_doc_pagerank = sorted(doc_pagerank, key=lambda x: x[1], reverse=True)
        response_list = document_service_pb2.DocumentList()
        for doc_name, pr_score in sorted_doc_pagerank:
            response_list.documents.add(documentName=doc_name, pageRankScore=pr_score)
        return response_list


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    document_service_pb2_grpc.add_DocumentProcessorServicer_to_server(DocumentProcessorServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("gRPC starting")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
