import grpc
import document_service_pb2
import document_service_pb2_grpc

def get_document_rankings(directory):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = document_service_pb2_grpc.DocumentProcessorStub(channel)
        response = stub.ProcessDocuments(document_service_pb2.DocumentRequest(directory=directory))
        for document in response.documents:
            print(f"Document: {document.documentName}, PageRank Score: {document.pageRankScore}")

if __name__ == '__main__':
    get_document_rankings('C:/pythonProject/PageRank/测试文档')
