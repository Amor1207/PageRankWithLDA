# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import pagerank_service_pb2 as pagerank__service__pb2

GRPC_GENERATED_VERSION = '1.65.0'
GRPC_VERSION = grpc.__version__
EXPECTED_ERROR_RELEASE = '1.65.0'
SCHEDULED_RELEASE_DATE = 'June 25, 2024'
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    warnings.warn(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in pagerank_service_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
        + f' This warning will become an error in {EXPECTED_ERROR_RELEASE},'
        + f' scheduled for release on {SCHEDULED_RELEASE_DATE}.',
        RuntimeWarning
    )


class DocumentProcessorStub(object):
    """定义服务接口
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ProcessDocuments = channel.unary_unary(
                '/documentprocessor.DocumentProcessor/ProcessDocuments',
                request_serializer=pagerank__service__pb2.DocumentRequest.SerializeToString,
                response_deserializer=pagerank__service__pb2.DocumentList.FromString,
                _registered_method=True)


class DocumentProcessorServicer(object):
    """定义服务接口
    """

    def ProcessDocuments(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_DocumentProcessorServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ProcessDocuments': grpc.unary_unary_rpc_method_handler(
                    servicer.ProcessDocuments,
                    request_deserializer=pagerank__service__pb2.DocumentRequest.FromString,
                    response_serializer=pagerank__service__pb2.DocumentList.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'documentprocessor.DocumentProcessor', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('documentprocessor.DocumentProcessor', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class DocumentProcessor(object):
    """定义服务接口
    """

    @staticmethod
    def ProcessDocuments(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/documentprocessor.DocumentProcessor/ProcessDocuments',
            pagerank__service__pb2.DocumentRequest.SerializeToString,
            pagerank__service__pb2.DocumentList.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
