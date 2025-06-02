# cache/serializers.py
"""Cache serialization implementations."""

import json
import pickle
import zlib
from typing import Any
from .core import Serializer

class JSONSerializer(Serializer):
    """JSON serializer for cache values."""
    
    def serialize(self, data: Any) -> bytes:
        """Serialize data to JSON bytes."""
        return json.dumps(data, default=str).encode('utf-8')
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize JSON bytes to data."""
        return json.loads(data.decode('utf-8'))

class PickleSerializer(Serializer):
    """Pickle serializer for cache values."""
    
    def serialize(self, data: Any) -> bytes:
        """Serialize data to pickle bytes."""
        return pickle.dumps(data)
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize pickle bytes to data."""
        return pickle.loads(data)

class CompressedSerializer(Serializer):
    """Compressed serializer wrapper."""
    
    def __init__(self, base_serializer: Serializer, compression_level: int = 6):
        self.base_serializer = base_serializer
        self.compression_level = compression_level
    
    def serialize(self, data: Any) -> bytes:
        """Serialize and compress data."""
        serialized = self.base_serializer.serialize(data)
        return zlib.compress(serialized, self.compression_level)
    
    def deserialize(self, data: bytes) -> Any:
        """Decompress and deserialize data."""
        decompressed = zlib.decompress(data)
        return self.base_serializer.deserialize(decompressed)