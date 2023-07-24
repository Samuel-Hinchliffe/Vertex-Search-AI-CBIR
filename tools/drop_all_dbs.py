
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models
import sys
import numpy as np
import json
from pymilvus import connections, db, CollectionSchema, FieldSchema, DataType, Collection, utility 

# Drop m
conn = connections.connect(host="127.0.0.1", port=19530)
utility.drop_collection("image_vectors")

# Qdrant
client = QdrantClient("localhost", port=6333)
client.delete_collection(collection_name="vector_db")

# Double check (Expect error)
print(utility.list_collections())

try:
    print(client.get_collection(collection_name="vector_db"))
except Exception as e:
    print(e)
    print('All DBs dropped, looks good')