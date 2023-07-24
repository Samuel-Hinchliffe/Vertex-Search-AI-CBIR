
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models
import sys
import numpy as np
import json
from pymilvus import connections, db, CollectionSchema, FieldSchema, DataType, Collection, utility 

# m
conn = connections.connect(host="127.0.0.1", port=19530)
client = QdrantClient("localhost", port=6333)

# Qdrant
x = utility.list_collections()
t = client.get_collection(collection_name="vector_db")


test = np.load('./static/cache/630503-6179461-large.npy')

query_vector = test
hits = client.search(
    collection_name="vector_db",
    query_vector=query_vector,
    with_vectors=False,
    limit=10000000,
)

print('QDRANT:')
print(len(hits))


collection = Collection("image_vectors")  
collection.load()
res = collection.query(
  expr = "id != 1",
  output_fields=["id"],
  offset = 0,
)

print('MILV')
print(len(res))
