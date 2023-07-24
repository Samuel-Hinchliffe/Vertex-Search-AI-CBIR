
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models
import sys
import numpy as np
import json
from pymilvus import connections, db, CollectionSchema, FieldSchema, DataType, Collection, utility 

# m
conn = connections.connect(host="127.0.0.1", port=19530)
obj_id = FieldSchema(
  name="id",
  dtype=DataType.INT64,
  is_primary=True,
   auto_id=True
)

file_name = FieldSchema(
  name="file_name",
  dtype=DataType.VARCHAR,
  max_length=999
)

product_url = FieldSchema(
  name="product_url",
  dtype=DataType.VARCHAR,
  max_length=999
)
product_image = FieldSchema(
  name="product_image",
  dtype=DataType.VARCHAR,
  max_length=999
)
tags = FieldSchema(
  name="tags",
  dtype=DataType.JSON,
)

vectorsSchema = FieldSchema(
  name="vector",
  dtype=DataType.FLOAT_VECTOR,
  dim=2048
)

schema = CollectionSchema(
  fields=[obj_id, file_name, product_url, product_image, tags, vectorsSchema],
  description="FCBP Vectors Collection",
  enable_dynamic_field=False,
  message="hi",
)

collection_name = "image_vectors"

collection = Collection(
    name=collection_name,
    schema=schema,
    using='default',
    shards_num=2
    )

# INDEX
collection = Collection("image_vectors")      # Get an existing collection.
index_params = {
  "metric_type":"L2",
  "index_type":"IVF_FLAT",
  "params":{"nlist":1024}
}
collection.create_index(
  field_name="vector", 
  index_params=index_params
)
utility.index_building_progress("image_vectors")

# Qdrant
client = QdrantClient("localhost", port=6333)
client.recreate_collection(
    collection_name="vector_db",
    vectors_config=VectorParams(size=2048, distance=Distance.EUCLID),
)


x = utility.list_collections()
t = client.get_collection(collection_name="vector_db")
print(t)
print(x)

