from qdrant_client import QdrantClient

client = QdrantClient(url='http://qdrant:6333')
points = client.scroll(collection_name='knowledge_base', limit=10)
for point in points[0]:
    print(point.payload)
