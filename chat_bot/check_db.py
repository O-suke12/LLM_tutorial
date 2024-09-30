# この場合も OPENAI_API_KEY を利用するので、設定を忘れないようにしてください。
import qdrant_client
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant

client = qdrant_client.QdrantClient(path="./local_qdrant")
qdrant = Qdrant(
    client=client, collection_name="my_collection", embeddings=OpenAIEmbeddings()
)
query = "トヨタとの決算について"
docs = qdrant.similarity_search(query=query, k=2)
for i in docs:
    print({"content": i.page_content, "metadata": i.metadata})
