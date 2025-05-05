# conda init
# conda activate medgraph
cd app
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
uvicorn backend.api.main:app --reload
#streamlit run streamlit/app.py
#docker pull qdrant/qdrant