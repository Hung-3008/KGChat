# conda init
# conda activate medgraph
cd app
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
uvicorn backend.api.main:app --reload
#streamlit run streamlit/app.py
#docker pull qdrant/qdrant


# snapshot data 
# sudo neo4j stop
# sudo neo4j-admin database dump neo4j --to-path=/home/hung/Documents/hung/code/KG_Hung/KGChat/data/backup/
# sudo neo4j start

