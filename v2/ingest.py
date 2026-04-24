from ingestion import update_vector_store, build_topics_json

if __name__ == "__main__":
    update_vector_store(force_reload=True)
    build_topics_json()