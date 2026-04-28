- src
    - RAG
        - vector db
            * class resposible for constucting the vector database from the embeddings. it should handle eveything from loading data, embedding, preprocessing, chunking, cleaning and normalization until the vector database is ready
            * can utilize the API embedding class, and data loading class
            * input: data loading class, API embedding class, vdb type, Chunking strategy (to be implemented later for now simply use fixed size), chunk size , vector database saving path should be configured in .env file
            * methods: split the long task into smaller methods for easy debugging, the final method needed is build vector db


        - retriever based LLM API
            * class for retrieving relevant chunks from the vector database and generating the final response
            * can utilize the API LLM class for final output generation
            * input: vector database path, number of chunks to retrieve, Retriever type (to be implemented later for now simply use cosine similarity)
            * methods: retrieve relevant chunks and generate response
    - LLM
        - API LLM
            * class for LLM API
            * input: model name, expected API in .env file
            * methods: generate response
        - API embedding
            * class for embedding model API
            * input: model name, expected API in .env file
            * methods: generate embedding
- data_utils
    - data_loading
        - class built specifically for out data source to extract our content
        - input: path to data source should be configured in .env file
        - methods: load content