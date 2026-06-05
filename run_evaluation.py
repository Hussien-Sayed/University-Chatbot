import os
from pathlib import Path
from dotenv import load_dotenv
from src.data_utils.testset_utils import generate_testset, load_testset
from src.llm.embedding_api import EmbeddingAPI
from src.llm.llm_api import LLMAPI
from src.rag.retriever.rag_retriever import RAGRetriever
from src.rag.evaluation.rag_evaluator import RAGEvaluator

def main():
    load_dotenv()
    data_source_path = os.getenv("DATA_SOURCE_PATH", "data/intents-v2.json")
    test_data_path = Path(os.getenv("RAG_TEST_DATA_PATH", "data/rag_test_data.json"))
    test_set_ratio = float(os.getenv("RAG_TEST_SET_RATIO", "0.2"))
    refined_data_path = os.getenv("RAG_REFINED_DATA_PATH")
    experiment_name = os.getenv("RAG_EXPERIMENT_NAME", "baseline")
    experiment_results_dir = os.getenv("RAG_EXPERIMENTS_DIR", "data/evaluation_results")
    
    # 1. Initialize core components
    print("Initializing components...")
    embedding_api = EmbeddingAPI()
    llm_api = LLMAPI()
    
    retriever = RAGRetriever(
        vector_db_path=os.getenv("VDB_SAVE_PATH", "data/vector_db"),
        llm_api=llm_api
    )
    
    # 2. Initialize Evaluator
    evaluator = RAGEvaluator(
        retriever=retriever,
        embedding_api=embedding_api,
        data_source_path=data_source_path,
        test_data_path=str(test_data_path),
        experiment_results_dir=experiment_results_dir
    )

    if test_data_path.exists():
        test_samples = load_testset(test_data_path)
    else:
        test_samples = generate_testset(
            data_source_path,
            test_set_ratio=test_set_ratio,
            test_data_path=test_data_path,
            refined_data_path=Path(refined_data_path) if refined_data_path else None
        )
    
    # 3. Run evaluation
    print("Starting evaluation...")
    results = evaluator.run_evaluation(test_samples, experiment_name=experiment_name)
    
    print("\n" + "="*40)
    print("RAGAS Evaluation Results")
    print("="*40)
    print(results)

if __name__ == "__main__":
    main()
