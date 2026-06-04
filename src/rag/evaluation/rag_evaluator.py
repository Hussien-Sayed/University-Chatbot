import os
from typing import List, Dict, Any, Optional
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

from src.llm.embedding_api import EmbeddingAPI
from src.llm.llm_api import LLMAPI
from src.rag.retriever.rag_retriever import RAGRetriever

class RAGEvaluator:
    """Class for evaluating RAG pipeline performance using RAGAS metrics"""

    def __init__(
        self,
        retriever: RAGRetriever,
        embedding_api: EmbeddingAPI,
        data_source_path: Optional[str] = None,
        test_data_path: Optional[str] = None,
        llm_model: str = "llama-3.1-8b-instant",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.retriever = retriever
        self.embedding_api = embedding_api
        self.data_source_path = data_source_path
        self.test_data_path = test_data_path
        self.llm_model = llm_model
        self.embedding_model = embedding_model

    def run_evaluation(self, test_samples: List[Dict[str, str]]) -> Dict[str, Any]:
        """Run the RAGAS evaluation process"""
        if not test_samples:
            return {"error": "No test samples generated"}

        questions = []
        answers = []
        contexts = []
        ground_truths = []

        print(f"Running queries for {len(test_samples)} samples...")
        for sample in tqdm(test_samples, desc="Running RAG queries"):
            q = sample["question"]
            
            # Generate embedding and retrieve
            q_emb = self.embedding_api.generate_embedding(q)
            retrieved_chunks = self.retriever.retrieve_chunks(q_emb)
            ctx = [item['chunk'].get('content', '') for item in retrieved_chunks]
            
            # Generate response
            ans = self.retriever.generate_response(q, q_emb)
            
            questions.append(q)
            answers.append(ans)
            contexts.append(ctx)
            ground_truths.append(sample["ground_truth"])

        # Create RAGAS dataset
        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })

        # Configure evaluator models
        eval_llm = ChatGroq(
            model_name=self.llm_model,
            api_key=os.getenv("GROQ_API_KEY")
        )
        eval_embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model
        )

        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]

        print("Evaluating with RAGAS metrics...")
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=eval_llm,
            embeddings=eval_embeddings
        )

        return result
