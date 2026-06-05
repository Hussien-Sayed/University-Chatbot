import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
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
        experiment_results_dir: Optional[str] = None,
        llm_model: str = "llama-3.1-8b-instant",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.retriever = retriever
        self.embedding_api = embedding_api
        self.data_source_path = data_source_path
        self.test_data_path = test_data_path
        self.experiment_results_dir = Path(experiment_results_dir or os.getenv("RAG_EXPERIMENTS_DIR", "data/evaluation_results"))
        self.llm_model = llm_model
        self.embedding_model = embedding_model

    def _safe_experiment_name(self, experiment_name: str) -> str:
        name = experiment_name.strip()
        if not name:
            raise ValueError("experiment_name cannot be empty")
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)

    def _json_safe(self, value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, dict):
            return {key: self._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._json_safe(item) for item in value]
        return value

    def _experiment_paths(self, experiment_name: str) -> Dict[str, Path]:
        safe_name = self._safe_experiment_name(experiment_name)
        self.experiment_results_dir.mkdir(parents=True, exist_ok=True)
        return {
            "summary": self.experiment_results_dir / f"{safe_name}.json",
            "samples_json": self.experiment_results_dir / f"{safe_name}_samples.json",
            "samples_csv": self.experiment_results_dir / f"{safe_name}_samples.csv",
            "failures": self.experiment_results_dir / f"{safe_name}_failures.json"
        }

    def _write_json(self, path: Path, data: Any) -> None:
        with open(path, "w") as f:
            json.dump(self._json_safe(data), f, indent=2)

    def _save_samples_csv(self, path: Path, samples: List[Dict[str, Any]]) -> None:
        base_fields = ["sample_index", "question", "answer", "contexts", "ground_truth"]
        extra_fields = sorted({
            key
            for sample in samples
            for key in sample.keys()
            if key not in base_fields
        })
        fieldnames = base_fields + extra_fields
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sample in samples:
                row = sample.copy()
                row["contexts"] = json.dumps(row.get("contexts", []), ensure_ascii=False)
                writer.writerow(row)

    def _load_existing_experiment(
        self,
        experiment_name: str
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        paths = self._experiment_paths(experiment_name)
        query_results = []
        failures = []

        if paths["samples_json"].exists():
            with open(paths["samples_json"], "r") as f:
                query_results = json.load(f)

        if paths["failures"].exists():
            with open(paths["failures"], "r") as f:
                failures = json.load(f)

        # Remove samples with NaN or missing required metric values so they get re-evaluated
        required_metrics = {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
        excluded_fields = {"sample_index", "question", "answer", "contexts", "ground_truth"}
        cleaned_query_results = []
        for sample in query_results:
            has_nan = any(
                isinstance(v, float) and (v != v)  # NaN check
                for k, v in sample.items()
                if k not in excluded_fields
            )
            missing_metrics = required_metrics - set(sample.keys())
            if not has_nan and not missing_metrics:
                cleaned_query_results.append(sample)

        # Clear failures so they get re-evaluated
        cleaned_failures = []

        return cleaned_query_results, cleaned_failures

    def _calculate_summary_scores(self, query_results: List[Dict[str, Any]]) -> Dict[str, float]:
        excluded_fields = {"sample_index", "question", "answer", "contexts", "ground_truth"}
        metric_names = sorted({
            key
            for sample in query_results
            for key, value in sample.items()
            if key not in excluded_fields and isinstance(value, (int, float))
        })
        summary_scores = {}

        for metric_name in metric_names:
            values = [
                sample[metric_name]
                for sample in query_results
                if isinstance(sample.get(metric_name), (int, float))
            ]
            if values:
                summary_scores[metric_name] = sum(values) / len(values)

        return summary_scores

    def _save_experiment_progress(
        self,
        experiment_name: str,
        query_results: List[Dict[str, Any]],
        failures: List[Dict[str, Any]],
        total_samples: int,
        status: str = "running",
        run_id: Optional[str] = None
    ) -> None:
        paths = self._experiment_paths(experiment_name)
        summary = {
            "experiment_name": experiment_name,
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "num_requested_samples": total_samples,
            "num_successful_samples": len(query_results),
            "num_failed_samples": len(failures),
            "scores": self._calculate_summary_scores(query_results)
        }

        self._write_json(paths["summary"], summary)
        self._write_json(paths["samples_json"], query_results)
        self._write_json(paths["failures"], failures)
        self._save_samples_csv(paths["samples_csv"], query_results)

    def _save_experiment_results(
        self,
        experiment_name: str,
        result: Any,
        failures: Optional[List[Dict[str, Any]]] = None,
        total_samples: Optional[int] = None
    ) -> None:
        paths = self._experiment_paths(experiment_name)
        failures = failures or []

        scores = self._json_safe(result.scores)
        metric_names = sorted({metric for score in scores for metric in score.keys()})
        summary_scores = {}

        for metric_name in metric_names:
            values = [
                score[metric_name]
                for score in scores
                if isinstance(score.get(metric_name), (int, float))
            ]
            if values:
                summary_scores[metric_name] = sum(values) / len(values)

        summary = {
            "experiment_name": experiment_name,
            "run_id": str(result.run_id) if result.run_id else None,
            "created_at": datetime.now().isoformat(),
            "status": "completed",
            "num_requested_samples": total_samples if total_samples is not None else len(scores) + len(failures),
            "num_successful_samples": len(scores),
            "num_failed_samples": len(failures),
            "scores": summary_scores
        }

        self._write_json(paths["summary"], summary)
        self._write_json(paths["failures"], failures)

        result_df = result.to_pandas()
        result_df.to_json(paths["samples_json"], orient="records", indent=2)
        result_df.to_csv(paths["samples_csv"], index=False)

        print(f"Saved experiment results to {self.experiment_results_dir}")

    def run_evaluation(self, test_samples: List[Dict[str, str]], experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """Run the RAGAS evaluation process"""
        if not test_samples:
            return {"error": "No test samples generated"}

        if experiment_name:
            query_results, failures = self._load_existing_experiment(experiment_name)
        else:
            query_results = []
            failures = []

        completed_sample_indexes = {
            item["sample_index"]
            for item in query_results + failures
            if "sample_index" in item
        }

        if experiment_name:
            self._save_experiment_progress(experiment_name, query_results, failures, len(test_samples))

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

        print(f"Running queries for {len(test_samples)} samples...")
        for sample_index, sample in enumerate(tqdm(test_samples, desc="Running RAG queries")):
            if sample_index in completed_sample_indexes:
                continue

            q = sample["question"]

            query_start_time = time.time()
            try:
                q_emb = self.embedding_api.generate_embedding(q)
                retrieved_chunks = self.retriever.retrieve_chunks(q_emb)
                ctx = [item['chunk'].get('content', '') for item in retrieved_chunks]
                ans = self.retriever.generate_response(q, q_emb)
                query_time_seconds = time.time() - query_start_time
            except Exception as exc:
                failures.append({
                    "sample_index": sample_index,
                    "stage": "query_generation",
                    "question": q,
                    "ground_truth": sample.get("ground_truth"),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)
                })
                if experiment_name:
                    self._save_experiment_progress(experiment_name, query_results, failures, len(test_samples))
                continue

            query_result = {
                "sample_index": sample_index,
                "question": q,
                "answer": ans,
                "contexts": ctx,
                "ground_truth": sample["ground_truth"],
                "query_time_seconds": query_time_seconds
            }

            if experiment_name:
                query_results.append(query_result)
                self._save_experiment_progress(experiment_name, query_results, failures, len(test_samples))
                query_results.pop()

            try:
                eval_dataset = Dataset.from_dict({
                    "question": [q],
                    "answer": [ans],
                    "contexts": [ctx],
                    "ground_truth": [sample["ground_truth"]]
                })
                result = evaluate(
                    dataset=eval_dataset,
                    metrics=metrics,
                    llm=eval_llm,
                    embeddings=eval_embeddings,
                    experiment_name=experiment_name
                )
                if result.scores:
                    query_result.update(self._json_safe(result.scores[0]))
                query_results.append(query_result)
                completed_sample_indexes.add(sample_index)
            except Exception as exc:
                failures.append({
                    "sample_index": sample_index,
                    "stage": "ragas_evaluation",
                    "question": q,
                    "answer": ans,
                    "contexts": ctx,
                    "ground_truth": sample.get("ground_truth"),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)
                })
                completed_sample_indexes.add(sample_index)

            if experiment_name:
                self._save_experiment_progress(experiment_name, query_results, failures, len(test_samples))

        if not query_results:
            if experiment_name:
                self._save_experiment_progress(
                    experiment_name,
                    query_results,
                    failures,
                    len(test_samples),
                    status="failed"
                )
            return {"error": "All test samples failed", "failures": failures}

        final_status = "completed_with_failures" if failures else "completed"
        if experiment_name:
            self._save_experiment_progress(
                experiment_name,
                query_results,
                failures,
                len(test_samples),
                status=final_status
            )

        return {
            "experiment_name": experiment_name,
            "status": final_status,
            "num_requested_samples": len(test_samples),
            "num_successful_samples": len(query_results),
            "num_failed_samples": len(failures),
            "scores": self._calculate_summary_scores(query_results)
        }
