"""Evaluation page for the Streamlit app."""
import json
import os
from pathlib import Path

import streamlit as st

from ..data_utils.testset_utils import generate_testset, load_testset
from ..rag.evaluation.rag_evaluator import RAGEvaluator


def evaluation_page():
    """Render the evaluation page."""
    st.set_page_config(page_title="Evaluation - University Chatbot", page_icon="🧪", layout="wide")

    st.title("🧪 RAG Evaluation")
    st.caption("Run RAGAS evaluation and view results")

    # Read global configuration (from Settings page)
    data_source_path = os.getenv("DATA_SOURCE_PATH", "data/intents-v2.json")
    test_data_path = os.getenv("RAG_TEST_DATA_PATH", "data/rag_test_data.json")
    experiment_results_dir = os.getenv("RAG_EXPERIMENTS_DIR", "data/evaluation_results")

    # Display current global configuration (read-only from Settings)
    with st.expander("📋 Current Global Configuration", expanded=False):
        st.info("These settings are configured in the ⚙️ Settings page")
        config_cols = st.columns(4)
        config_cols[0].metric("Retriever", os.getenv("RETRIEVAL_TYPE", "vector"))
        config_cols[1].metric("Self-Eval", "Enabled" if os.getenv("ENABLE_SELF_EVAL", "false").lower() == "true" else "Disabled")
        config_cols[2].metric("Data Source", Path(data_source_path).name)
        config_cols[3].metric("Test Data", Path(test_data_path).name)

    # Only experiment name is configurable per-run
    st.subheader("Experiment Configuration")
    experiment_name = st.text_input(
        "Experiment Name",
        value=os.getenv("RAG_EXPERIMENT_NAME", "baseline"),
        help="Unique name for this evaluation run (results will be saved with this name)"
    )

    # Run evaluation button
    st.divider()

    if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
        # Set only experiment name for this run
        os.environ["RAG_EXPERIMENT_NAME"] = experiment_name

        # Create progress containers
        progress_container = st.container()
        with progress_container:
            status_text = st.empty()
            progress_bar = st.progress(0)
            log_output = st.empty()

        try:
            status_text.text("Initializing evaluator...")
            progress_bar.progress(10)

            # Initialize evaluator with centralized pipeline
            evaluator = RAGEvaluator(
                data_source_path=data_source_path,
                test_data_path=test_data_path,
                experiment_results_dir=experiment_results_dir
            )

            # Load or generate test samples
            test_path = Path(test_data_path)
            if test_path.exists():
                status_text.text("Loading test samples...")
                test_samples = load_testset(test_path)
                progress_bar.progress(20)
                st.success(f"✅ Loaded {len(test_samples)} test samples from {test_data_path}")
            else:
                status_text.text("Generating test samples...")
                test_samples = generate_testset(
                    data_source_path,
                    test_set_ratio=0.2,
                    test_data_path=test_path,
                    refined_data_path=None
                )
                progress_bar.progress(30)
                st.success(f"✅ Generated {len(test_samples)} test samples")

            # Run evaluation
            status_text.text(f"Running evaluation with {len(test_samples)} samples...")
            progress_bar.progress(40)

            results = evaluator.run_evaluation(test_samples, experiment_name=experiment_name)

            progress_bar.progress(100)
            status_text.text("Evaluation complete!")

            # Display results
            st.divider()
            st.subheader("📊 Evaluation Results")

            if "error" in results:
                st.error(f"Error: {results['error']}")
            else:
                # Summary metrics
                summary = results.get("summary", {})
                scores = summary.get("scores", {})

                st.success(f"✅ Evaluation completed: {summary.get('num_successful_samples', 0)} successful, {summary.get('num_failed_samples', 0)} failed")

                # Main metrics
                st.subheader("Overall Metrics")
                metric_cols = st.columns(4)
                metrics_to_show = [
                    ("faithfulness", "Faithfulness"),
                    ("answer_relevancy", "Answer Relevancy"),
                    ("context_precision", "Context Precision"),
                    ("context_recall", "Context Recall")
                ]

                for i, (key, label) in enumerate(metrics_to_show):
                    if key in scores:
                        value = scores[key]
                        metric_cols[i].metric(label, f"{value:.3f}" if isinstance(value, (int, float)) else "N/A")

                # Breakdown by exists_in_source
                if "by_exists_in_source" in scores:
                    st.subheader("Metrics by Question Type")
                    breakdown = scores["by_exists_in_source"]

                    breakdown_cols = st.columns(2)

                    with breakdown_cols[0]:
                        st.write("**In Source (Knowledge Base)**")
                        in_source = breakdown.get("in_source", {})
                        st.write(f"Count: {in_source.get('count', 0)}")
                        for key, label in metrics_to_show:
                            if key in in_source:
                                st.write(f"{label}: {in_source[key]:.3f}")

                    with breakdown_cols[1]:
                        st.write("**Not In Source (Out-of-Domain)**")
                        not_in_source = breakdown.get("not_in_source", {})
                        st.write(f"Count: {not_in_source.get('count', 0)}")
                        for key, label in metrics_to_show:
                            if key in not_in_source:
                                st.write(f"{label}: {not_in_source[key]:.3f}")

                # Show file paths
                st.subheader("📁 Result Files")
                paths = evaluator._experiment_paths(experiment_name)
                st.write(f"- Summary: `{paths['summary']}`")
                st.write(f"- Samples (JSON): `{paths['samples_json']}`")
                st.write(f"- Samples (CSV): `{paths['samples_csv']}`")
                st.write(f"- Failures: `{paths['failures']}`")

                # Option to view samples
                if st.checkbox("View Sample Results"):
                    samples_path = paths['samples_json']
                    if samples_path.exists():
                        with open(samples_path, 'r') as f:
                            samples = json.load(f)
                        st.json(samples[:3])  # Show first 3 samples

        except Exception as e:
            st.error(f"❌ Error during evaluation: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    # Previous results section
    st.divider()
    st.subheader("📂 Previous Results")

    results_dir = Path(experiment_results_dir)
    if results_dir.exists():
        # Find all experiment result files
        summary_files = list(results_dir.glob("*_summary.json"))
        if summary_files:
            selected_file = st.selectbox(
                "Select experiment to view",
                options=[f.stem.replace("_summary", "") for f in summary_files],
                format_func=lambda x: f"📊 {x}"
            )

            if selected_file:
                summary_path = results_dir / f"{selected_file}_summary.json"
                if summary_path.exists():
                    with open(summary_path, 'r') as f:
                        prev_results = json.load(f)

                    st.json(prev_results)
        else:
            st.info("No previous evaluation results found.")
    else:
        st.info(f"Results directory does not exist: {results_dir}")
