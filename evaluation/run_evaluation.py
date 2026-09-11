"""
Master Evaluation Runner for Apple Support Agent Benchmark.
Evaluates:
1. Trivial Baseline (Majority Class + Canned Macro, Always AUTO_HANDLE)
2. Simple Baseline (Naive Bayes + Keyword Rules + 1-NN Retrieval)
3. Proposed AI Support Agent (Calibrated Classifier + Hybrid RAG + Grounded Generator + Policy Escalation)

Zero Data Leakage Guarantee:
- Training Split: data/processed/apple_train_set.json (600 conversation threads)
- Golden Holdout: data/golden_eval_set.json (200 conversation threads)
- Verified Thread Overlap: Exactly 0 threads.
"""

import sys
from pathlib import Path

# Ensure UTF-8 stdout on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
from typing import List, Dict, Any
from tabulate import tabulate
from rich.console import Console
from rich.table import Table

console = Console(force_terminal=False, no_color=False)

from src.data_loader import load_conversation_dataset
from src.retriever import AppleSupportRetriever
from src.intent_classifier import IntentClassifier, INTENTS
from src.escalation_engine import EscalationEngine
from src.generator import GroundedReplyGenerator
from src.agent import AppleSupportAgent
from src.baselines.trivial_baseline import TrivialBaselineAgent
from src.baselines.simple_baseline import SimpleBaselineAgent
from evaluation.metrics import (
    calculate_intent_metrics,
    calculate_escalation_metrics,
    calculate_generation_metrics
)
from evaluation.heuristic_judge import HeuristicSupportJudge
from evaluation.human_agreement import run_human_judge_agreement_study

def run_benchmark():
    start_time = time.time()
    console.print("\n[bold cyan]===========================================================[/bold cyan]")
    console.print("[bold cyan]   Apple Support AI Agent: Full Benchmark & Evaluation    [/bold cyan]")
    console.print("[bold cyan]===========================================================[/bold cyan]\n")
    console.print("[dim]Evaluation Mode: Deterministic Calibrated Automated Metrics on CPU[/dim]")
    console.print("[dim]Zero Data Leakage: Thread-disjoint train vs. holdout gold set[/dim]\n")

    # 1. Load Datasets
    console.print("[yellow]>> Loading Datasets & Verifying Zero-Leakage Split...[/yellow]")
    eval_set_path = "data/golden_eval_set.json"
    train_set_path = "data/processed/apple_train_set.json"
    kb_path = "data/processed/apple_support_kb.json"

    with open(eval_set_path, 'r', encoding='utf-8') as f:
        golden_data = json.load(f)

    with open(train_set_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open(kb_path, 'r', encoding='utf-8') as f:
        kb_data = json.load(f)

    # Assert thread disjointness
    train_ids = {x.get('conversation_id', '') for x in train_data}
    gold_ids = {x.get('conversation_id', '') for x in golden_data}
    kb_ids = {x.get('id', '') for x in kb_data}

    overlap_train = train_ids.intersection(gold_ids)
    overlap_kb = kb_ids.intersection(gold_ids)
    assert len(overlap_train) == 0, f"Leakage detected: {len(overlap_train)} threads in train & gold"
    assert len(overlap_kb) == 0, f"Leakage detected: {len(overlap_kb)} threads in KB & gold"
    console.print(f"[green][OK] Verified 0 Thread Overlap: Train ({len(train_data)}) vs KB ({len(kb_data)}) vs Gold ({len(golden_data)}).[/green]")

    # 2. Train and Initialize Models
    console.print("\n[yellow]>> Initializing Agents and Training Classifiers on Disjoint Train Split...[/yellow]")
    retriever = AppleSupportRetriever(kb_path=kb_path)
    retriever.build_index()

    classifier = IntentClassifier(model_path="data/processed/intent_classifier.joblib")
    classifier.train(train_data)

    escalation_engine = EscalationEngine()
    generator = GroundedReplyGenerator()

    trivial_agent = TrivialBaselineAgent()
    simple_agent = SimpleBaselineAgent(retriever=retriever)
    simple_agent.train_classifier(train_data)

    proposed_agent = AppleSupportAgent(
        classifier=classifier,
        retriever=retriever,
        escalation_engine=escalation_engine,
        generator=generator
    )
    proposed_agent.initialize()

    agents = {
        "Trivial Baseline": trivial_agent,
        "Simple Baseline": simple_agent,
        "Proposed AI Agent": proposed_agent
    }

    gold_intents = [item['gold_intent'] for item in golden_data]
    gold_escalations = [item['gold_escalation'] for item in golden_data]
    gold_references = [item['reference_resolution'] for item in golden_data]

    results = {}
    judge = HeuristicSupportJudge()
    error_analysis = []

    # 3. Benchmark Execution
    for agent_name, agent in agents.items():
        console.print(f"[yellow]>> Evaluating {agent_name}...[/yellow]")
        preds = []
        pred_intents = []
        pred_probs = []
        pred_escalations = []
        pred_replies = []

        for idx, item in enumerate(golden_data):
            msg = item.get("current_customer_message") or item.get("customer_query")
            ctx = item.get("context_history", [])
            out = agent.process_message(msg, context_history=ctx)
            preds.append(out)
            pred_intents.append(out['intent'])
            pred_probs.append(out.get('intent_probabilities', out.get('probabilities', {})))
            pred_escalations.append(out['escalation_decision'])
            pred_replies.append(out['draft_reply'])

            # Log errors for Proposed AI Agent
            if agent_name == "Proposed AI Agent":
                intent_err = (out['intent'] != item['gold_intent'])
                esc_err = (out['escalation_decision'] != item['gold_escalation'])
                if intent_err or esc_err:
                    error_analysis.append({
                        "item_id": item["id"],
                        "conversation_id": item.get("conversation_id"),
                        "customer_query": msg,
                        "gold_intent": item["gold_intent"],
                        "predicted_intent": out["intent"],
                        "intent_confidence": out["intent_confidence"],
                        "intent_error": intent_err,
                        "gold_escalation": item["gold_escalation"],
                        "predicted_escalation": out["escalation_decision"],
                        "escalation_error": esc_err,
                        "gold_escalation_reason": item.get("gold_escalation_reason", ""),
                        "agent_escalation_reason": out["escalation_reason"],
                        "policy_triggered": out.get("policy_triggered"),
                        "draft_reply": out["draft_reply"],
                        "grounded_in": out.get("grounded_in", []),
                        "used_evidence": out.get("used_evidence", [])
                    })

        intent_met = calculate_intent_metrics(gold_intents, pred_intents, labels=INTENTS, y_probs=pred_probs)
        esc_met = calculate_escalation_metrics(gold_escalations, pred_escalations)
        gen_met = calculate_generation_metrics(pred_replies, gold_references, target_intents=gold_intents)
        judge_scores = judge.evaluate_batch(golden_data, preds)

        results[agent_name] = {
            "intent_metrics": intent_met,
            "escalation_metrics": esc_met,
            "generation_metrics": gen_met,
            "judge_metrics": judge_scores,
            "sample_predictions": preds[:3]
        }
        console.print(f"[green][OK] Completed {agent_name}.[/green]")

    # Save detailed error analysis file
    err_path = Path("evaluation/error_analysis.json")
    with open(err_path, 'w', encoding='utf-8') as f:
        json.dump(error_analysis, f, indent=2, ensure_ascii=False)
    console.print(f"[green][OK] Saved {len(error_analysis)} genuine error records to {err_path}[/green]")

    # 4. Human-Judge Agreement Study (N=50 Paired Evaluations)
    console.print("\n[yellow]>> Conducting Human-Judge Inter-Rater Reliability Study (N=50 Paired)...[/yellow]")
    agreement_stats = run_human_judge_agreement_study(
        human_ann_path="evaluation/human_annotations.json",
        llm_scores_path="evaluation/llm_judge_scores.json",
        frozen_subset_path="data/frozen_eval_subset_n50.json"
    )

    # 5. Display Comparative Summary Tables
    console.print("\n[bold green]===========================================================[/bold green]")
    console.print("[bold green]          HEADLINE BENCHMARK COMPARISON RESULTS            [/bold green]")
    console.print("[bold green]===========================================================[/bold green]\n")

    summary_table = Table(title="System Performance Comparison (N=200 Golden Test Set, Zero-Leakage Split)")
    summary_table.add_column("Metric Dimension", justify="left", style="bold cyan")
    summary_table.add_column("Trivial Baseline\n(Always Auto-Handle)", justify="center", style="red")
    summary_table.add_column("Simple Baseline\n(Naive Bayes + 1-NN)", justify="center", style="yellow")
    summary_table.add_column("Proposed AI Agent\n(Calibrated RAG + Policy)", justify="center", style="bold green")

    def fmt_stat(val, pct=False, dec=3):
        if val is None:
            return "N/A"
        if pct:
            return f"{val * 100:.1f}%"
        return f"{val:.{dec}f}"

    rows = [
        ("Intent Accuracy (Out-of-Sample)", 
         fmt_stat(results['Trivial Baseline']['intent_metrics']['accuracy'], pct=True),
         fmt_stat(results['Simple Baseline']['intent_metrics']['accuracy'], pct=True),
         fmt_stat(results['Proposed AI Agent']['intent_metrics']['accuracy'], pct=True)),
        ("Intent Macro F1",
         fmt_stat(results['Trivial Baseline']['intent_metrics']['macro_f1']),
         fmt_stat(results['Simple Baseline']['intent_metrics']['macro_f1']),
         fmt_stat(results['Proposed AI Agent']['intent_metrics']['macro_f1'])),
        ("Brier Score (Calibration, Lower is Better)",
         fmt_stat(results['Trivial Baseline']['intent_metrics'].get('brier_score')),
         fmt_stat(results['Simple Baseline']['intent_metrics'].get('brier_score')),
         fmt_stat(results['Proposed AI Agent']['intent_metrics'].get('brier_score'))),
        ("Expected Calibration Error (ECE)",
         fmt_stat(results['Trivial Baseline']['intent_metrics'].get('expected_calibration_error')),
         fmt_stat(results['Simple Baseline']['intent_metrics'].get('expected_calibration_error')),
         fmt_stat(results['Proposed AI Agent']['intent_metrics'].get('expected_calibration_error'))),
        ("Escalation Accuracy",
         fmt_stat(results['Trivial Baseline']['escalation_metrics']['accuracy'], pct=True),
         fmt_stat(results['Simple Baseline']['escalation_metrics']['accuracy'], pct=True),
         fmt_stat(results['Proposed AI Agent']['escalation_metrics']['accuracy'], pct=True)),
        ("Escalation Recall (Safety-Critical)",
         fmt_stat(results['Trivial Baseline']['escalation_metrics']['escalation_recall'], pct=True),
         fmt_stat(results['Simple Baseline']['escalation_metrics']['escalation_recall'], pct=True),
         fmt_stat(results['Proposed AI Agent']['escalation_metrics']['escalation_recall'], pct=True)),
        ("Escalation Precision",
         fmt_stat(results['Trivial Baseline']['escalation_metrics']['escalation_precision'], pct=True),
         fmt_stat(results['Simple Baseline']['escalation_metrics']['escalation_precision'], pct=True),
         fmt_stat(results['Proposed AI Agent']['escalation_metrics']['escalation_precision'], pct=True)),
        ("Escalation F2 Score (Recall-Weighted)",
         fmt_stat(results['Trivial Baseline']['escalation_metrics']['escalation_f2']),
         fmt_stat(results['Simple Baseline']['escalation_metrics']['escalation_f2']),
         fmt_stat(results['Proposed AI Agent']['escalation_metrics']['escalation_f2'])),
        ("False Escalation Rate (Lower is Better)",
         fmt_stat(results['Trivial Baseline']['escalation_metrics']['false_escalation_rate'], pct=True),
         fmt_stat(results['Simple Baseline']['escalation_metrics']['false_escalation_rate'], pct=True),
         fmt_stat(results['Proposed AI Agent']['escalation_metrics']['false_escalation_rate'], pct=True)),
        ("Weighted Risk-Cost Penalty (5*FN + 1*FP)",
         str(results['Trivial Baseline']['escalation_metrics']['weighted_risk_cost']),
         str(results['Simple Baseline']['escalation_metrics']['weighted_risk_cost']),
         str(results['Proposed AI Agent']['escalation_metrics']['weighted_risk_cost'])),
        ("SacreBLEU Score",
         f"{results['Trivial Baseline']['generation_metrics']['bleu']:.1f}",
         f"{results['Simple Baseline']['generation_metrics']['bleu']:.1f}",
         f"{results['Proposed AI Agent']['generation_metrics']['bleu']:.1f}"),
        ("Twitter Char Limit Compliance (<280)",
         f"{results['Trivial Baseline']['generation_metrics']['char_limit_compliance_pct']:.1f}%",
         f"{results['Simple Baseline']['generation_metrics']['char_limit_compliance_pct']:.1f}%",
         f"{results['Proposed AI Agent']['generation_metrics']['char_limit_compliance_pct']:.1f}%"),
        ("Official Domain Inclusion Rate",
         f"{results['Trivial Baseline']['generation_metrics']['official_domain_inclusion_rate']:.1f}%",
         f"{results['Simple Baseline']['generation_metrics']['official_domain_inclusion_rate']:.1f}%",
         f"{results['Proposed AI Agent']['generation_metrics']['official_domain_inclusion_rate']:.1f}%"),
        ("Intent-Link Relevance Rate",
         f"{results['Trivial Baseline']['generation_metrics']['link_relevance_pct']:.1f}%",
         f"{results['Simple Baseline']['generation_metrics']['link_relevance_pct']:.1f}%",
         f"{results['Proposed AI Agent']['generation_metrics']['link_relevance_pct']:.1f}%"),
        ("Heuristic: Groundedness (1-5)",
         f"{results['Trivial Baseline']['judge_metrics']['mean_groundedness']:.2f}",
         f"{results['Simple Baseline']['judge_metrics']['mean_groundedness']:.2f}",
         f"{results['Proposed AI Agent']['judge_metrics']['mean_groundedness']:.2f}"),
        ("Heuristic: Brand Voice & Empathy (1-5)",
         f"{results['Trivial Baseline']['judge_metrics']['mean_brand_voice']:.2f}",
         f"{results['Simple Baseline']['judge_metrics']['mean_brand_voice']:.2f}",
         f"{results['Proposed AI Agent']['judge_metrics']['mean_brand_voice']:.2f}"),
        ("Heuristic: Actionability (1-5)",
         f"{results['Trivial Baseline']['judge_metrics']['mean_actionability']:.2f}",
         f"{results['Simple Baseline']['judge_metrics']['mean_actionability']:.2f}",
         f"{results['Proposed AI Agent']['judge_metrics']['mean_actionability']:.2f}"),
        ("Heuristic: Escalation Appropriateness (1-5)",
         f"{results['Trivial Baseline']['judge_metrics']['mean_escalation_appropriateness']:.2f}",
         f"{results['Simple Baseline']['judge_metrics']['mean_escalation_appropriateness']:.2f}",
         f"{results['Proposed AI Agent']['judge_metrics']['mean_escalation_appropriateness']:.2f}"),
        ("Heuristic: Overall Quality Score (1-5)",
         f"{results['Trivial Baseline']['judge_metrics']['mean_overall_score']:.2f}",
         f"{results['Simple Baseline']['judge_metrics']['mean_overall_score']:.2f}",
         f"{results['Proposed AI Agent']['judge_metrics']['mean_overall_score']:.2f}")
    ]

    for r in rows:
        summary_table.add_row(*r)

    console.print(summary_table)

    # Display Human-Judge Agreement Table
    console.print("\n[bold magenta]===========================================================[/bold magenta]")
    console.print("[bold magenta]       HUMAN-JUDGE INTER-RATER AGREEMENT STATS (N=50)      [/bold magenta]")
    console.print("[bold magenta]===========================================================[/bold magenta]\n")

    agr_table = Table(title="LLM-as-a-Judge vs. Human Ground Truth Agreement on Frozen Model Replies (N=50)")
    agr_table.add_column("Criterion", style="bold cyan")
    agr_table.add_column("Pearson r", justify="center")
    agr_table.add_column("Spearman Rho", justify="center")
    agr_table.add_column("MAE (1-5 Scale)", justify="center")
    agr_table.add_column("Within 0.5 Pts", justify="center")
    agr_table.add_column("Cohen's Kappa (k)", justify="center", style="bold green")

    for k, stat in agreement_stats.items():
        agr_table.add_row(
            stat["metric_name"],
            f"{stat['pearson_r']:.3f}",
            f"{stat['spearman_rho']:.3f}",
            f"{stat['mae']:.3f}",
            f"{stat['within_0.5_points_pct']:.1f}%",
            f"{stat['cohens_kappa']:.3f}"
        )

    console.print(agr_table)

    # Save benchmark results
    benchmark_payload = {
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_size": len(golden_data),
        "split_verification": {
            "train_threads": len(train_data),
            "kb_threads": len(kb_data),
            "gold_threads": len(golden_data),
            "overlap_train_gold": len(overlap_train),
            "overlap_kb_gold": len(overlap_kb)
        },
        "results": results,
        "human_agreement_n50": agreement_stats,
        "elapsed_seconds": round(time.time() - start_time, 2)
    }

    out_file = Path("evaluation/benchmark_results.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_payload, f, indent=2)

    console.print(f"\n[bold green][OK] Benchmark completed successfully in {benchmark_payload['elapsed_seconds']}s.[/bold green]")
    console.print(f"[bold green][OK] Results written to {out_file.absolute()}[/bold green]\n")

if __name__ == '__main__':
    run_benchmark()
