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

# Ensure project root is in sys.path
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
from evaluation.judge import LLMSupportJudge
from evaluation.human_agreement import run_comprehensive_agreement_study

def run_benchmark():
    start_time = time.time()
    console.print("\n[bold cyan]===========================================================[/bold cyan]")
    console.print("[bold cyan]   Apple Support AI Agent: Full Benchmark & Evaluation    [/bold cyan]")
    console.print("[bold cyan]===========================================================[/bold cyan]\n")
    console.print("[dim]Mode: Deterministic Calibrated Multi-Rubric Benchmark on CPU[/dim]")
    console.print("[dim]Reproducibility Guarantee: Zero API keys required, < 15 minutes runtime[/dim]\n")

    # 1. Load Datasets
    console.print("[yellow]>> Loading Datasets (Verifying Zero-Leakage Split)...[/yellow]")
    eval_set_path = "data/golden_eval_set.json"
    train_set_path = "data/processed/apple_train_set.json"
    kb_path = "data/processed/apple_support_kb.json"

    with open(eval_set_path, 'r', encoding='utf-8') as f:
        golden_data = json.load(f)

    with open(train_set_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # Verify zero thread leakage
    train_ids = {x.get('conversation_id', '') for x in train_data}
    gold_ids = {x.get('conversation_id', '') for x in golden_data}
    overlap = train_ids.intersection(gold_ids)
    assert len(overlap) == 0, f"Critical Data Contamination Detected! {len(overlap)} threads overlap."
    console.print(f"[green][OK] Verified Zero Thread Overlap between Train ({len(train_data)}) and Gold ({len(golden_data)}).[/green]")

    # 2. Train and Initialize Models
    console.print("\n[yellow]>> Initializing Agents and Training Classifiers on Disjoint Train Split...[/yellow]")
    retriever = AppleSupportRetriever(kb_path=kb_path)
    retriever.build_index()

    # Train Intent Classifier strictly on disjoint training set
    classifier = IntentClassifier(model_path="data/processed/intent_classifier.joblib")
    classifier.train(train_data)

    escalation_engine = EscalationEngine()
    generator = GroundedReplyGenerator()

    # Instantiate Agents
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

    # Ground Truth Vectors
    gold_intents = [item['gold_intent'] for item in golden_data]
    gold_escalations = [item['gold_escalation'] for item in golden_data]
    gold_references = [item['reference_resolution'] for item in golden_data]

    results = {}
    judge = LLMSupportJudge()

    # 3. Benchmark Execution
    for agent_name, agent in agents.items():
        console.print(f"[yellow]>> Evaluating {agent_name}...[/yellow]")
        preds = []
        pred_intents = []
        pred_escalations = []
        pred_replies = []

        for item in golden_data:
            out = agent.process_message(item['customer_query'])
            preds.append(out)
            pred_intents.append(out['intent'])
            pred_escalations.append(out['escalation_decision'])
            pred_replies.append(out['draft_reply'])

        # Automated Metrics
        intent_met = calculate_intent_metrics(gold_intents, pred_intents, labels=INTENTS)
        esc_met = calculate_escalation_metrics(gold_escalations, pred_escalations)
        gen_met = calculate_generation_metrics(pred_replies, gold_references, target_intents=gold_intents)
        
        # LLM-as-a-Judge Evaluation
        judge_scores = judge.evaluate_batch(golden_data, preds)

        results[agent_name] = {
            "intent_metrics": intent_met,
            "escalation_metrics": esc_met,
            "generation_metrics": gen_met,
            "judge_metrics": judge_scores,
            "sample_predictions": preds[:3]
        }
        console.print(f"[green][OK] Completed {agent_name}.[/green]")

    # 4. Human-Judge Agreement Study (on Proposed Agent)
    console.print("\n[yellow]>> Conducting Human-Judge Inter-Rater Reliability Study (N=200)...[/yellow]")
    agreement_stats = run_comprehensive_agreement_study(
        golden_data,
        results["Proposed AI Agent"]["judge_metrics"]
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

    rows = [
        ("Intent Accuracy (Out-of-Sample)", 
         f"{results['Trivial Baseline']['intent_metrics']['accuracy']*100:.1f}%",
         f"{results['Simple Baseline']['intent_metrics']['accuracy']*100:.1f}%",
         f"{results['Proposed AI Agent']['intent_metrics']['accuracy']*100:.1f}%"),
        ("Intent Macro F1",
         f"{results['Trivial Baseline']['intent_metrics']['macro_f1']:.3f}",
         f"{results['Simple Baseline']['intent_metrics']['macro_f1']:.3f}",
         f"{results['Proposed AI Agent']['intent_metrics']['macro_f1']:.3f}"),
        ("Escalation Accuracy",
         f"{results['Trivial Baseline']['escalation_metrics']['accuracy']*100:.1f}%",
         f"{results['Simple Baseline']['escalation_metrics']['accuracy']*100:.1f}%",
         f"{results['Proposed AI Agent']['escalation_metrics']['accuracy']*100:.1f}%"),
        ("Escalation Recall (Safety-Critical)",
         f"{results['Trivial Baseline']['escalation_metrics']['escalation_recall']*100:.1f}%",
         f"{results['Simple Baseline']['escalation_metrics']['escalation_recall']*100:.1f}%",
         f"{results['Proposed AI Agent']['escalation_metrics']['escalation_recall']*100:.1f}%"),
        ("Escalation Precision",
         f"{results['Trivial Baseline']['escalation_metrics']['escalation_precision']*100:.1f}%",
         f"{results['Simple Baseline']['escalation_metrics']['escalation_precision']*100:.1f}%",
         f"{results['Proposed AI Agent']['escalation_metrics']['escalation_precision']*100:.1f}%"),
        ("False Escalation Rate (Lower is Better)",
         f"{results['Trivial Baseline']['escalation_metrics']['false_escalation_rate']*100:.1f}%",
         f"{results['Simple Baseline']['escalation_metrics']['false_escalation_rate']*100:.1f}%",
         f"{results['Proposed AI Agent']['escalation_metrics']['false_escalation_rate']*100:.1f}%"),
        ("Twitter Char Limit Compliance (<280)",
         f"{results['Trivial Baseline']['generation_metrics']['char_limit_compliance_pct']:.1f}%",
         f"{results['Simple Baseline']['generation_metrics']['char_limit_compliance_pct']:.1f}%",
         f"{results['Proposed AI Agent']['generation_metrics']['char_limit_compliance_pct']:.1f}%"),
        ("Official Domain Link Validity",
         f"{results['Trivial Baseline']['generation_metrics']['official_domain_validity_pct']:.1f}%",
         f"{results['Simple Baseline']['generation_metrics']['official_domain_validity_pct']:.1f}%",
         f"{results['Proposed AI Agent']['generation_metrics']['official_domain_validity_pct']:.1f}%"),
        ("Intent-Link Relevance Rate",
         f"{results['Trivial Baseline']['generation_metrics']['link_relevance_pct']:.1f}%",
         f"{results['Simple Baseline']['generation_metrics']['link_relevance_pct']:.1f}%",
         f"{results['Proposed AI Agent']['generation_metrics']['link_relevance_pct']:.1f}%"),
        ("Judge: Groundedness (1-5)",
         f"{results['Trivial Baseline']['judge_metrics']['mean_groundedness']:.2f}",
         f"{results['Simple Baseline']['judge_metrics']['mean_groundedness']:.2f}",
         f"{results['Proposed AI Agent']['judge_metrics']['mean_groundedness']:.2f}"),
        ("Judge: Brand Voice & Empathy (1-5)",
         f"{results['Trivial Baseline']['judge_metrics']['mean_brand_voice']:.2f}",
         f"{results['Simple Baseline']['judge_metrics']['mean_brand_voice']:.2f}",
         f"{results['Proposed AI Agent']['judge_metrics']['mean_brand_voice']:.2f}"),
        ("Judge: Actionability (1-5)",
         f"{results['Trivial Baseline']['judge_metrics']['mean_actionability']:.2f}",
         f"{results['Simple Baseline']['judge_metrics']['mean_actionability']:.2f}",
         f"{results['Proposed AI Agent']['judge_metrics']['mean_actionability']:.2f}"),
        ("Judge: Escalation Appropriateness (1-5)",
         f"{results['Trivial Baseline']['judge_metrics']['mean_escalation_appropriateness']:.2f}",
         f"{results['Simple Baseline']['judge_metrics']['mean_escalation_appropriateness']:.2f}",
         f"{results['Proposed AI Agent']['judge_metrics']['mean_escalation_appropriateness']:.2f}"),
        ("Judge: Overall Quality Score (1-5)",
         f"{results['Trivial Baseline']['judge_metrics']['mean_overall_score']:.2f}",
         f"{results['Simple Baseline']['judge_metrics']['mean_overall_score']:.2f}",
         f"{results['Proposed AI Agent']['judge_metrics']['mean_overall_score']:.2f}")
    ]

    for r in rows:
        summary_table.add_row(*r)

    console.print(summary_table)

    # Display Human-Judge Agreement Table
    console.print("\n[bold magenta]===========================================================[/bold magenta]")
    console.print("[bold magenta]       HUMAN-JUDGE INTER-RATER AGREEMENT STATS (N=200)     [/bold magenta]")
    console.print("[bold magenta]===========================================================[/bold magenta]\n")

    agr_table = Table(title="LLM-as-a-Judge vs. Human Ground Truth Agreement")
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
            "gold_threads": len(golden_data),
            "overlap_threads": len(overlap)
        },
        "results": results,
        "human_agreement": agreement_stats,
        "elapsed_seconds": round(time.time() - start_time, 2)
    }

    out_file = Path("evaluation/benchmark_results.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_payload, f, indent=2)

    console.print(f"\n[bold green][OK] Benchmark completed successfully in {benchmark_payload['elapsed_seconds']}s.[/bold green]")
    console.print(f"[bold green][OK] Results written to {out_file.absolute()}[/bold green]\n")

if __name__ == '__main__':
    run_benchmark()
