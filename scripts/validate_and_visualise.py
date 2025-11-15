"""
RUN COMPLETE PIPELINE: VALIDATION + VISUALISATIONS (FINAL)

ONE COMMAND to run validation + visualisations:
1. Validate on 6 months of real market data (uses existing models)
2. Generate 4 professional visualisations   
3. Save results with progress bars and logging
"""

import sys
sys.path.append('.')

import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_banner(title):
    """Print formatted banner"""
    print("")
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print("")

def run_step(step_num, script_name, description):
    """Run a pipeline step"""
    print_banner(f"STEP {step_num}: {description}")
    
    try:
        result = subprocess.run(
            [sys.executable, f"scripts/{script_name}"],
            check=True,
            capture_output=False
        )
        logger.info(f"Step {step_num} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Step {step_num} failed with error code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"Step {step_num} error: {e}")
        return False


# MAIN PIPELINE
def main():
    """Run complete pipeline"""
    
    print_banner("RL OPTIMAL EXECUTION - VALIDATION PIPELINE")
    
    logger.info("Pipeline workflow:")
    logger.info("  Step 1: Validate DQN , PPO , baselines on 6 months of real market data")
    logger.info("  Step 2: Generate 4 professional visualisations")
    logger.info("  Step 3: Generate summary report")
    logger.info("")
    logger.info("Expected time: 15-20 minutes")
    logger.info("")
    
    # Create required directories
    Path("logs").mkdir(exist_ok=True)
    Path("results/visualisations").mkdir(parents=True, exist_ok=True)
    
    # STEP 1: Validation
    success = run_step(
        1,
        "validate_models_on_real_data.py",
        "VALIDATION - Test DQN + PPO on 6 months"
    )
    
    if not success:
        logger.error("Pipeline failed at Step 1")
        return False
    
    # STEP 2: Visualisations
    success = run_step(
        2,
        "generate_visualisations.py",
        "VISUALISATION - Generate 4 professional charts"
    )
    
    if not success:
        logger.error("Pipeline failed at Step 2")
        return False
    
    # STEP 3: Summary Report
    print_banner("STEP 3: GENERATE SUMMARY REPORT")
    
    try:
        generate_summary_report()
        logger.info("Step 3 completed successfully")
    except Exception as e:
        logger.error(f"Step 3 error: {e}")
        
    
    # Final summary
    print_banner("COMPLETED PIPELINE: ")
    logger.info("Output files generated:")
    logger.info("")
    logger.info("Results Data:")
    logger.info("   results/validation_results.csv")
    logger.info("")
    logger.info("Visualisations (4 charts):")
    logger.info("   results/visualisations/01_slippage_comparison.png")
    logger.info("   results/visualisations/02_execution_efficiency.png")
    logger.info("   results/visualisations/03_model_performance_heatmap.png")
    logger.info("   results/visualisations/04_statistical_summary.png")
    logger.info("")
    logger.info("Logs:")
    logger.info("   logs/pipeline.log")
    logger.info("   logs/validation.log")
    logger.info("   logs/visualisation.log")
    logger.info("")
    
    return True

# SUMMARY REPORT GENERATION
def generate_summary_report():
    """Generate text summary of results"""
    import pandas as pd
    
    results_path = Path("results/validation_results.csv")
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")
    
    results_df = pd.read_csv(results_path)
    
    # Create summary
    summary_lines = []
    summary_lines.append("\n" + "=" * 80)
    summary_lines.append("EXECUTION RESULTS SUMMARY (6-MONTH VALIDATION)")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # Statistics by model
    summary_lines.append("Performance by Model (Average Slippage in basis points):")
    summary_lines.append("-" * 80)
    
    stats = results_df.groupby('model')['slippage_bps'].agg(['mean', 'std', 'min', 'max', 'count']).round(2)
    stats = stats.sort_values('mean')
    
    for model, row in stats.iterrows():
        summary_lines.append(
            f"  {model:25s} | Mean: {row['mean']:7.2f} | Std: {row['std']:6.2f} | "
            f"Episodes: {int(row['count']):4d}"
        )
    
    summary_lines.append("")
    summary_lines.append("Key Findings:")
    summary_lines.append("-" * 80)
    
    best_model = results_df.groupby('model')['slippage_bps'].mean().idxmin()
    best_slippage = results_df.groupby('model')['slippage_bps'].mean().min()
    vwap_slippage = results_df[results_df['model'] == 'vwap']['slippage_bps'].mean()
    
    if vwap_slippage != 0:
        improvement = abs((vwap_slippage - best_slippage) / vwap_slippage) * 100
        summary_lines.append(f"  [BEST] {best_model}: {best_slippage:.2f} bps")
        summary_lines.append(f"  [BASELINE] VWAP: {vwap_slippage:.2f} bps")
        summary_lines.append(f"  [IMPROVEMENT] {improvement:.1f}% better")
    
    summary_lines.append(f"  [TOTAL EPISODES] {len(results_df)}")
    summary_lines.append(f"  [SYMBOLS] {results_df['symbol'].nunique()}")
    summary_lines.append(f"  [TRADING DAYS] {results_df['date'].nunique()}")
    summary_lines.append("")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # Print to console and log
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    logger.info(summary_text)
    
    # Save to file
    report_path = Path("results/summary_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    logger.info(f"Summary saved to: {report_path}")


# ENTRY POINT
if __name__ == "__main__":
    print_banner("RL OPTIMAL EXECUTION - PIPELINE STARTER")
    
    success = main()
    
    if not success:
        logger.error("PIPELINE FAILED")
        sys.exit(1)
    else:
        logger.info("PIPELINE SUCCESS!")
        sys.exit(0)