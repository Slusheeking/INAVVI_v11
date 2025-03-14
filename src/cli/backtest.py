#!/usr/bin/env python3
"""
CLI command to run backtests for the autonomous trading system.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("inavvi-backtest")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run backtests for the autonomous trading system")
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file",
        default=os.environ.get("INAVVI_CONFIG", ".env")
    )
    parser.add_argument(
        "--log-level", "-l",
        help="Log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=os.environ.get("LOG_LEVEL", "INFO")
    )
    parser.add_argument(
        "--start-date", "-s",
        help="Start date for backtest (YYYY-MM-DD)",
        default=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    )
    parser.add_argument(
        "--end-date", "-e",
        help="End date for backtest (YYYY-MM-DD)",
        default=datetime.now().strftime("%Y-%m-%d")
    )
    parser.add_argument(
        "--symbols", "-sym",
        help="Comma-separated list of symbols to backtest",
        default="SPY,QQQ,AAPL,MSFT,GOOGL"
    )
    parser.add_argument(
        "--timeframes", "-tf",
        help="Comma-separated list of timeframes to backtest",
        default="1m,5m,15m,1h,1d"
    )
    parser.add_argument(
        "--strategy", "-st",
        help="Strategy to backtest",
        choices=["ml_ensemble", "peak_detection", "momentum", "mean_reversion", "all"],
        default="ml_ensemble"
    )
    parser.add_argument(
        "--initial-capital", "-ic",
        help="Initial capital for backtest",
        type=float,
        default=100000.0
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for backtest results (JSON format)",
        default=""
    )
    parser.add_argument(
        "--plot", "-p",
        help="Generate performance plots",
        action="store_true"
    )
    return parser.parse_args()

def run_backtest(args):
    """Run the backtest."""
    logger.info(f"Running backtest from {args.start_date} to {args.end_date}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Timeframes: {args.timeframes}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Initial capital: ${args.initial_capital:,.2f}")
    
    # Import here to avoid circular imports
    try:
        from src.backtesting.engine.backtest_engine import BacktestEngine
        from src.backtesting.analysis.strategy_analyzer import StrategyAnalyzer
        from src.backtesting.reporting.performance_reporter import PerformanceReporter
    except ImportError:
        logger.error("Failed to import backtest modules. Make sure the package is installed correctly.")
        return False
    
    try:
        # Create backtest engine
        engine = BacktestEngine(
            start_date=args.start_date,
            end_date=args.end_date,
            symbols=args.symbols.split(","),
            timeframes=args.timeframes.split(","),
            strategy=args.strategy,
            initial_capital=args.initial_capital
        )
        
        # Run backtest
        results = engine.run()
        
        # Analyze results
        analyzer = StrategyAnalyzer(initial_capital=args.initial_capital)
        metrics = analyzer.calculate_performance_metrics(results)
        
        # Generate report
        reporter = PerformanceReporter()
        report = reporter.generate_performance_report(results, metrics)
        
        # Print summary
        print("\n" + "=" * 80)
        print("BACKTEST SUMMARY")
        print("=" * 80)
        print(f"Strategy: {args.strategy}")
        print(f"Period: {args.start_date} to {args.end_date}")
        print(f"Initial Capital: ${args.initial_capital:,.2f}")
        print(f"Final Capital: ${metrics['final_capital']:,.2f}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print("=" * 80)
        
        # Save results to file if specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump({
                    "config": vars(args),
                    "metrics": metrics,
                    "trades": [trade.to_dict() for trade in results["trades"]]
                }, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        
        # Generate plots if specified
        if args.plot:
            try:
                import matplotlib.pyplot as plt
                
                # Create plots directory if it doesn't exist
                plots_dir = os.path.join("results", "plots")
                os.makedirs(plots_dir, exist_ok=True)
                
                # Generate equity curve plot
                fig = analyzer.plot_equity_curve(results["equity_curve"])
                plot_path = os.path.join(plots_dir, f"equity_curve_{args.strategy}_{args.start_date}_{args.end_date}.png")
                fig.savefig(plot_path)
                plt.close(fig)
                logger.info(f"Equity curve plot saved to {plot_path}")
                
                # Generate return distribution plot
                fig = analyzer.plot_return_distribution(results["returns"])
                plot_path = os.path.join(plots_dir, f"return_distribution_{args.strategy}_{args.start_date}_{args.end_date}.png")
                fig.savefig(plot_path)
                plt.close(fig)
                logger.info(f"Return distribution plot saved to {plot_path}")
            except ImportError:
                logger.warning("Matplotlib not installed. Skipping plot generation.")
        
        return True
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return False

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Run backtest
    if not run_backtest(args):
        logger.error("Backtest failed")
        sys.exit(1)
    
    logger.info("Backtest completed successfully")

if __name__ == "__main__":
    main()