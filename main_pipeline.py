"""
COVID-19 Data Science Pipeline - Main Runner
===========================================
Orchestrates the complete end-to-end data science pipeline.

Author: [Your Name]
Date: September 2025
"""

import os
import sys
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_cleaning import COVID19DataCleaner
from trend_analysis import COVID19TrendAnalyzer
from policy_impact import PolicyImpactAnalyzer  
from predictive_models import COVID19Predictor

class COVID19Pipeline:
    """
    Main pipeline orchestrator for COVID-19 data science project.
    Coordinates data cleaning, analysis, and modeling components.
    """

    def __init__(self, output_dir="reports/"):
        self.output_dir = output_dir
        self.results = {}
        self.start_time = datetime.now()

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def log_step(self, step_name, status="START"):
        """Log pipeline step with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {status}: {step_name}")

    def run_data_pipeline(self):
        """Execute data cleaning and preprocessing pipeline."""
        self.log_step("Data Cleaning & Preprocessing Pipeline")

        try:
            cleaner = COVID19DataCleaner()
            processed_data = cleaner.run_complete_pipeline()
            self.results['data_pipeline'] = processed_data
            self.log_step("Data Cleaning & Preprocessing Pipeline", "COMPLETE")
            return True
        except Exception as e:
            self.log_step(f"Data Pipeline ERROR: {e}", "FAILED")
            return False

    def run_trend_analysis(self):
        """Execute comprehensive trend analysis."""
        self.log_step("Trend Analysis Pipeline")

        try:
            analyzer = COVID19TrendAnalyzer()
            analyzer.load_data()

            # Generate comprehensive trend report
            trend_report = analyzer.generate_trend_report()
            self.results['trend_analysis'] = trend_report

            # Create dashboard figures
            from trend_analysis import create_summary_dashboard
            figures = create_summary_dashboard(analyzer)
            self.results['trend_figures'] = figures

            self.log_step("Trend Analysis Pipeline", "COMPLETE")
            return True
        except Exception as e:
            self.log_step(f"Trend Analysis ERROR: {e}", "FAILED")
            return False

    def run_policy_analysis(self):
        """Execute policy impact analysis."""
        self.log_step("Policy Impact Analysis Pipeline")

        try:
            analyzer = PolicyImpactAnalyzer()
            policy_results = analyzer.run_complete_analysis()
            self.results['policy_analysis'] = policy_results

            self.log_step("Policy Impact Analysis Pipeline", "COMPLETE")
            return True
        except Exception as e:
            self.log_step(f"Policy Analysis ERROR: {e}", "FAILED")
            return False

    def run_predictive_modeling(self, countries=['United States', 'India', 'Brazil']):
        """Execute predictive modeling pipeline."""
        self.log_step("Predictive Modeling Pipeline")

        try:
            # Run modeling for multiple countries
            from predictive_models import run_multi_country_analysis

            modeling_results = run_multi_country_analysis(
                countries=countries,
                case_type='confirmed',
                forecast_days=21
            )

            self.results['predictive_modeling'] = modeling_results

            self.log_step("Predictive Modeling Pipeline", "COMPLETE")
            return True
        except Exception as e:
            self.log_step(f"Predictive Modeling ERROR: {e}", "FAILED")
            return False

    def generate_executive_summary(self):
        """Generate executive summary report."""
        self.log_step("Generating Executive Summary")

        try:
            summary = {
                'pipeline_execution_time': datetime.now() - self.start_time,
                'data_quality_metrics': self._extract_data_quality_metrics(),
                'key_insights': self._extract_key_insights(),
                'model_performance': self._extract_model_performance(),
                'recommendations': self._generate_recommendations()
            }

            # Save summary to file
            self._save_executive_summary(summary)
            self.results['executive_summary'] = summary

            self.log_step("Executive Summary Generation", "COMPLETE")
            return True
        except Exception as e:
            self.log_step(f"Executive Summary ERROR: {e}", "FAILED")
            return False

    def _extract_data_quality_metrics(self):
        """Extract data quality metrics from pipeline results."""
        metrics = {}

        if 'data_pipeline' in self.results:
            cumulative_data = self.results['data_pipeline'].get('cumulative', {})
            if 'confirmed' in cumulative_data:
                df = cumulative_data['confirmed']
                metrics['countries_processed'] = len(df)
                metrics['date_range_days'] = len([col for col in df.columns if col not in ['Country', 'Lat', 'Long']])
                metrics['total_global_cases'] = df.iloc[:, -1].sum()

        return metrics

    def _extract_key_insights(self):
        """Extract key insights from analysis results."""
        insights = []

        # Trend analysis insights
        if 'trend_analysis' in self.results:
            trend_data = self.results['trend_analysis']
            if 'global_summary' in trend_data:
                global_summary = trend_data['global_summary']
                insights.append(f"Global confirmed cases: {global_summary.get('global_confirmed', 0):,.0f}")
                insights.append(f"Global death rate: {global_summary.get('global_death_rate', 0):.2f}%")

        # Policy analysis insights
        if 'policy_analysis' in self.results:
            policy_data = self.results['policy_analysis']
            if 'insights' in policy_data and 'recommendations' in policy_data['insights']:
                insights.extend(policy_data['insights']['recommendations'][:2])

        return insights

    def _extract_model_performance(self):
        """Extract model performance metrics."""
        performance = {}

        if 'predictive_modeling' in self.results:
            modeling_data = self.results['predictive_modeling']

            # Aggregate performance across countries
            model_metrics = {'arima': [], 'prophet': []}

            for country, country_results in modeling_data.items():
                if 'comparison' in country_results and country_results['comparison']:
                    best_model = country_results['comparison'].get('overall_best')
                    performance[country] = best_model

                    # Collect metrics
                    if 'models' in country_results:
                        for model_name, model_data in country_results['models'].items():
                            if 'metrics' in model_data and model_data['metrics']:
                                metrics = model_data['metrics']
                                if model_name in model_metrics:
                                    model_metrics[model_name].append(metrics.get('MAPE', 0))

            # Calculate average performance
            for model_name, mape_scores in model_metrics.items():
                if mape_scores:
                    performance[f'{model_name}_avg_mape'] = sum(mape_scores) / len(mape_scores)

        return performance

    def _generate_recommendations(self):
        """Generate actionable recommendations."""
        recommendations = []

        # Data quality recommendations
        recommendations.append("Implement automated data validation checks for real-time monitoring")
        recommendations.append("Establish data refresh schedules to ensure dashboard currency")

        # Analysis recommendations
        if 'policy_analysis' in self.results:
            recommendations.append("Focus on early policy intervention within 14 days of case increases")
            recommendations.append("Combine multiple policy measures for enhanced effectiveness")

        # Modeling recommendations
        if 'predictive_modeling' in self.results:
            recommendations.append("Use ensemble models combining ARIMA and Prophet for robust forecasting")
            recommendations.append("Update models weekly with latest data for optimal accuracy")

        return recommendations

    def _save_executive_summary(self, summary):
        """Save executive summary to file."""
        filename = f"{self.output_dir}executive_summary.txt"

        with open(filename, 'w') as f:
            f.write("COVID-19 DATA SCIENCE PIPELINE\n")
            f.write("Executive Summary Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Pipeline Execution Time: {summary['pipeline_execution_time']}\n\n")

            f.write("DATA QUALITY METRICS:\n")
            for key, value in summary['data_quality_metrics'].items():
                f.write(f"  • {key}: {value}\n")

            f.write("\nKEY INSIGHTS:\n")
            for insight in summary['key_insights']:
                f.write(f"  • {insight}\n")

            f.write("\nMODEL PERFORMANCE:\n")
            for key, value in summary['model_performance'].items():
                f.write(f"  • {key}: {value}\n")

            f.write("\nRECOMMENDATIONS:\n")
            for rec in summary['recommendations']:
                f.write(f"  • {rec}\n")

            f.write(f"\nReport Generated: {datetime.now()}\n")

        print(f"💾 Executive summary saved to: {filename}")

    def run_complete_pipeline(self, include_predictions=True):
        """
        Execute the complete end-to-end pipeline.

        Args:
            include_predictions (bool): Whether to run predictive modeling (slower)

        Returns:
            dict: Complete pipeline results
        """
        print("🚀 Starting COVID-19 Data Science Pipeline")
        print("=" * 60)

        pipeline_success = True

        # Step 1: Data Pipeline
        if not self.run_data_pipeline():
            pipeline_success = False

        # Step 2: Trend Analysis
        if not self.run_trend_analysis():
            pipeline_success = False

        # Step 3: Policy Analysis
        if not self.run_policy_analysis():
            pipeline_success = False

        # Step 4: Predictive Modeling (optional)
        if include_predictions:
            if not self.run_predictive_modeling():
                pipeline_success = False

        # Step 5: Executive Summary
        if not self.generate_executive_summary():
            pipeline_success = False

        # Pipeline completion
        total_time = datetime.now() - self.start_time

        if pipeline_success:
            print("\n🎉 Pipeline completed successfully!")
            print(f"⏱️  Total execution time: {total_time}")
            print("\n📋 Pipeline Results Summary:")
            print(f"  • Data processing: {'✅' if 'data_pipeline' in self.results else '❌'}")
            print(f"  • Trend analysis: {'✅' if 'trend_analysis' in self.results else '❌'}")
            print(f"  • Policy analysis: {'✅' if 'policy_analysis' in self.results else '❌'}")
            print(f"  • Predictive modeling: {'✅' if 'predictive_modeling' in self.results else '❌'}")
            print(f"  • Executive summary: {'✅' if 'executive_summary' in self.results else '❌'}")
        else:
            print("\n⚠️  Pipeline completed with some errors")
            print(f"⏱️  Total execution time: {total_time}")

        return self.results

def main():
    """Main entry point for pipeline execution."""
    print("COVID-19 Data Science Project Pipeline")
    print("Built by [Your Name] - September 2025")
    print()

    # Initialize and run pipeline
    pipeline = COVID19Pipeline()
    results = pipeline.run_complete_pipeline(include_predictions=True)

    # Display summary
    if 'executive_summary' in results:
        summary = results['executive_summary']
        print("\n" + "=" * 60)
        print("EXECUTIVE SUMMARY")
        print("=" * 60)

        for insight in summary['key_insights']:
            print(f"📊 {insight}")

        print("\n💡 Key Recommendations:")
        for rec in summary['recommendations'][:3]:
            print(f"  • {rec}")

        print("\n🎯 Next Steps:")
        print("  • Launch interactive dashboard: streamlit run dashboards/streamlit_app.py")
        print("  • Review detailed reports in reports/ directory")
        print("  • Customize analysis for specific use cases")

if __name__ == "__main__":
    main()
