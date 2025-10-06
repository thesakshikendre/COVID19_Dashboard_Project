"""
COVID-19 Policy Impact Analysis Module
=====================================
Analyzes the correlation between government policies and COVID-19 case trends.

Author: [Your Name]
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PolicyImpactAnalyzer:
    """
    Analyzes the impact of government policies on COVID-19 case trends.
    Integrates Oxford Government Response Tracker data with case data.
    """

    def __init__(self, data_path="data/processed/"):
        self.data_path = data_path
        self.policy_data = None
        self.case_data = None

    def create_sample_policy_data(self):
        """
        Create sample Oxford Government Response Tracker data for demonstration.
        In production, this would load from the actual OxCGRT dataset.
        """
        print("🏛️  Creating sample policy data...")

        countries = ['United States', 'Brazil', 'India', 'Russia', 'France', 
                    'Iran', 'Germany', 'Turkey', 'United Kingdom', 'Italy',
                    'Argentina', 'Ukraine', 'Poland', 'Colombia', 'Mexico']

        # Create date range
        start_date = datetime(2020, 1, 22)
        end_date = datetime(2025, 9, 8)
        dates = pd.date_range(start_date, end_date, freq='D')

        policy_records = []

        for country in countries:
            for date in dates[::7]:  # Weekly data points
                # Create realistic policy stringency patterns
                days_since_start = (date - start_date).days

                # Base stringency varies by country and time period
                if days_since_start < 60:  # Early period - low response
                    base_stringency = np.random.uniform(10, 30)
                elif days_since_start < 150:  # First wave - high response
                    base_stringency = np.random.uniform(60, 85)
                elif days_since_start < 300:  # Gradual relaxation
                    base_stringency = np.random.uniform(40, 70)
                elif days_since_start < 500:  # Second wave response
                    base_stringency = np.random.uniform(50, 75)
                else:  # Endemic phase - moderate response
                    base_stringency = np.random.uniform(20, 50)

                # Add country-specific tendencies
                if country in ['China', 'Australia']:
                    base_stringency += 20  # Stricter policies
                elif country in ['Sweden', 'Brazil']:
                    base_stringency -= 15  # More relaxed policies

                # Ensure valid range [0, 100]
                stringency_index = np.clip(base_stringency + np.random.uniform(-10, 10), 0, 100)

                # Individual policy indicators (0-4 scale typically)
                school_closing = min(3, int(stringency_index / 25))
                workplace_closing = min(3, int(stringency_index / 30))
                cancel_events = min(2, int(stringency_index / 40))
                gatherings_restrictions = min(4, int(stringency_index / 20))
                transport_closing = min(2, int(stringency_index / 45))
                stay_home_requirements = min(3, int(stringency_index / 30))
                internal_movement = min(2, int(stringency_index / 45))
                international_travel = min(4, int(stringency_index / 25))

                # Health policies
                testing_policy = min(3, int(stringency_index / 35))
                contact_tracing = min(2, int(stringency_index / 50))
                mask_policies = min(4, int(stringency_index / 25))

                # Economic support (higher during lockdowns)
                income_support = min(2, int(stringency_index / 40))
                debt_relief = min(2, int(stringency_index / 45))

                policy_records.append({
                    'CountryName': country,
                    'Date': date.strftime('%Y-%m-%d'),
                    'StringencyIndex': round(stringency_index, 2),
                    'C1_School_closing': school_closing,
                    'C2_Workplace_closing': workplace_closing,
                    'C3_Cancel_public_events': cancel_events,
                    'C4_Restrictions_on_gatherings': gatherings_restrictions,
                    'C5_Close_public_transport': transport_closing,
                    'C6_Stay_at_home_requirements': stay_home_requirements,
                    'C7_Restrictions_on_internal_movement': internal_movement,
                    'C8_International_travel_controls': international_travel,
                    'H1_Public_info_campaigns': min(2, int(stringency_index / 45)),
                    'H2_Testing_policy': testing_policy,
                    'H3_Contact_tracing': contact_tracing,
                    'H6_Facial_Coverings': mask_policies,
                    'E1_Income_support': income_support,
                    'E2_Debt_contract_relief': debt_relief
                })

        self.policy_data = pd.DataFrame(policy_records)
        print(f"   ✅ Created policy data: {len(self.policy_data)} records for {len(countries)} countries")

        return self.policy_data

    def load_case_data(self):
        """Load processed COVID-19 case data."""
        try:
            self.case_data = {}
            for case_type in ['confirmed', 'deaths']:
                filename = f"{self.data_path}country_daily_{case_type}.csv"
                df = pd.read_csv(filename)
                self.case_data[case_type] = df
                print(f"📖 Loaded {case_type} case data")
        except FileNotFoundError as e:
            print(f"❌ Error loading case data: {e}")

    def align_policy_and_case_data(self):
        """
        Align policy dates with case data for correlation analysis.

        Returns:
            pd.DataFrame: Combined dataset with policies and cases
        """
        if self.policy_data is None or self.case_data is None:
            print("❌ Policy or case data not loaded")
            return None

        print("🔗 Aligning policy and case data...")

        # Get daily confirmed cases
        case_df = self.case_data['confirmed'].copy()
        date_cols = [col for col in case_df.columns if col not in ['Country', 'Lat', 'Long']]

        combined_records = []

        for _, country_row in case_df.iterrows():
            country = country_row['Country']

            # Get policy data for this country
            country_policies = self.policy_data[self.policy_data['CountryName'] == country].copy()
            if country_policies.empty:
                continue

            # Convert policy dates
            country_policies['Date'] = pd.to_datetime(country_policies['Date'])
            country_policies = country_policies.sort_values('Date')

            # Match case data dates with policy dates
            for date_col in date_cols:
                try:
                    case_date = pd.to_datetime(date_col, format='%m/%d/%y')
                    daily_cases = country_row[date_col]

                    # Find closest policy date (forward fill)
                    policy_match = country_policies[country_policies['Date'] <= case_date]
                    if not policy_match.empty:
                        latest_policy = policy_match.iloc[-1]

                        combined_records.append({
                            'Country': country,
                            'Date': case_date,
                            'Daily_Cases': daily_cases,
                            'StringencyIndex': latest_policy['StringencyIndex'],
                            'School_Closing': latest_policy['C1_School_closing'],
                            'Workplace_Closing': latest_policy['C2_Workplace_closing'],
                            'Cancel_Events': latest_policy['C3_Cancel_public_events'],
                            'Gathering_Restrictions': latest_policy['C4_Restrictions_on_gatherings'],
                            'Stay_Home': latest_policy['C6_Stay_at_home_requirements'],
                            'Travel_Controls': latest_policy['C8_International_travel_controls'],
                            'Testing_Policy': latest_policy['H2_Testing_policy'],
                            'Contact_Tracing': latest_policy['H3_Contact_tracing'],
                            'Mask_Policies': latest_policy['H6_Facial_Coverings']
                        })
                except:
                    continue

        combined_df = pd.DataFrame(combined_records)
        print(f"   ✅ Created combined dataset: {len(combined_df)} records")

        return combined_df

    def analyze_policy_effectiveness(self, combined_df, analysis_window=14):
        """
        Analyze the effectiveness of different policies on case reduction.

        Args:
            combined_df (pd.DataFrame): Combined policy and case data
            analysis_window (int): Days to look ahead for measuring impact

        Returns:
            dict: Policy effectiveness analysis results
        """
        print(f"📊 Analyzing policy effectiveness (window: {analysis_window} days)...")

        results = {}

        # Policy indicators to analyze
        policy_indicators = [
            'StringencyIndex', 'School_Closing', 'Workplace_Closing', 
            'Cancel_Events', 'Gathering_Restrictions', 'Stay_Home',
            'Travel_Controls', 'Testing_Policy', 'Contact_Tracing', 'Mask_Policies'
        ]

        for policy in policy_indicators:
            policy_effectiveness = self.calculate_policy_impact(
                combined_df, policy, analysis_window
            )
            results[policy] = policy_effectiveness

        return results

    def calculate_policy_impact(self, df, policy_column, window=14):
        """
        Calculate the impact of a specific policy on case growth.

        Args:
            df (pd.DataFrame): Combined data
            policy_column (str): Policy indicator column name
            window (int): Analysis window in days

        Returns:
            dict: Impact analysis results
        """
        # Sort by country and date
        df_sorted = df.sort_values(['Country', 'Date']).copy()

        impact_records = []

        for country in df_sorted['Country'].unique():
            country_data = df_sorted[df_sorted['Country'] == country].copy()

            if len(country_data) < window * 2:
                continue

            for i in range(len(country_data) - window):
                current_row = country_data.iloc[i]
                future_rows = country_data.iloc[i+1:i+window+1]

                if len(future_rows) < window:
                    continue

                # Policy stringency level
                policy_level = current_row[policy_column]

                # Calculate case growth rate over the window
                current_cases = current_row['Daily_Cases']
                future_cases = future_rows['Daily_Cases'].mean()

                if current_cases > 0:
                    growth_rate = (future_cases - current_cases) / current_cases
                else:
                    growth_rate = 0

                impact_records.append({
                    'Policy_Level': policy_level,
                    'Growth_Rate': growth_rate,
                    'Country': country,
                    'Date': current_row['Date']
                })

        if not impact_records:
            return {'correlation': 0, 'avg_impact': {}, 'sample_size': 0}

        impact_df = pd.DataFrame(impact_records)

        # Calculate correlation between policy stringency and growth rate
        correlation = impact_df['Policy_Level'].corr(impact_df['Growth_Rate'])

        # Calculate average impact by policy level
        avg_impact = {}
        for level in sorted(impact_df['Policy_Level'].unique()):
            level_data = impact_df[impact_df['Policy_Level'] == level]
            avg_impact[level] = level_data['Growth_Rate'].mean()

        return {
            'correlation': correlation,
            'avg_impact': avg_impact,
            'sample_size': len(impact_df),
            'policy_column': policy_column
        }

    def find_policy_timing_effects(self, combined_df):
        """
        Analyze the timing effects of policy implementation.

        Args:
            combined_df (pd.DataFrame): Combined data

        Returns:
            dict: Timing analysis results
        """
        print("⏰ Analyzing policy timing effects...")

        timing_results = {}

        for country in combined_df['Country'].unique()[:10]:  # Limit for performance
            country_data = combined_df[combined_df['Country'] == country].copy()
            country_data = country_data.sort_values('Date')

            # Find policy implementation dates (when stringency increases significantly)
            policy_changes = []
            for i in range(1, len(country_data)):
                current_stringency = country_data.iloc[i]['StringencyIndex']
                prev_stringency = country_data.iloc[i-1]['StringencyIndex']

                # Significant policy tightening (increase of 20+ points)
                if current_stringency - prev_stringency > 20:
                    policy_changes.append({
                        'date': country_data.iloc[i]['Date'],
                        'stringency_change': current_stringency - prev_stringency,
                        'index': i
                    })

            # Analyze impact of each policy change
            change_impacts = []
            for change in policy_changes:
                impact = self.measure_policy_change_impact(country_data, change['index'])
                change_impacts.append(impact)

            if change_impacts:
                timing_results[country] = {
                    'num_policy_changes': len(policy_changes),
                    'avg_impact': np.mean([imp['impact'] for imp in change_impacts if imp['impact'] is not None]),
                    'policy_changes': policy_changes
                }

        return timing_results

    def measure_policy_change_impact(self, country_data, change_index, before_window=7, after_window=14):
        """
        Measure the impact of a specific policy change.

        Args:
            country_data (pd.DataFrame): Country-specific data
            change_index (int): Index of the policy change
            before_window (int): Days before policy change to consider
            after_window (int): Days after policy change to measure impact

        Returns:
            dict: Impact measurement
        """
        # Calculate average cases before policy change
        before_start = max(0, change_index - before_window)
        before_data = country_data.iloc[before_start:change_index]['Daily_Cases']
        before_avg = before_data.mean() if not before_data.empty else 0

        # Calculate average cases after policy change
        after_end = min(len(country_data), change_index + after_window)
        after_data = country_data.iloc[change_index:after_end]['Daily_Cases']
        after_avg = after_data.mean() if not after_data.empty else 0

        # Calculate impact (negative means case reduction)
        if before_avg > 0:
            impact = (after_avg - before_avg) / before_avg
        else:
            impact = None

        return {
            'before_avg': before_avg,
            'after_avg': after_avg,
            'impact': impact
        }

    def generate_policy_insights(self, effectiveness_results, timing_results):
        """
        Generate actionable insights from policy analysis.

        Args:
            effectiveness_results (dict): Policy effectiveness analysis
            timing_results (dict): Policy timing analysis

        Returns:
            dict: Policy insights and recommendations
        """
        print("💡 Generating policy insights...")

        insights = {
            'most_effective_policies': [],
            'least_effective_policies': [],
            'timing_insights': [],
            'recommendations': []
        }

        # Rank policies by effectiveness (more negative correlation = more effective)
        policy_rankings = []
        for policy, results in effectiveness_results.items():
            if results['sample_size'] > 100:  # Only consider policies with sufficient data
                policy_rankings.append({
                    'policy': policy,
                    'correlation': results['correlation'],
                    'sample_size': results['sample_size']
                })

        # Sort by correlation (more negative = better at reducing growth)
        policy_rankings.sort(key=lambda x: x['correlation'])

        # Most effective policies
        insights['most_effective_policies'] = policy_rankings[:3]
        insights['least_effective_policies'] = policy_rankings[-3:]

        # Timing insights
        if timing_results:
            avg_impacts = [r['avg_impact'] for r in timing_results.values() if not np.isnan(r['avg_impact'])]
            if avg_impacts:
                insights['timing_insights'] = [
                    f"Average policy impact: {np.mean(avg_impacts):.2%}",
                    f"Countries with policy changes: {len(timing_results)}",
                    f"Most responsive country: {min(timing_results.keys(), key=lambda k: timing_results[k]['avg_impact'])}"
                ]

        # Generate recommendations
        if policy_rankings:
            most_effective = policy_rankings[0]['policy']
            insights['recommendations'] = [
                f"Prioritize implementation of {most_effective} policies for maximum impact",
                "Implement policies within 14 days of case increase for optimal effectiveness",
                "Combine multiple policy measures for enhanced effect",
                "Monitor policy impact continuously and adjust as needed"
            ]

        return insights

    def run_complete_analysis(self):
        """
        Run the complete policy impact analysis pipeline.

        Returns:
            dict: Complete analysis results
        """
        print("🚀 Starting Policy Impact Analysis Pipeline\n")

        # Step 1: Create/load policy data
        self.create_sample_policy_data()

        # Step 2: Load case data
        self.load_case_data()

        # Step 3: Align datasets
        combined_df = self.align_policy_and_case_data()

        if combined_df is None:
            return None

        # Step 4: Analyze policy effectiveness
        effectiveness_results = self.analyze_policy_effectiveness(combined_df)

        # Step 5: Analyze timing effects
        timing_results = self.find_policy_timing_effects(combined_df)

        # Step 6: Generate insights
        insights = self.generate_policy_insights(effectiveness_results, timing_results)

        print("\n🎉 Policy impact analysis completed!")

        return {
            'effectiveness_results': effectiveness_results,
            'timing_results': timing_results,
            'insights': insights,
            'combined_data': combined_df
        }

if __name__ == "__main__":
    # Run the complete policy analysis
    analyzer = PolicyImpactAnalyzer()
    results = analyzer.run_complete_analysis()

    if results:
        print("\n📋 Policy Analysis Summary:")
        print("\n🏆 Most Effective Policies:")
        for policy in results['insights']['most_effective_policies']:
            print(f"   • {policy['policy']}: {policy['correlation']:.3f} correlation")

        print("\n💡 Key Recommendations:")
        for rec in results['insights']['recommendations']:
            print(f"   • {rec}")

        # Save results
        combined_df = results['combined_data']
        combined_df.to_csv('COVID19_Dashboard_Project/data/processed/policy_impact_analysis.csv', index=False)
        print("\n💾 Saved policy analysis results to CSV")
