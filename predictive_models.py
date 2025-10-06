"""
COVID-19 Predictive Modeling Module
==================================
Time series forecasting using ARIMA and Prophet models for COVID-19 prediction.

Author: [Your Name]
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import itertools

# Prophet import (with fallback)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("⚠️  Prophet not available. Install with: pip install prophet")
    PROPHET_AVAILABLE = False

class COVID19Predictor:
    """
    Time series forecasting for COVID-19 cases using ARIMA and Prophet models.
    Includes model comparison, validation, and uncertainty quantification.
    """

    def __init__(self, data_path="data/processed/"):
        self.data_path = data_path
        self.models = {}
        self.predictions = {}
        self.metrics = {}

    def load_data_for_prediction(self, country='United States', case_type='confirmed'):
        """
        Load and prepare time series data for a specific country.

        Args:
            country (str): Country name
            case_type (str): Type of cases ('confirmed', 'deaths', 'recovered')

        Returns:
            pd.DataFrame: Time series data ready for modeling
        """
        try:
            # Load daily data
            filename = f"{self.data_path}country_daily_{case_type}.csv"
            df = pd.read_csv(filename)

            # Filter for specific country
            country_data = df[df['Country'] == country]
            if country_data.empty:
                print(f"❌ No data found for {country}")
                return None

            # Extract time series
            date_cols = [col for col in df.columns if col not in ['Country', 'Lat', 'Long']]
            ts_data = country_data[date_cols].iloc[0]

            # Create DataFrame with proper date index
            dates = pd.to_datetime(date_cols, format='%m/%d/%y')
            ts_df = pd.DataFrame({
                'date': dates,
                'cases': ts_data.values
            })

            # Remove negative values and outliers
            ts_df['cases'] = ts_df['cases'].clip(lower=0)

            # Handle extreme outliers (values > 99th percentile * 2)
            threshold = ts_df['cases'].quantile(0.99) * 2
            ts_df['cases'] = ts_df['cases'].clip(upper=threshold)

            # Add rolling averages to smooth the data
            ts_df['cases_7d'] = ts_df['cases'].rolling(window=7, center=True).mean()
            ts_df['cases_14d'] = ts_df['cases'].rolling(window=14, center=True).mean()

            print(f"✅ Loaded {len(ts_df)} days of {case_type} data for {country}")
            return ts_df

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    def prepare_train_test_split(self, ts_df, test_days=30):
        """
        Split time series into training and testing sets.

        Args:
            ts_df (pd.DataFrame): Time series data
            test_days (int): Number of days for testing

        Returns:
            tuple: (train_df, test_df)
        """
        split_index = len(ts_df) - test_days
        train_df = ts_df.iloc[:split_index].copy()
        test_df = ts_df.iloc[split_index:].copy()

        print(f"📊 Training data: {len(train_df)} days")
        print(f"📊 Testing data: {len(test_df)} days")

        return train_df, test_df

    def find_optimal_arima_params(self, ts_data, max_p=5, max_d=2, max_q=5):
        """
        Find optimal ARIMA parameters using grid search with AIC criterion.

        Args:
            ts_data (pd.Series): Time series data
            max_p, max_d, max_q (int): Maximum values for ARIMA parameters

        Returns:
            tuple: Best (p, d, q) parameters
        """
        print("🔍 Finding optimal ARIMA parameters...")

        # Generate all combinations of parameters
        p_values = range(0, max_p + 1)
        d_values = range(0, max_d + 1)
        q_values = range(0, max_q + 1)

        best_aic = float('inf')
        best_params = None
        results = []

        for p, d, q in itertools.product(p_values, d_values, q_values):
            try:
                model = ARIMA(ts_data, order=(p, d, q))
                fitted_model = model.fit()
                aic = fitted_model.aic
                results.append((p, d, q, aic))

                if aic < best_aic:
                    best_aic = aic
                    best_params = (p, d, q)

            except Exception:
                continue

        print(f"   ✅ Best ARIMA parameters: {best_params} (AIC: {best_aic:.2f})")
        return best_params

    def train_arima_model(self, train_df, use_smoothed=True, auto_params=True):
        """
        Train ARIMA model on training data.

        Args:
            train_df (pd.DataFrame): Training data
            use_smoothed (bool): Whether to use smoothed data
            auto_params (bool): Whether to automatically find optimal parameters

        Returns:
            statsmodels ARIMA model: Fitted ARIMA model
        """
        print("🤖 Training ARIMA model...")

        # Choose data series
        if use_smoothed and 'cases_7d' in train_df.columns:
            ts_data = train_df['cases_7d'].dropna()
            print("   Using 7-day smoothed data")
        else:
            ts_data = train_df['cases']
            print("   Using raw daily data")

        # Find optimal parameters
        if auto_params:
            try:
                best_params = self.find_optimal_arima_params(ts_data)
            except:
                print("   ⚠️  Auto parameter search failed, using default (2,1,2)")
                best_params = (2, 1, 2)
        else:
            best_params = (2, 1, 2)  # Default parameters

        # Train model
        try:
            model = ARIMA(ts_data, order=best_params)
            fitted_model = model.fit()

            self.models['arima'] = {
                'model': fitted_model,
                'params': best_params,
                'training_data': ts_data
            }

            print(f"   ✅ ARIMA{best_params} model trained successfully")
            print(f"   📊 AIC: {fitted_model.aic:.2f}")

            return fitted_model

        except Exception as e:
            print(f"   ❌ Error training ARIMA model: {e}")
            return None

    def train_prophet_model(self, train_df, use_smoothed=True):
        """
        Train Prophet model on training data.

        Args:
            train_df (pd.DataFrame): Training data
            use_smoothed (bool): Whether to use smoothed data

        Returns:
            Prophet model: Fitted Prophet model
        """
        if not PROPHET_AVAILABLE:
            print("❌ Prophet not available")
            return None

        print("🔮 Training Prophet model...")

        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_df = train_df.copy()
        prophet_df['ds'] = prophet_df['date']

        if use_smoothed and 'cases_7d' in prophet_df.columns:
            prophet_df['y'] = prophet_df['cases_7d'].fillna(prophet_df['cases'])
            print("   Using 7-day smoothed data")
        else:
            prophet_df['y'] = prophet_df['cases']
            print("   Using raw daily data")

        # Configure Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.8,  # 80% confidence interval
            changepoint_prior_scale=0.05  # Flexibility of trend changes
        )

        try:
            # Fit model
            model.fit(prophet_df[['ds', 'y']])

            self.models['prophet'] = {
                'model': model,
                'training_data': prophet_df
            }

            print("   ✅ Prophet model trained successfully")
            return model

        except Exception as e:
            print(f"   ❌ Error training Prophet model: {e}")
            return None

    def make_predictions(self, model_type, forecast_days=30, test_df=None):
        """
        Generate predictions using trained models.

        Args:
            model_type (str): 'arima' or 'prophet'
            forecast_days (int): Number of days to forecast
            test_df (pd.DataFrame): Test data for validation

        Returns:
            pd.DataFrame: Predictions with confidence intervals
        """
        if model_type not in self.models:
            print(f"❌ {model_type} model not trained")
            return None

        print(f"🔮 Generating {forecast_days}-day forecast with {model_type.upper()}...")

        if model_type == 'arima':
            return self._make_arima_predictions(forecast_days, test_df)
        elif model_type == 'prophet':
            return self._make_prophet_predictions(forecast_days, test_df)

    def _make_arima_predictions(self, forecast_days, test_df):
        """Generate ARIMA predictions."""
        fitted_model = self.models['arima']['model']

        try:
            # Generate forecast
            forecast = fitted_model.forecast(steps=forecast_days)
            conf_int = fitted_model.get_forecast(steps=forecast_days).conf_int()

            # Create prediction DataFrame
            if test_df is not None:
                start_date = test_df['date'].iloc[0]
                dates = pd.date_range(start_date, periods=forecast_days, freq='D')
            else:
                # Extend from last training date
                train_data = self.models['arima']['training_data']
                last_date = train_data.index[-1] if hasattr(train_data.index, 'date') else pd.Timestamp.now()
                dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')

            pred_df = pd.DataFrame({
                'date': dates,
                'predicted': forecast.values,
                'lower_ci': conf_int.iloc[:, 0].values,
                'upper_ci': conf_int.iloc[:, 1].values
            })

            # Ensure non-negative predictions
            pred_df['predicted'] = pred_df['predicted'].clip(lower=0)
            pred_df['lower_ci'] = pred_df['lower_ci'].clip(lower=0)
            pred_df['upper_ci'] = pred_df['upper_ci'].clip(lower=0)

            self.predictions['arima'] = pred_df
            print(f"   ✅ Generated {len(pred_df)} ARIMA predictions")

            return pred_df

        except Exception as e:
            print(f"   ❌ Error generating ARIMA predictions: {e}")
            return None

    def _make_prophet_predictions(self, forecast_days, test_df):
        """Generate Prophet predictions."""
        if not PROPHET_AVAILABLE:
            return None

        model = self.models['prophet']['model']

        try:
            # Create future dataframe
            if test_df is not None:
                start_date = test_df['date'].iloc[0]
                future_dates = pd.date_range(start_date, periods=forecast_days, freq='D')
            else:
                # Extend from last training date
                train_data = self.models['prophet']['training_data']
                last_date = train_data['ds'].max()
                future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')

            future_df = pd.DataFrame({'ds': future_dates})

            # Generate forecast
            forecast = model.predict(future_df)

            # Create prediction DataFrame
            pred_df = pd.DataFrame({
                'date': future_dates,
                'predicted': forecast['yhat'].values,
                'lower_ci': forecast['yhat_lower'].values,
                'upper_ci': forecast['yhat_upper'].values
            })

            # Ensure non-negative predictions
            pred_df['predicted'] = pred_df['predicted'].clip(lower=0)
            pred_df['lower_ci'] = pred_df['lower_ci'].clip(lower=0)
            pred_df['upper_ci'] = pred_df['upper_ci'].clip(lower=0)

            self.predictions['prophet'] = pred_df
            print(f"   ✅ Generated {len(pred_df)} Prophet predictions")

            return pred_df

        except Exception as e:
            print(f"   ❌ Error generating Prophet predictions: {e}")
            return None

    def evaluate_model_performance(self, model_type, test_df):
        """
        Evaluate model performance against test data.

        Args:
            model_type (str): 'arima' or 'prophet'
            test_df (pd.DataFrame): Test data

        Returns:
            dict: Performance metrics
        """
        if model_type not in self.predictions:
            print(f"❌ No predictions available for {model_type}")
            return None

        pred_df = self.predictions[model_type]

        # Align predictions with test data
        test_dates = test_df['date'].dt.date
        pred_dates = pred_df['date'].dt.date

        # Find common dates
        common_dates = set(test_dates) & set(pred_dates)
        if not common_dates:
            print(f"❌ No common dates between predictions and test data")
            return None

        # Extract actual and predicted values for common dates
        actual_values = []
        predicted_values = []

        for date in sorted(common_dates):
            actual_val = test_df[test_df['date'].dt.date == date]['cases'].iloc[0]
            pred_val = pred_df[pred_df['date'].dt.date == date]['predicted'].iloc[0]

            actual_values.append(actual_val)
            predicted_values.append(pred_val)

        if len(actual_values) == 0:
            return None

        # Calculate metrics
        mae = mean_absolute_error(actual_values, predicted_values)
        mse = mean_squared_error(actual_values, predicted_values)
        rmse = np.sqrt(mse)

        # MAPE (handle division by zero)
        mape = np.mean([abs((a - p) / a) for a, p in zip(actual_values, predicted_values) if a != 0]) * 100

        # R-squared
        ss_res = sum((a - p) ** 2 for a, p in zip(actual_values, predicted_values))
        ss_tot = sum((a - np.mean(actual_values)) ** 2 for a in actual_values)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'sample_size': len(actual_values)
        }

        self.metrics[model_type] = metrics

        print(f"📊 {model_type.upper()} Performance Metrics:")
        print(f"   MAE: {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   R²: {r2:.3f}")

        return metrics

    def compare_models(self):
        """
        Compare performance of all trained models.

        Returns:
            dict: Model comparison results
        """
        if not self.metrics:
            print("❌ No model metrics available for comparison")
            return None

        print("\n🏆 Model Comparison:")

        comparison = {}
        metrics_to_compare = ['MAE', 'RMSE', 'MAPE', 'R2']

        for metric in metrics_to_compare:
            values = {}
            for model_type, model_metrics in self.metrics.items():
                values[model_type] = model_metrics.get(metric, None)

            if metric == 'R2':
                # Higher is better for R²
                best_model = max(values.keys(), key=lambda k: values[k] if values[k] is not None else -1)
            else:
                # Lower is better for error metrics
                best_model = min(values.keys(), key=lambda k: values[k] if values[k] is not None else float('inf'))

            comparison[metric] = {
                'values': values,
                'best_model': best_model
            }

            print(f"\n📊 {metric}:")
            for model, value in values.items():
                marker = "🏆" if model == best_model else "  "
                print(f"   {marker} {model.upper()}: {value:.3f}")

        # Overall best model (lowest average rank)
        model_ranks = {model: 0 for model in self.metrics.keys()}
        for metric_data in comparison.values():
            models_sorted = sorted(metric_data['values'].items(), 
                                 key=lambda x: x[1] if x[1] is not None else float('inf'))
            for rank, (model, _) in enumerate(models_sorted):
                model_ranks[model] += rank

        best_overall = min(model_ranks.keys(), key=lambda k: model_ranks[k])
        comparison['overall_best'] = best_overall

        print(f"\n🏆 Overall Best Model: {best_overall.upper()}")

        return comparison

    def run_complete_modeling_pipeline(self, country='United States', case_type='confirmed', 
                                     forecast_days=30, test_days=30):
        """
        Run the complete predictive modeling pipeline.

        Args:
            country (str): Country to model
            case_type (str): Type of cases
            forecast_days (int): Days to forecast
            test_days (int): Days for testing

        Returns:
            dict: Complete modeling results
        """
        print(f"🚀 Starting Predictive Modeling Pipeline for {country}\n")

        # Step 1: Load and prepare data
        ts_df = self.load_data_for_prediction(country, case_type)
        if ts_df is None:
            return None

        # Step 2: Train-test split
        train_df, test_df = self.prepare_train_test_split(ts_df, test_days)

        # Step 3: Train models
        arima_model = self.train_arima_model(train_df)
        prophet_model = self.train_prophet_model(train_df)

        # Step 4: Generate predictions
        results = {}

        if arima_model:
            arima_pred = self.make_predictions('arima', forecast_days, test_df)
            if arima_pred is not None:
                arima_metrics = self.evaluate_model_performance('arima', test_df)
                results['arima'] = {
                    'predictions': arima_pred,
                    'metrics': arima_metrics
                }

        if prophet_model:
            prophet_pred = self.make_predictions('prophet', forecast_days, test_df)
            if prophet_pred is not None:
                prophet_metrics = self.evaluate_model_performance('prophet', test_df)
                results['prophet'] = {
                    'predictions': prophet_pred,
                    'metrics': prophet_metrics
                }

        # Step 5: Compare models
        comparison = self.compare_models()

        print("\n🎉 Predictive modeling pipeline completed!")

        return {
            'country': country,
            'case_type': case_type,
            'data': ts_df,
            'train_data': train_df,
            'test_data': test_df,
            'models': results,
            'comparison': comparison
        }

def run_multi_country_analysis(countries=['United States', 'India', 'Brazil'], 
                             case_type='confirmed', forecast_days=14):
    """
    Run predictive analysis for multiple countries.

    Args:
        countries (list): List of countries to analyze
        case_type (str): Type of cases
        forecast_days (int): Days to forecast

    Returns:
        dict: Multi-country analysis results
    """
    print(f"🌍 Running multi-country analysis for {len(countries)} countries...")

    results = {}

    for country in countries:
        print(f"\n{'='*50}")
        print(f"Analyzing: {country}")
        print('='*50)

        predictor = COVID19Predictor()
        country_results = predictor.run_complete_modeling_pipeline(
            country=country, 
            case_type=case_type, 
            forecast_days=forecast_days
        )

        if country_results:
            results[country] = country_results

    return results

if __name__ == "__main__":
    # Run single country analysis
    predictor = COVID19Predictor()
    results = predictor.run_complete_modeling_pipeline(
        country='United States',
        case_type='confirmed',
        forecast_days=21,
        test_days=21
    )

    if results and results['comparison']:
        best_model = results['comparison']['overall_best']
        print(f"\n🎯 Recommended model for forecasting: {best_model.upper()}")

        # Save predictions
        if best_model in results['models']:
            predictions = results['models'][best_model]['predictions']
            predictions.to_csv('COVID19_Dashboard_Project/data/processed/predictions.csv', index=False)
            print("💾 Saved predictions to CSV")
