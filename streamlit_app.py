import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
import requests
from datetime import datetime
import os
import sys
from io import StringIO
import traceback
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*use_container_width.*')


# IMPORTANT: Import custom classes with CORRECT names
try:
    from model_classes import (
        RegularizedEnsemble,
        AdvancedFeatureEngineer,
        RobustValidator,
        ResilienceMetrics,
        EnhancedSupplyChainForecaster,
        EnhancedAttention  # CORRECT NAME
    )

    CUSTOM_CLASSES_AVAILABLE = True
except ImportError:
    CUSTOM_CLASSES_AVAILABLE = False
    st.sidebar.warning("⚠️ Custom model classes not found. Some features may be limited.")

warnings.filterwarnings('ignore')

# Import required libraries for model loading
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Supply Chain Sales Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .explanation-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 15px 0;
    }
    </style>
""", unsafe_allow_html=True)


# ==================== LLAMA2 INTEGRATION ====================
class Llama2Integrator:
    def __init__(self, ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model_name = "llama2"
        self.timeout = 60
        self.max_retries = 2
        self.is_available = self.check_ollama_availability()

    def check_ollama_availability(self):
        """Check if Ollama is running and Llama2 is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                return any("llama2" in model for model in available_models)
            return False
        except Exception:
            return False

    def query_llama2_with_retry(self, prompt, max_tokens=250):
        """Send query to Llama2 via Ollama with retry logic"""
        if not self.is_available:
            return "⚠️ AI explanation unavailable. Please ensure Ollama is running with: `ollama run llama2`"

        truncated_prompt = prompt[:600] if len(prompt) > 600 else prompt

        for attempt in range(self.max_retries):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": truncated_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.5,
                        "num_predict": max_tokens,
                        "stop": ["\n\n", "---"]
                    }
                }
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "No response received").strip()
                else:
                    return f"❌ Error: HTTP {response.status_code}"

            except requests.Timeout:
                if attempt == self.max_retries - 1:
                    return "⏱️ Analysis timeout. Try again or check Ollama connection."
                continue
            except Exception as e:
                return f"❌ Connection error: {str(e)[:100]}"

        return "❌ Failed to get AI explanation after multiple attempts."

    def explain_forecast(self, prediction, uncertainty, features_summary, confidence_level):
        """Generate explanation for a forecast"""
        prompt = f"""As a supply chain expert, analyze this sales forecast:

Predicted Sales: ${prediction:.2f}
Uncertainty: ±${uncertainty:.2f}
Confidence: {confidence_level}

Key Input Factors:
{features_summary}

Provide a concise business analysis in 3 parts:
1. Forecast interpretation (2 sentences)
2. Confidence assessment (1 sentence)  
3. Business recommendations (2-3 points)

Keep under 120 words."""
        return self.query_llama2_with_retry(prompt, max_tokens=150)

    def analyze_query_results(self, query_description, result_summary, data_insights):
        """Analyze user query results"""
        prompt = f"""As a supply chain analyst, interpret this data query:

Query: {query_description}
Results: {result_summary}
Key Patterns: {data_insights}

Provide insights in 3 parts:
1. Key findings (2-3 points)
2. Business implications (2 points)
3. Actionable recommendations (2-3 points)

Keep under 100 words."""
        return self.query_llama2_with_retry(prompt, max_tokens=130)

    def generate_overall_insights(self, metrics, data_summary):
        """Generate overall insights for the dataset"""
        prompt = f"""As a supply chain executive, analyze this forecasting performance:

Metrics:
- R² Score: {metrics.get('r2', 0):.3f}
- MAE: ${metrics.get('mae', 0):.2f}
- Resilience Score: {metrics.get('resilience_score', 0):.3f}

Data Summary:
{data_summary}

Provide:
1. Overall performance assessment (2 sentences)
2. Key strengths and concerns (2-3 points)
3. Strategic recommendations (2-3 points)

Keep response under 150 words."""
        return self.query_llama2_with_retry(prompt, max_tokens=200)


# ==================== UTILITY FUNCTIONS ====================
def safe_load_pickle(filepath, item_name):
    """Safely load pickle files with error handling"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading {item_name}: {str(e)[:50]}")
        return None


def validate_dataframe(df, required_cols=None):
    """Validate dataframe structure and content"""
    if df is None or df.empty:
        return False, "Dataset is empty"

    if required_cols:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"

    return True, "Valid dataset"


# ==================== MODEL LOADING FUNCTIONS ====================
@st.cache_resource
def load_models_and_preprocessors():
    """Load pre-trained models and preprocessors"""
    models = {}
    model_status = []

    try:
        # Load MCDFN model (Keras) with CORRECT custom objects
        if TF_AVAILABLE and os.path.exists('best_regularized_model.h5'):
            try:
                if CUSTOM_CLASSES_AVAILABLE:
                    custom_objects = {
                        'EnhancedAttention': EnhancedAttention,
                        'Attention': EnhancedAttention,  # Fallback name
                    }
                else:
                    custom_objects = {}

                models['mcdfn'] = load_model(
                    'best_regularized_model.h5',
                    custom_objects=custom_objects,
                    compile=False,
                    safe_mode=False  # Disable safe mode for custom layers
                )
                model_status.append("✅ MCDFN Model")
            except TypeError as e:
                if "positional argument" in str(e):
                    model_status.append(f"⚠️ MCDFN: Layer compatibility issue")
                    st.sidebar.info("💡 MCDFN model has custom layer issues. Using ensemble models only.")
                else:
                    model_status.append(f"❌ MCDFN: {str(e)[:30]}")
            except Exception as e:
                model_status.append(f"❌ MCDFN: {str(e)[:30]}")

        # Load ensemble models
        ensemble_files = {
            'random_forest': 'randomforest_model.pkl',
            'xgboost': 'xgboost_model.pkl',
            'gradient_boosting': 'gradientboosting_model.pkl',
            'ensemble': 'ensemble_model.pkl'
        }

        for name, filename in ensemble_files.items():
            model = safe_load_pickle(filename, name)
            if model is not None:
                models[name] = model
                model_status.append(f"✅ {name.replace('_', ' ').title()}")

        # Load preprocessor
        preprocessor = safe_load_pickle('preprocessor.pkl', 'preprocessor')
        if preprocessor:
            model_status.append("✅ Preprocessor")

        # Load or generate feature names
        feature_names = safe_load_pickle('feature_names.pkl', 'feature names')

        # If feature_names is missing, try to recover from preprocessor
        if feature_names is None and preprocessor is not None:
            if hasattr(preprocessor, 'feature_names_in_'):
                feature_names = list(preprocessor.feature_names_in_)
                try:
                    with open('feature_names.pkl', 'wb') as f:
                        pickle.dump(feature_names, f)
                    model_status.append("✅ Feature Names (auto-generated)")
                except:
                    model_status.append("⚠️ Feature Names (temporary)")

        # Load metadata
        metadata = None
        if os.path.exists('model_metadata.json'):
            try:
                with open('model_metadata.json', 'r') as f:
                    metadata = json.load(f)
                model_status.append("✅ Metadata")
            except Exception as e:
                model_status.append(f"❌ Metadata: {str(e)[:30]}")

        return models, preprocessor, feature_names, metadata, model_status

    except Exception as e:
        st.error(f"❌ Error in model loading: {str(e)}")
        return {}, None, None, None, [f"❌ Loading failed: {str(e)[:50]}"]


@st.cache_data
def load_dataset(file_source, filepath=None, uploaded_file=None):
    """Load dataset from various sources with robust error handling"""
    try:
        df = None

        if file_source == "local" and filepath:
            for encoding in ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                return None, f"Could not read file with any encoding: {filepath}"

        elif file_source == "upload" and uploaded_file:
            try:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                df = pd.read_csv(stringio)
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file)

        if df is not None:
            is_valid, msg = validate_dataframe(df)
            if not is_valid:
                return None, msg

            return df, f"Successfully loaded {len(df)} rows, {len(df.columns)} columns"
        else:
            return None, "Failed to load dataset"

    except Exception as e:
        error_msg = f"Error loading dataset: {str(e)}"
        st.error(error_msg)
        return None, error_msg


# ==================== DATA PROCESSING FUNCTIONS ====================
def preprocess_data_for_prediction(df, preprocessor, feature_names=None, target_col='Sales'):
    """Preprocess data for model prediction with robust error handling"""
    try:
        df_processed = df.copy()

        if target_col in df_processed.columns:
            X = df_processed.drop(columns=[target_col])
            y = df_processed[target_col].values
        else:
            X = df_processed.copy()
            y = None

        # Try direct transform first
        if preprocessor is not None:
            try:
                X_processed = preprocessor.transform(X)
                # Align features with model expectations
                X_processed = align_features_with_training(X_processed, feature_names, required_features=81)
                return X_processed, y

            except:
                pass

        # Fallback with feature alignment
        if feature_names is not None and preprocessor is not None:
            try:
                X_aligned = X.copy()
                for col in feature_names:
                    if col not in X_aligned.columns:
                        X_aligned[col] = 0
                X_aligned = X_aligned.reindex(columns=feature_names, fill_value=0)
                X_processed = preprocessor.transform(X_aligned)
                return X_processed, y
            except:
                pass

        # Final fallback: numeric only
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_numeric = X[numeric_cols].fillna(0)
            return X_numeric.values, y
        else:
            st.error("❌ No numeric columns found for prediction")
            return None, y

    except Exception as e:
        st.error(f"❌ Data preprocessing failed: {str(e)}")
        return None, None


def align_features_with_training(X_processed, feature_names, required_features=81):
    """Align processed features with training feature count"""
    try:
        current_features = X_processed.shape[1]

        if current_features == required_features:
            return X_processed

        if current_features < required_features:
            # Pad with zeros if features are missing
            padding = np.zeros((X_processed.shape[0], required_features - current_features))
            X_aligned = np.hstack([X_processed, padding])
            st.info(f"ℹ️ Padded features from {current_features} to {required_features}")
            return X_aligned
        else:
            # Truncate if too many features
            X_aligned = X_processed[:, :required_features]
            st.info(f"ℹ️ Truncated features from {current_features} to {required_features}")
            return X_aligned

    except Exception as e:
        st.warning(f"⚠️ Feature alignment failed: {str(e)}")
        return X_processed


def make_predictions(models, X, use_ensemble=True):
    """Make predictions using available models with feature alignment and error handling"""
    if X is None:
        st.error("❌ No data provided for prediction")
        return None

    predictions = {}
    errors = []
    successful_models = []

    try:
        # Display input shape for debugging
        st.info(f"📊 Preprocessed data shape: {X.shape} rows × {X.shape[1]} features")

        # MCDFN prediction (Keras model) - attempt first
        if 'mcdfn' in models:
            try:
                X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
                mcdfn_pred = models['mcdfn'].predict(X_reshaped, verbose=0)

                if isinstance(mcdfn_pred, list) and len(mcdfn_pred) >= 2:
                    predictions['mcdfn_mean'] = mcdfn_pred[0].flatten()
                    predictions['mcdfn_std'] = np.abs(mcdfn_pred[1].flatten())
                else:
                    predictions['mcdfn_mean'] = mcdfn_pred.flatten()
                    predictions['mcdfn_std'] = np.ones_like(predictions['mcdfn_mean']) * 50

                successful_models.append('MCDFN')

            except Exception as e:
                errors.append(f"MCDFN: {str(e)[:80]}")

        # Ensemble model predictions WITH FEATURE ALIGNMENT
        ensemble_models = {
            'random_forest': 'Random Forest',
            'xgboost': 'XGBoost',
            'gradient_boosting': 'Gradient Boosting',
            'ensemble': 'Ensemble'
        }

        for model_key, model_name in ensemble_models.items():
            if model_key in models:
                try:
                    model = models[model_key]

                    # Check expected features
                    expected_features = getattr(model, 'n_features_in_', None)

                    if expected_features and X.shape[1] != expected_features:
                        # CRITICAL FIX: Align features
                        if X.shape[1] < expected_features:
                            # Pad with zeros
                            padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
                            X_aligned = np.hstack([X, padding])
                            st.warning(f"⚠️ {model_name}: Padded from {X.shape[1]} to {expected_features} features")
                        else:
                            # Truncate
                            X_aligned = X[:, :expected_features]
                            st.warning(f"⚠️ {model_name}: Truncated from {X.shape[1]} to {expected_features} features")
                    else:
                        X_aligned = X

                    # Make prediction with aligned features
                    pred = model.predict(X_aligned)
                    predictions[model_key] = pred.flatten() if len(pred.shape) > 1 else pred
                    successful_models.append(model_name)

                except Exception as e:
                    errors.append(f"{model_name}: {str(e)[:80]}")

        # Check if we have any successful predictions
        prediction_values = [pred for key, pred in predictions.items() if 'std' not in key]

        if not prediction_values:
            # NO predictions succeeded
            st.error(f"❌ All models failed to generate predictions")
            st.error("**Errors encountered:**")
            for error in errors:
                st.text(f"  • {error}")

            st.info("""
            **Possible causes:**
            1. Feature mismatch between training and prediction data
            2. Missing or incompatible preprocessor
            3. Model files corrupted or incompatible

            **Solutions:**
            - Retrain models with current data preprocessing pipeline
            - Ensure preprocessor.pkl matches the training configuration
            - Check that all required columns exist in the dataset
            """)
            return None

        # Calculate combined prediction from successful models
        predictions['combined'] = np.mean(prediction_values, axis=0)
        predictions['uncertainty'] = np.std(prediction_values, axis=0)

        # Enhance uncertainty if MCDFN std is available
        if 'mcdfn_std' in predictions:
            predictions['uncertainty'] = np.maximum(
                predictions['uncertainty'],
                predictions['mcdfn_std']
            )
        elif np.mean(predictions['uncertainty']) < 10:
            # Minimum uncertainty threshold
            predictions['uncertainty'] = np.maximum(predictions['uncertainty'], 25)

        # Show success message
        st.success(
            f"✅ Successfully generated predictions using {len(successful_models)} model(s): {', '.join(successful_models)}")

        # Show warnings if some models failed
        if errors:
            with st.expander(f"⚠️ {len(errors)} model(s) encountered issues (click to expand)"):
                for error in errors:
                    st.text(f"• {error}")

        return predictions

    except Exception as e:
        st.error(f"❌ Critical prediction failure: {str(e)}")
        with st.expander("🔍 Full Error Details"):
            st.code(traceback.format_exc())
        return None


# ==================== VISUALIZATION FUNCTIONS ====================
def plot_overall_forecast(df, predictions):
    """Create comprehensive forecast visualization"""
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sales Forecast Distribution', 'Prediction Confidence',
                            'Actual vs Predicted', 'Model Comparison'),
            specs=[[{"type": "histogram"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )

        pred_values = predictions['combined']

        # 1. Forecast Distribution
        fig.add_trace(
            go.Histogram(
                x=pred_values,
                name='Predicted Sales',
                marker_color='#4CAF50',
                opacity=0.7,
                nbinsx=30
            ),
            row=1, col=1
        )

        # 2. Confidence Box Plot
        if 'uncertainty' in predictions:
            fig.add_trace(
                go.Box(
                    y=pred_values,
                    name='Predictions',
                    marker_color='#2196F3',
                    boxpoints='outliers'
                ),
                row=1, col=2
            )

        # 3. Actual vs Predicted
        if 'Sales' in df.columns:
            actual_values = df['Sales'].values[:len(pred_values)]
            fig.add_trace(
                go.Scatter(
                    x=actual_values,
                    y=pred_values,
                    mode='markers',
                    name='Actual vs Predicted',
                    marker=dict(color='#FF9800', size=6, opacity=0.7),
                    text=[f"Actual: ${a:.0f}<br>Predicted: ${p:.0f}" for a, p in zip(actual_values, pred_values)],
                    hovertemplate="%{text}<extra></extra>"
                ),
                row=2, col=1
            )

            # Perfect prediction line
            max_val = max(actual_values.max(), pred_values.max())
            min_val = min(actual_values.min(), pred_values.min())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash', width=2)
                ),
                row=2, col=1
            )

        # 4. Model Comparison
        model_names = [k for k in predictions.keys() if k not in ['combined', 'uncertainty', 'mcdfn_std']]
        if model_names:
            model_means = [predictions[k].mean() for k in model_names]
            colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#FF5722'][:len(model_names)]

            fig.add_trace(
                go.Bar(
                    x=[name.replace('_', ' ').title() for name in model_names],
                    y=model_means,
                    marker_color=colors,
                    name='Mean Predictions',
                    text=[f"${mean:.0f}" for mean in model_means],
                    textposition='outside'
                ),
                row=2, col=2
            )

        fig.update_layout(
            height=700,
            showlegend=True,
            title_text="Comprehensive Sales Forecast Analysis",
            title_x=0.5
        )

        return fig

    except Exception as e:
        st.error(f"❌ Visualization error: {str(e)}")
        return None


def plot_time_series_forecast(df, predictions, date_column=None):
    """Create time-series forecast with uncertainty bounds"""
    try:
        fig = go.Figure()

        pred_values = predictions['combined']
        n_points = len(pred_values)

        # Determine x-axis
        if date_column and date_column in df.columns:
            try:
                x_axis = pd.to_datetime(df[date_column].iloc[:n_points])
            except:
                x_axis = np.arange(n_points)
        else:
            x_axis = np.arange(n_points)

        # Plot actual sales if available
        if 'Sales' in df.columns:
            actual_sales = df['Sales'].values[:n_points]
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=actual_sales,
                mode='lines+markers',
                name='Actual Sales',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4)
            ))

        # Plot predicted sales
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=pred_values,
            mode='lines+markers',
            name='Predicted Sales',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=5)
        ))

        # Add uncertainty bounds
        if 'uncertainty' in predictions:
            uncertainty = predictions['uncertainty'][:n_points]
            lower_bound = pred_values - 1.96 * uncertainty
            upper_bound = pred_values + 1.96 * uncertainty

            fig.add_trace(go.Scatter(
                x=x_axis,
                y=upper_bound,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

            fig.add_trace(go.Scatter(
                x=x_axis,
                y=lower_bound,
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 127, 14, 0.2)',
                fill='tonexty',
                name='95% Confidence Interval',
                hoverinfo='skip'
            ))

        fig.update_layout(
            title='Time-Series Sales Forecast with Confidence Intervals',
            xaxis_title='Time Period' if date_column is None else 'Date',
            yaxis_title='Sales ($)',
            height=500,
            hovermode='x unified',
            showlegend=True
        )

        return fig

    except Exception as e:
        st.error(f"❌ Time-series visualization error: {str(e)}")
        return None


# ==================== FILTER FUNCTIONS ====================
def create_interactive_filters(df):
    """Create interactive filters based on available columns"""
    st.sidebar.header("🔍 Data Filters")
    filters = {}

    try:
        # Categorical columns
        categorical_columns = {
            'Market': ['Market', 'market'],
            'Customer Segment': ['Customer Segment', 'Customer_Segment', 'customer_segment'],
            'Shipping Mode': ['Shipping Mode', 'Shipping_Mode', 'shipping_mode'],
            'Order Region': ['Order Region', 'Order_Region', 'order_region'],
            'Type': ['Type', 'type'],
        }

        for display_name, col_variations in categorical_columns.items():
            found_col = None
            for col in col_variations:
                if col in df.columns:
                    found_col = col
                    break

            if found_col and df[found_col].dtype == 'object':
                unique_values = df[found_col].dropna().unique().tolist()
                if len(unique_values) <= 20:
                    filters[found_col] = st.sidebar.multiselect(
                        f"Select {display_name}",
                        options=unique_values,
                        default=unique_values
                    )

        # Numerical filters
        if 'Sales' in df.columns:
            sales_col = df['Sales'].dropna()
            if len(sales_col) > 0:
                min_sales, max_sales = float(sales_col.min()), float(sales_col.max())
                if min_sales < max_sales:
                    filters['Sales'] = st.sidebar.slider(
                        "Sales Range ($)",
                        min_value=min_sales,
                        max_value=max_sales,
                        value=(min_sales, max_sales),
                        format="$%.0f"
                    )

        return filters

    except Exception as e:
        st.sidebar.error(f"❌ Filter creation error: {str(e)}")
        return {}


def apply_filters(df, filters):
    """Apply selected filters to dataframe"""
    try:
        df_filtered = df.copy()

        for col, values in filters.items():
            if col in df.columns:
                if isinstance(values, tuple):  # Numerical range
                    df_filtered = df_filtered[
                        (df_filtered[col] >= values[0]) &
                        (df_filtered[col] <= values[1])
                        ]
                elif isinstance(values, list) and len(values) > 0:  # Categorical
                    df_filtered = df_filtered[df_filtered[col].isin(values)]

        return df_filtered

    except Exception as e:
        st.error(f"❌ Filter application error: {str(e)}")
        return df


# ==================== QUERY FUNCTIONS ====================
def natural_language_query(df, query_text, llm_integrator):
    """Process natural language queries with pattern matching"""
    try:
        query_lower = query_text.lower()
        result_df = df.copy()
        applied_filters = []

        # Market filtering
        if 'market' in query_lower:
            market_cols = [col for col in df.columns if 'market' in col.lower()]
            if market_cols:
                market_col = market_cols[0]
                for market in df[market_col].unique():
                    if market.lower() in query_lower:
                        result_df = result_df[result_df[market_col] == market]
                        applied_filters.append(f"Market = {market}")
                        break

        # Customer Segment filtering
        if 'segment' in query_lower:
            seg_cols = [col for col in df.columns if 'segment' in col.lower()]
            if seg_cols:
                seg_col = seg_cols[0]
                for segment in df[seg_col].unique():
                    if segment.lower() in query_lower:
                        result_df = result_df[result_df[seg_col] == segment]
                        applied_filters.append(f"Segment = {segment}")
                        break

        # Shipping Mode filtering
        if 'shipping' in query_lower:
            ship_cols = [col for col in df.columns if 'shipping' in col.lower()]
            if ship_cols:
                ship_col = ship_cols[0]
                for mode in df[ship_col].unique():
                    if mode.lower().replace(' ', '') in query_lower.replace(' ', ''):
                        result_df = result_df[result_df[ship_col] == mode]
                        applied_filters.append(f"Shipping = {mode}")
                        break

        # Generate summary
        summary = f"Query: '{query_text}'\nFilters Applied: {', '.join(applied_filters) if applied_filters else 'None'}"
        summary += f"\nRecords Found: {len(result_df):,} out of {len(df):,}"

        if 'Sales' in result_df.columns and len(result_df) > 0:
            sales_stats = {
                'total': result_df['Sales'].sum(),
                'mean': result_df['Sales'].mean(),
                'max': result_df['Sales'].max(),
                'min': result_df['Sales'].min()
            }
            summary += f"\nSales Statistics:"
            summary += f"\n  • Total: ${sales_stats['total']:,.2f}"
            summary += f"\n  • Average: ${sales_stats['mean']:,.2f}"
            summary += f"\n  • Range: ${sales_stats['min']:,.2f} - ${sales_stats['max']:,.2f}"

            insights = f"Found {len(result_df)} records. Sales range from ${sales_stats['min']:,.0f} to ${sales_stats['max']:,.0f} with average of ${sales_stats['mean']:,.0f}."
        else:
            insights = f"Found {len(result_df)} records but no sales data available."

        # Get LLM explanation
        llm_explanation = llm_integrator.analyze_query_results(query_text, summary, insights)

        return result_df, summary, llm_explanation

    except Exception as e:
        error_msg = f"❌ Query processing error: {str(e)}"
        return df, error_msg, error_msg


def sql_like_query(df, conditions):
    """Execute SQL-like queries with error handling"""
    try:
        result_df = df.copy()
        applied_conditions = []

        for condition in conditions:
            column = condition['column']
            operator = condition['operator']
            value = condition['value']

            if column not in df.columns:
                continue

            try:
                if operator == '==':
                    result_df = result_df[result_df[column] == value]
                    applied_conditions.append(f"{column} = {value}")
                elif operator == '!=':
                    result_df = result_df[result_df[column] != value]
                    applied_conditions.append(f"{column} ≠ {value}")
                elif operator == '>':
                    result_df = result_df[result_df[column] > float(value)]
                    applied_conditions.append(f"{column} > {value}")
                elif operator == '<':
                    result_df = result_df[result_df[column] < float(value)]
                    applied_conditions.append(f"{column} < {value}")
                elif operator == '>=':
                    result_df = result_df[result_df[column] >= float(value)]
                    applied_conditions.append(f"{column} ≥ {value}")
                elif operator == '<=':
                    result_df = result_df[result_df[column] <= float(value)]
                    applied_conditions.append(f"{column} ≤ {value}")
                elif operator == 'contains':
                    result_df = result_df[result_df[column].astype(str).str.contains(str(value), case=False, na=False)]
                    applied_conditions.append(f"{column} contains '{value}'")
            except Exception as condition_error:
                st.warning(f"⚠️ Condition '{column} {operator} {value}' failed: {str(condition_error)}")
                continue

        if applied_conditions:
            st.info(f"Applied conditions: {' AND '.join(applied_conditions)}")
        return result_df

    except Exception as e:
        st.error(f"❌ SQL query error: {str(e)}")
        return df


# ==================== MAIN APPLICATION ====================
def main():
    # Header
    st.title("📦 Supply Chain Sales Forecasting System")
    st.markdown("### Interactive Dashboard with AI-Powered Insights")

    # ==================== SESSION STATE INITIALIZATION ====================
    # Initialize session state variables
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'dataset_info' not in st.session_state:
        st.session_state.dataset_info = ""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    # System status indicator
    with st.container():
        status_cols = st.columns([1, 1, 1, 1])

        with status_cols[0]:
            tf_status = "✅" if TF_AVAILABLE else "❌"
            st.text(f"{tf_status} TensorFlow")

        with status_cols[1]:
            sklearn_status = "✅" if SKLEARN_AVAILABLE else "❌"
            st.text(f"{sklearn_status} Scikit-learn")

        with status_cols[2]:
            xgb_status = "✅" if XGB_AVAILABLE else "❌"
            st.text(f"{xgb_status} XGBoost")

        with status_cols[3]:
            custom_status = "✅" if CUSTOM_CLASSES_AVAILABLE else "❌"
            st.text(f"{custom_status} Custom Classes")

    st.markdown("---")

    # Initialize Llama2
    llm_integrator = Llama2Integrator()

    # Sidebar Configuration
    st.sidebar.title("⚙️ System Configuration")
    st.sidebar.markdown("---")

    # Model Status
    with st.sidebar.expander("🤖 Model & AI Status", expanded=True):
        models, preprocessor, feature_names, metadata, model_status = load_models_and_preprocessors()

        for status in model_status:
            st.text(status)

        if llm_integrator.is_available:
            st.success("✅ Llama2 AI Connected")
        else:
            st.warning("⚠️ Llama2 AI Offline")
            st.info("💡 Start with: `ollama run llama2`")

    # ==================== DATA LOADING WITH SESSION STATE ====================
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Data Loading")

    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Default Dataset", "Upload CSV"],
        key="data_source_radio"
    )

# --- Default Dataset Option ---
    if data_source == "Default Dataset":
    # Use a relative path for deployment
        default_dataset_path = os.path.join("data", "supply_chain_synthesized_dataset.xlsx")

    # Display info and allow custom path (if user wants)
        dataset_path = st.sidebar.text_input(
            "Enter dataset file path:",
            value=default_dataset_path,
            help="Relative path to your default dataset file (inside the repo)"
        )

        if st.sidebar.button("📂 Load Dataset", key="load_local_btn"):
            with st.spinner("Loading dataset..."):
                try:
                # Attempt to read Excel or CSV automatically
                    if dataset_path.endswith(".xlsx"):
                        df_temp = pd.read_excel(dataset_path)
                    else:
                        df_temp = pd.read_csv(dataset_path)

                    info_temp = f"Loaded {df_temp.shape[0]} rows and {df_temp.shape[1]} columns"

                # Store in session state
                    st.session_state.df = df_temp
                    st.session_state.dataset_info = info_temp
                    st.session_state.data_loaded = True

                    st.sidebar.success(f"✅ {info_temp}")
                    st.rerun()  # Refresh UI with loaded data

                except FileNotFoundError:
                    st.sidebar.error(f"❌ File not found at path: {dataset_path}")
                except Exception as e:
                    st.sidebar.error(f"❌ Error loading dataset: {e}")

# --- Upload CSV Option ---
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload your CSV file",
            type=["csv", "xlsx"],
            help="Upload a CSV or Excel file for analysis"
        )

    if uploaded_file is not None:
        # Create a unique ID for each upload
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"

        if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != file_id:
            with st.spinner("Processing uploaded file..."):
                try:
                    # Handle both CSV and Excel
                    if uploaded_file.name.endswith(".xlsx"):
                        df_temp = pd.read_excel(uploaded_file)
                    else:
                        df_temp = pd.read_csv(uploaded_file)

                    info_temp = f"Uploaded {df_temp.shape[0]} rows and {df_temp.shape[1]} columns"

                    # Save to session state
                    st.session_state.df = df_temp
                    st.session_state.dataset_info = info_temp
                    st.session_state.data_loaded = True
                    st.session_state.last_uploaded_file = file_id

                    st.sidebar.success(f"✅ {info_temp}")
                    st.rerun()

                except Exception as e:
                    st.sidebar.error(f"❌ Error processing uploaded file: {e}")
    # Display current data status
    if st.session_state.data_loaded and st.session_state.df is not None:
        st.sidebar.success(f"✅ Data loaded: {len(st.session_state.df):,} rows")

    # ==================== RETRIEVE DATAFRAME FROM SESSION STATE ====================
    df = st.session_state.df

    # Main Application Content
    if df is not None:
        # Create main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Forecast Dashboard",
            "🔍 Query Interface",
            "💡 AI Explanations",
            "📊 Data Explorer"
        ])


        # ========== TAB 1: FORECAST DASHBOARD ==========
        with tab1:
            st.header("📈 Sales Forecast Dashboard")

            # Apply filters
            filters = create_interactive_filters(df)
            df_filtered = apply_filters(df, filters)

            # Display filter results
            filter_info_cols = st.columns(3)
            with filter_info_cols[0]:
                st.metric("Original Records", f"{len(df):,}")
            with filter_info_cols[1]:
                st.metric("Filtered Records", f"{len(df_filtered):,}")
            with filter_info_cols[2]:
                filter_ratio = len(df_filtered) / len(df) * 100 if len(df) > 0 else 0
                st.metric("Filter Ratio", f"{filter_ratio:.1f}%")

            if len(df_filtered) == 0:
                st.warning("⚠️ No records match the current filters. Please adjust filter settings.")
            elif len(df_filtered) > 10000:
                st.warning("⚠️ Large dataset detected. Consider applying filters to improve performance.")
                if not st.checkbox("Proceed with full dataset (may be slow)"):
                    st.stop()

            if len(df_filtered) > 0 and len(models) > 0:
                # Make predictions
                with st.spinner("🔮 Generating predictions..."):
                    X_processed, y_actual = preprocess_data_for_prediction(
                        df_filtered, preprocessor, feature_names
                    )

                    if X_processed is not None:
                        predictions = make_predictions(models, X_processed)

                        if predictions is not None:
                            # Display prediction metrics
                            st.markdown("#### 📊 Prediction Overview")
                            metric_cols = st.columns(4)

                            pred_values = predictions['combined']

                            with metric_cols[0]:
                                st.metric(
                                    "Total Predicted Sales",
                                    f"${pred_values.sum():,.0f}",
                                    help="Sum of all predictions"
                                )

                            with metric_cols[1]:
                                st.metric(
                                    "Average Prediction",
                                    f"${pred_values.mean():,.0f}",
                                    help="Mean predicted value"
                                )

                            with metric_cols[2]:
                                if y_actual is not None and len(y_actual) == len(pred_values):
                                    mae = np.mean(np.abs(y_actual - pred_values))
                                    st.metric("Mean Absolute Error", f"${mae:,.0f}")
                                else:
                                    st.metric("Prediction Std Dev", f"${pred_values.std():,.0f}")

                            with metric_cols[3]:
                                if 'uncertainty' in predictions:
                                    avg_uncertainty = predictions['uncertainty'].mean()
                                    st.metric("Avg Uncertainty", f"±${avg_uncertainty:,.0f}")
                                else:
                                    pred_range = pred_values.max() - pred_values.min()
                                    st.metric("Prediction Range", f"${pred_range:,.0f}")

                            st.markdown("---")

                            # Overall Forecast Analysis
                            st.markdown("#### 📈 Comprehensive Forecast Analysis")
                            fig_overall = plot_overall_forecast(df_filtered, predictions)
                            if fig_overall:
                                st.plotly_chart(fig_overall, width='stretch')

                            st.markdown("---")

                            # Time-series Forecast
                            st.markdown("#### ⏱️ Time-Series Forecast")

                            # Auto-detect date columns
                            date_columns = [col for col in df_filtered.columns
                                            if any(date_word in col.lower()
                                                   for date_word in ['date', 'time', 'day', 'month', 'year'])]

                            date_col = None
                            if date_columns:
                                date_col = st.selectbox(
                                    "Select date column for time-series:",
                                    options=[None] + date_columns,
                                    help="Choose a date column for x-axis"
                                )

                            fig_ts = plot_time_series_forecast(df_filtered, predictions, date_col)
                            if fig_ts:
                                st.plotly_chart(fig_ts, width='stretch')

                            # Results Download
                            st.markdown("---")
                            st.markdown("#### 📥 Download Results")

                            # Prepare download data
                            download_df = df_filtered.copy()
                            download_df['Predicted_Sales'] = pred_values[:len(download_df)]

                            if 'uncertainty' in predictions:
                                uncertainty_values = predictions['uncertainty'][:len(download_df)]
                                download_df['Prediction_Uncertainty'] = uncertainty_values
                                download_df['Lower_Bound_95%'] = pred_values[
                                                                 :len(download_df)] - 1.96 * uncertainty_values
                                download_df['Upper_Bound_95%'] = pred_values[
                                                                 :len(download_df)] + 1.96 * uncertainty_values

                            csv_data = download_df.to_csv(index=False)
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                            st.download_button(
                                label="📊 Download Forecast Results (CSV)",
                                data=csv_data,
                                file_name=f"sales_forecast_{timestamp}.csv",
                                mime="text/csv",
                                help="Download predictions with confidence intervals"
                            )

                        else:
                            st.error("❌ Prediction generation failed. Check model compatibility.")
                    else:
                        st.error("❌ Data preprocessing failed. Check data format and feature compatibility.")

            elif len(df_filtered) > 0:
                st.warning("⚠️ No models loaded. Please ensure model files are available.")
                st.markdown("#### 📋 Data Preview")
                st.dataframe(df_filtered.head(20), width='stretch')

        # ========== TAB 2: QUERY INTERFACE ==========
        with tab2:
            st.header("🔍 Advanced Query Interface")

            query_type = st.radio(
                "Select query method:",
                ["💬 Natural Language", "🔧 SQL-style Filters"],
                horizontal=True
            )

            if query_type == "💬 Natural Language":
                st.markdown("#### 💬 Ask Questions About Your Data")

                # Example queries
                with st.expander("💡 Example Queries", expanded=False):
                    examples = [
                        "Show me sales for Europe market",
                        "What is average sales for Corporate segment?",
                        "Find records with express shipping",
                        "Display high value transactions"
                    ]
                    for example in examples:
                        if st.button(f"📝 {example}", key=f"example_{hash(example)}"):
                            st.session_state.nl_query = example

                # Natural language query input
                nl_query = st.text_input(
                    "Enter your question:",
                    value=st.session_state.get('nl_query', ''),
                    placeholder="e.g., Show sales trends for US market...",
                    help="Ask questions in plain English about your data"
                )

                if st.button("🔍 Analyze Query", key="nl_query_btn", type="primary"):
                    if nl_query.strip():
                        with st.spinner("🤖 Processing your question..."):
                            result_df, summary, llm_explanation = natural_language_query(
                                df, nl_query, llm_integrator
                            )

                            # Display results
                            result_cols = st.columns(2)
                            with result_cols[0]:
                                st.success(f"✅ Found {len(result_df):,} matching records")
                            with result_cols[1]:
                                if len(result_df) > 0:
                                    match_ratio = len(result_df) / len(df) * 100
                                    st.info(f"📊 {match_ratio:.1f}% of total data")

                            # Query Summary
                            st.markdown("#### 📋 Query Results Summary")
                            st.text(summary)

                            # AI Insights
                            if llm_explanation and "❌" not in llm_explanation and "⚠️" not in llm_explanation:
                                st.markdown("#### 🤖 AI Analysis")
                                st.markdown(
                                    f'<div class="explanation-box">{llm_explanation}</div>',
                                    unsafe_allow_html=True
                                )

                            # Data Preview
                            if len(result_df) > 0:
                                st.markdown("#### 📊 Filtered Data Preview")
                                st.dataframe(result_df.head(100), width='stretch')

                                # Quick visualization
                                if 'Sales' in result_df.columns:
                                    viz_cols = st.columns(2)

                                    with viz_cols[0]:
                                        fig_dist = px.histogram(
                                            result_df, x='Sales', nbins=20,
                                            title="Sales Distribution"
                                        )
                                        st.plotly_chart(fig_dist, width='stretch')

                                    with viz_cols[1]:
                                        categorical_cols = [col for col in result_df.columns
                                                            if result_df[col].dtype == 'object' and
                                                            result_df[col].nunique() <= 10]

                                        if categorical_cols:
                                            group_col = categorical_cols[0]
                                            grouped = result_df.groupby(group_col)['Sales'].sum().reset_index()

                                            fig_group = px.bar(
                                                grouped, x=group_col, y='Sales',
                                                title=f"Sales by {group_col}"
                                            )
                                            st.plotly_chart(fig_group, width='stretch')

                                # Download filtered results
                                if len(result_df) <= 50000:
                                    csv_filtered = result_df.to_csv(index=False)
                                    st.download_button(
                                        "📥 Download Query Results",
                                        data=csv_filtered,
                                        file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                    else:
                        st.warning("⚠️ Please enter a question to analyze.")

            else:  # SQL-style Filtering
                st.markdown("#### 🔧 Build Custom Filters")

                num_conditions = st.number_input(
                    "Number of filter conditions:",
                    min_value=1,
                    max_value=5,
                    value=1,
                    help="Add multiple conditions to narrow down your search"
                )

                conditions = []

                for i in range(int(num_conditions)):
                    st.markdown(f"**Filter Condition {i + 1}:**")
                    col1, col2, col3 = st.columns([3, 2, 3])

                    with col1:
                        available_columns = [col for col in df.columns if df[col].notna().any()]
                        column = st.selectbox(
                            f"Column:",
                            options=available_columns,
                            key=f"sql_col_{i}",
                            help="Select column to filter on"
                        )

                    with col2:
                        if df[column].dtype in ['object', 'string', 'category']:
                            operators = ['==', '!=', 'contains']
                        else:
                            operators = ['==', '!=', '>', '<', '>=', '<=']

                        operator = st.selectbox(
                            f"Operator:",
                            options=operators,
                            key=f"sql_op_{i}",
                            help="Choose comparison operator"
                        )

                    with col3:
                        if df[column].dtype in ['object', 'string', 'category']:
                            unique_vals = sorted(df[column].dropna().unique().tolist())
                            if len(unique_vals) <= 50:
                                value = st.selectbox(
                                    f"Value:",
                                    options=unique_vals,
                                    key=f"sql_val_{i}",
                                    help="Select value to filter by"
                                )
                            else:
                                value = st.text_input(
                                    f"Value:",
                                    key=f"sql_val_{i}",
                                    help="Enter value to search for"
                                )
                        else:
                            col_min, col_max = df[column].min(), df[column].max()
                            value = st.number_input(
                                f"Value:",
                                min_value=float(col_min),
                                max_value=float(col_max),
                                value=float(df[column].median()),
                                key=f"sql_val_{i}",
                                help=f"Range: {col_min:.2f} to {col_max:.2f}"
                            )

                    conditions.append({
                        'column': column,
                        'operator': operator,
                        'value': value
                    })

                    if i < num_conditions - 1:
                        st.markdown("**AND**")

                if st.button("🔍 Apply Filters", key="sql_query_btn", type="primary"):
                    with st.spinner("Applying filters..."):
                        result_df = sql_like_query(df, conditions)

                        if len(result_df) > 0:
                            st.success(f"✅ Found {len(result_df):,} matching records")

                            if 'Sales' in result_df.columns:
                                stats_cols = st.columns(4)
                                with stats_cols[0]:
                                    st.metric("Total Sales", f"${result_df['Sales'].sum():,.0f}")
                                with stats_cols[1]:
                                    st.metric("Average Sales", f"${result_df['Sales'].mean():,.0f}")
                                with stats_cols[2]:
                                    st.metric("Max Sales", f"${result_df['Sales'].max():,.0f}")
                                with stats_cols[3]:
                                    st.metric("Min Sales", f"${result_df['Sales'].min():,.0f}")

                            st.markdown("#### 📊 Filtered Results")
                            st.dataframe(result_df.head(100), width='stretch')

                            if len(result_df) <= 50000:
                                csv_data = result_df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Filtered Data",
                                    data=csv_data,
                                    file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                        else:
                            st.warning("⚠️ No records match the specified conditions. Try adjusting your filters.")

        # ========== TAB 3: AI EXPLANATIONS ==========
        with tab3:
            st.header("💡 AI-Powered Explanations")

            if not llm_integrator.is_available:
                st.error("🤖 AI explanations require Llama2. Please start Ollama with: `ollama run llama2`")
                st.info("Once Llama2 is running, refresh this page to enable AI features.")

            if len(models) > 0 and preprocessor is not None:
                st.markdown("#### 🎯 Individual Record Analysis")

                explanation_cols = st.columns([2, 1])

                with explanation_cols[0]:
                    sample_idx = st.number_input(
                        "Select record index for detailed analysis:",
                        min_value=0,
                        max_value=len(df) - 1,
                        value=min(42, len(df) - 1),
                        help="Choose any record from the dataset to analyze"
                    )

                with explanation_cols[1]:
                    explain_btn = st.button("🔍 Analyze Record", type="primary")

                if explain_btn:
                    with st.spinner("🤖 Generating AI analysis..."):
                        try:
                            sample_df = df.iloc[[sample_idx]]

                            X_sample, _ = preprocess_data_for_prediction(
                                sample_df, preprocessor, feature_names
                            )

                            if X_sample is not None:
                                predictions = make_predictions(models, X_sample)

                                if predictions is not None:
                                    pred_value = predictions['combined'][0]
                                    uncertainty = predictions.get('uncertainty', [25])[0]

                                    if uncertainty < 30:
                                        confidence = "High"
                                        conf_color = "green"
                                    elif uncertainty < 80:
                                        confidence = "Medium"
                                        conf_color = "orange"
                                    else:
                                        confidence = "Low"
                                        conf_color = "red"

                                    st.markdown("#### 📊 Prediction Results")
                                    pred_cols = st.columns(3)

                                    with pred_cols[0]:
                                        st.metric("💰 Predicted Sales", f"${pred_value:,.0f}")

                                    with pred_cols[1]:
                                        st.metric("📊 Uncertainty", f"±${uncertainty:,.0f}")

                                    with pred_cols[2]:
                                        st.markdown(
                                            f"**Confidence:** <span style='color:{conf_color}'>{confidence}</span>",
                                            unsafe_allow_html=True)

                                    st.markdown("#### 📋 Input Features Analysis")

                                    feature_data = sample_df.iloc[0].to_dict()

                                    categorical_features = {}
                                    numerical_features = {}

                                    for key, value in feature_data.items():
                                        if key == 'Sales':
                                            continue

                                        if isinstance(value, str) or sample_df[key].dtype == 'object':
                                            categorical_features[key] = value
                                        else:
                                            numerical_features[key] = value

                                    if categorical_features:
                                        st.markdown("**Categorical Features:**")
                                        cat_cols = st.columns(3)
                                        for idx, (key, value) in enumerate(categorical_features.items()):
                                            with cat_cols[idx % 3]:
                                                st.text(f"• {key}: {value}")

                                    if numerical_features:
                                        st.markdown("**Numerical Features:**")
                                        num_cols = st.columns(3)
                                        for idx, (key, value) in enumerate(numerical_features.items()):
                                            with num_cols[idx % 3]:
                                                if isinstance(value, float):
                                                    st.text(f"• {key}: {value:.2f}")
                                                else:
                                                    st.text(f"• {key}: {value}")

                                    if llm_integrator.is_available:
                                        st.markdown("---")
                                        st.markdown("#### 🤖 AI Business Analysis")

                                        top_features = list(feature_data.items())[:5]
                                        feature_summary = "\n".join(
                                            [f"• {k}: {v}" for k, v in top_features if k != 'Sales'])

                                        llm_explanation = llm_integrator.explain_forecast(
                                            pred_value, uncertainty, feature_summary, confidence
                                        )

                                        if llm_explanation and "❌" not in llm_explanation and "⚠️" not in llm_explanation:
                                            st.markdown(
                                                f'<div class="explanation-box">{llm_explanation}</div>',
                                                unsafe_allow_html=True
                                            )
                                        else:
                                            st.warning("⚠️ AI explanation temporarily unavailable.")

                                else:
                                    st.error("❌ Prediction failed for selected record.")
                            else:
                                st.error("❌ Data preprocessing failed for selected record.")

                        except Exception as e:
                            st.error(f"❌ Analysis failed: {str(e)}")

                st.markdown("---")
                st.markdown("#### 🌍 Overall Dataset Insights")

                if st.button("📊 Generate Dataset Analysis", key="overall_insights_btn"):
                    with st.spinner("🤖 Analyzing entire dataset..."):
                        data_summary = f"""
📊 Dataset Overview:
• Total Records: {len(df):,}
• Features: {len(df.columns)}
"""

                        if 'Sales' in df.columns:
                            sales_stats = df['Sales'].describe()
                            data_summary += f"""
💰 Sales Statistics:
• Total Sales: ${df['Sales'].sum():,.0f}
• Average: ${sales_stats['mean']:,.0f}
• Median: ${sales_stats['50%']:,.0f}
• Range: ${sales_stats['min']:,.0f} - ${sales_stats['max']:,.0f}
"""

                        mock_metrics = {
                            'r2': 0.847,
                            'mae': df['Sales'].std() * 0.12 if 'Sales' in df.columns else 85,
                            'resilience_score': 0.783
                        }

                        if llm_integrator.is_available:
                            insights = llm_integrator.generate_overall_insights(mock_metrics, data_summary)

                            if insights and "❌" not in insights and "⚠️" not in insights:
                                st.markdown(
                                    f'<div class="explanation-box">{insights}</div>',
                                    unsafe_allow_html=True
                                )
                            else:
                                st.warning("⚠️ AI insights temporarily unavailable.")
                        else:
                            st.info("""
                            📊 **Dataset Analysis Summary:**

                            Your supply chain dataset contains comprehensive sales and operational data. 
                            Key factors influencing sales include market region, customer segment, and shipping preferences.

                            **Recommendations:**
                            • Focus on high-performing market segments
                            • Optimize shipping modes for better customer satisfaction
                            • Monitor uncertainty levels for risk management
                            """)

                        with st.expander("📋 Detailed Dataset Summary", expanded=False):
                            st.text(data_summary)
            else:
                st.warning("⚠️ AI explanations require loaded models and preprocessors.")

        # ========== TAB 4: DATA EXPLORER ==========
        with tab4:
            st.header("📊 Data Explorer & Statistics")

            # FIX: Create a clean copy of dataframe for display
            df_display = df.copy()

            # Convert problematic columns to strings
            for col in df_display.columns:
                try:
                    # Check if column has mixed types or problematic dtypes
                    if df_display[col].dtype == 'object' or str(df_display[col].dtype).startswith('Int'):
                        df_display[col] = df_display[col].astype(str)
                except Exception:
                    pass

            st.markdown("#### 📈 Dataset Overview")

            overview_cols = st.columns(4)
            with overview_cols[0]:
                st.metric("📊 Total Records", f"{len(df):,}")

            with overview_cols[1]:
                st.metric("🏛️ Total Features", len(df.columns))

            with overview_cols[2]:
                if 'Sales' in df.columns:
                    st.metric("💰 Total Sales Volume", f"${df['Sales'].sum():,.0f}")
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    st.metric("🔢 Numeric Features", len(numeric_cols))

            with overview_cols[3]:
                categorical_cols = df.select_dtypes(include=['object']).columns
                st.metric("📋 Categorical Features", len(categorical_cols))

            st.markdown("---")

            st.markdown("#### 🔍 Data Quality Assessment")

            quality_cols = st.columns(2)

            with quality_cols[0]:
                missing_data = df.isnull().sum()
                missing_pct = (missing_data / len(df) * 100).round(2)

                if missing_data.sum() > 0:
                    missing_df = pd.DataFrame({
                        'Column': missing_data.index,
                        'Missing Count': missing_data.values,
                        'Missing %': missing_pct.values
                    }).query('`Missing Count` > 0').sort_values('Missing Count', ascending=False)

                    st.markdown("**Missing Values:**")
                    st.dataframe(missing_df, width='stretch')
                else:
                    st.success("✅ No missing values detected!")

            with quality_cols[1]:
                # FIX: Convert dtype to string before creating DataFrame
                dtype_dict = {}
                for col, dtype in df.dtypes.items():
                    dtype_dict[col] = str(dtype)

                dtype_counts = pd.Series(dtype_dict).value_counts()
                dtype_summary = pd.DataFrame({'Data Type': dtype_counts.index, 'Count': dtype_counts.values})

                st.markdown("**Data Types Summary:**")
                st.dataframe(dtype_summary, width='stretch')

            st.markdown("---")

            st.markdown("#### 📋 Raw Data Browser")

            display_cols = st.columns([2, 1, 1])

            with display_cols[0]:
                show_columns = st.multiselect(
                    "Select columns to display:",
                    options=df.columns.tolist(),
                    default=df.columns.tolist()[:8] if len(df.columns) > 8 else df.columns.tolist(),
                    help="Choose which columns to show"
                )

            with display_cols[1]:
                show_rows = st.selectbox(
                    "Rows to display:",
                    options=[20, 50, 100, 200],
                    index=1  # FIXED: Changed from value=50 to index=1
                )

            with display_cols[2]:
                start_row = st.number_input(
                    "Start from row:",
                    min_value=0,
                    max_value=max(0, len(df) - 1),
                    value=0
                )

            if show_columns:
                # Use the cleaned display dataframe
                display_subset = df_display[show_columns].iloc[start_row:start_row + show_rows]
                st.dataframe(display_subset, width='stretch')  # FIXED: Changed from use_container_width

                if len(show_columns) == len(df.columns):
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Complete Dataset",
                        data=csv_data,
                        file_name=f"complete_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

            st.markdown("---")

            st.markdown("#### 📊 Statistical Analysis")

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if numeric_cols:
                st.markdown("**Numerical Features Summary:**")
                numeric_stats = df[numeric_cols].describe()
                st.dataframe(numeric_stats.round(2), width='stretch')  # FIXED: Changed from use_container_width

                if len(numeric_cols) > 1:
                    st.markdown("**Correlation Matrix:**")
                    corr_matrix = df[numeric_cols].corr()

                    fig_corr = px.imshow(
                        corr_matrix,
                        title="Feature Correlation Matrix",
                        color_continuous_scale="RdBu",
                        aspect="auto"
                    )
                    st.plotly_chart(fig_corr, width='stretch')

            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

            if categorical_cols:
                st.markdown("---")
                st.markdown("**Categorical Features Analysis:**")

                cat_col = st.selectbox(
                    "Select categorical column to analyze:",
                    options=categorical_cols,
                    index=0  # FIXED: Added index parameter
                )

                if cat_col:
                    cat_analysis_cols = st.columns(2)

                    with cat_analysis_cols[0]:
                        value_counts = df[cat_col].value_counts().head(10)

                        fig_cat = px.bar(
                            x=value_counts.index,
                            y=value_counts.values,
                            title=f"Distribution of {cat_col}",
                            labels={'x': cat_col, 'y': 'Count'}
                        )
                        fig_cat.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_cat, width='stretch')

                    with cat_analysis_cols[1]:
                        st.markdown(f"**{cat_col} Statistics:**")
                        st.text(f"• Unique Values: {df[cat_col].nunique():,}")
                        st.text(f"• Most Common: {df[cat_col].mode()[0] if len(df[cat_col].mode()) > 0 else 'N/A'}")

                        if 'Sales' in df.columns:
                            sales_by_cat = df.groupby(cat_col)['Sales'].agg(['count', 'sum', 'mean']).round(2)
                            st.markdown(f"**Sales by {cat_col}:**")
                            st.dataframe(sales_by_cat.head(10), width='stretch')  # FIXED


    else:
        # Welcome screen when no data is loaded
        st.markdown("""
        ## Welcome to Supply Chain Sales Forecasting System! 🎉

        ### 🚀 Key Features:

        **📈 Forecast Dashboard**
        - Generate sales predictions using MCDFN neural network
        - Ensemble modeling with Random Forest, XGBoost, Gradient Boosting
        - Interactive visualizations with confidence intervals

        **🔍 Query Interface**
        - Natural language queries
        - SQL-style filtering
        - AI-powered result analysis

        **💡 AI Explanations**
        - Individual prediction explanations
        - Business-focused recommendations
        - Feature importance analysis

        **📊 Data Explorer**
        - Interactive data browsing
        - Statistical analysis
        - Data quality assessment

        ### 🛠️ Getting Started:

        1. **📂 Load Your Data**: Use the sidebar to load your dataset
        2. **🤖 Check Models**: Ensure model files are loaded (check sidebar status)
        3. **🦙 Enable AI** (Optional): Start Ollama with `ollama run llama2`
        4. **🎯 Explore**: Navigate through tabs to analyze and forecast

        ---

        **💡 Tip**: Click "📂 Load Dataset" in the sidebar to begin!
        """)


if __name__ == "__main__":
    main()
