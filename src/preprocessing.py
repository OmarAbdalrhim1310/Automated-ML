import pandas as pd
import numpy as np

def run_advanced_preprocessing(data_frame):
    """
    محرك التنظيف التلقائي: يتعامل مع القيم المفقودة، القيم الشاذة، ويهيئ البيانات للموديلات.
    """
    cleaned_df = data_frame.copy()
    report = []
    
    # 1. معالجة القيم المفقودة (Null Values)
    total_missing = cleaned_df.isnull().sum().sum()
    if total_missing > 0:
        report.append(f"🔍 **Missing Values:** Found {total_missing} empty cells.")
        for col in cleaned_df.columns:
            if cleaned_df[col].isnull().sum() > 0:
                if cleaned_df[col].dtype in ['int64', 'float64']:
                    mean_val = cleaned_df[col].mean()
                    cleaned_df[col] = cleaned_df[col].fillna(mean_val)
                    report.append(f"   • Imputed Numerical column `[{col}]` using Mean ({mean_val:.2f}).")
                else:
                    mode_val = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else "Unknown"
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                    report.append(f"   • Imputed Categorical column `[{col}]` using Mode ('{mode_val}').")
    else:
        report.append("🔍 **Missing Values:** 0 missing values found. Dataset is clean.")

    # 2. معالجة القيم الشاذة (Outliers) باستخدام Z-Score مبسط
    outlier_count = 0
    for col in cleaned_df.select_dtypes(include=[np.number]).columns:
        col_mean = cleaned_df[col].mean()
        col_std = cleaned_df[col].std()
        if col_std > 0:
            outliers = cleaned_df[(cleaned_df[col] - col_mean).abs() > (3 * col_std)]
            if len(outliers) > 0:
                outlier_count += len(outliers)
                lower_bound = col_mean - 3 * col_std
                upper_bound = col_mean + 3 * col_std
                cleaned_df[col] = np.clip(cleaned_df[col], lower_bound, upper_bound)
    
    if outlier_count > 0:
        report.append(f"🚨 **Outliers:** Handled {outlier_count} extreme anomalies via statistical clipping.")
    else:
        report.append("🚨 **Outliers:** No extreme anomalies detected.")
        
    return cleaned_df, report