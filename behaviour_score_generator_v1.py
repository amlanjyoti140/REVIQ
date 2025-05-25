import pandas as pd
import numpy as np


def score_generator(patients_df: pd.DataFrame, activity_df: pd.DataFrame, income_df: pd.DataFrame=None) -> pd.DataFrame:
    # --- Merge income info into patient table ---
    patients = patients_df.merge(income_df, left_on='annual_income_grade', right_on='grade', how='left')

    # --- Preprocess activity log ---
    activity = activity_df.copy()
    activity['time_stamp'] = pd.to_datetime(activity['time_stamp'])
    activity = activity.sort_values(['patient_id', 'time_stamp'])

    # Adherence ratio
    activity['adherence_ratio'] = activity.apply(
        lambda row: row['supply_days'] / row['prescribed_medication_days']
        if pd.notnull(row['supply_days']) and pd.notnull(row['prescribed_medication_days']) and row[
            'prescribed_medication_days'] > 0
        else np.nan,
        axis=1
    )

    # Short supply flag
    activity['supply_short'] = activity['supply_days'].apply(lambda x: 1 if pd.notnull(x) and x <= 15 else 0)

    # Is refill event
    activity['is_refill'] = activity['event_type'].str.lower().str.contains('refill')

    # --- Aggregate behavioral metrics ---
    agg = activity.groupby('patient_id').agg(
        num_short_refills=('supply_short', 'sum'),
        num_refill_events=('is_refill', 'sum'),
        short_supply_rate=('supply_short', 'mean'),
        mean_adherence_ratio=('adherence_ratio', 'mean'),
        avg_supply_days=('supply_days', 'mean'),
        event_count=('time_stamp', 'count'),
        last_activity=('time_stamp', 'max'),
        first_activity=('time_stamp', 'min'),
        unique_channels=('channel', pd.Series.nunique),
        reminder_response_count=('refill_reminder_response', lambda x: x.fillna(False).sum())
    ).reset_index()

    # Derived metric: multiple short refills rate
    agg['multiple_short_refills_rate'] = agg.apply(
        lambda row: row['num_short_refills'] / row['num_refill_events']
        if row['num_refill_events'] > 0 else 0,
        axis=1
    )

    # Merge patient + activity features
    df = patients.merge(agg, left_on='id', right_on='patient_id', how='left')

    # Time features
    now = pd.Timestamp.now()
    df['days_since_last'] = (now - df['last_activity']).dt.days
    df['active_days_span'] = (df['last_activity'] - df['first_activity']).dt.days

    # --- Normalize helper ---
    def normalize(val, min_val, max_val):
        return max(min((val - min_val) / (max_val - min_val + 1e-9), 1), 0) if pd.notnull(val) else 0

    def normalize_series(series, min_val, max_val):
        return ((series - min_val) / (max_val - min_val + 1e-9)).clip(lower=0, upper=1)


    # Income score: lower income → higher score
    df['income_score'] = df['income_range_low'].apply(lambda x: normalize(1_000_000 - x, 0, 1_000_000))

    # Occupation risk (manual rule-based)
    risky_jobs = ['unemployed', 'retired', 'part-time']
    df['occupation_risk'] = df['occupation'].str.lower().apply(
        lambda x: 1 if isinstance(x, str) and x in risky_jobs else 0.2)

    # # Condition severity score (assumed scale 1–5)
    # df['condition_score'] = df['patient_condition'].apply(lambda x: normalize(x, 1, 5))

    # Convert 'patient_condition' to numeric before applying normalize
    condition_map = {"acute": 0, "chronic": 1}
    df['condition_numeric'] = df['patient_condition'].map(condition_map)

    # Now you can safely normalize if needed
    df['condition_score'] = df['condition_numeric'].apply(lambda x: normalize(x, 0, 1))

    # Age score
    df['age_score'] = df['age'].apply(lambda x: normalize(x, 18, 100))

    # Digital access
    df['digital_access_score'] = df.apply(
        lambda row: 1 if pd.notnull(row['email']) and pd.notnull(row['phone']) else 0,
        axis=1
    )

    # Channel mismatch score
    df['channel_mismatch_score'] = df['unique_channels'].apply(lambda x: 1 - normalize(x if pd.notnull(x) else 0, 1, 5))

    # Reminder responsiveness gap
    df['notification_gap_score'] = df['reminder_response_count'].apply(lambda x: 1 - normalize(x, 0, 10))

    # --- Final Behavioral Scores ---

    # 1. Price Sensitivity Score (includes multiple short refills)
    df['score_price_sensitivity'] = (
            df['short_supply_rate'].fillna(0) * 0.3 +
            df['multiple_short_refills_rate'].fillna(0) * 0.2 +
            (1 - df['mean_adherence_ratio'].fillna(1)) * 0.2 +
            df['income_score'] * 0.2 +
            df['occupation_risk'] * 0.1
    )

    # 2. Awareness Gap Score
    df['score_awareness_gap'] = (
            normalize(df['days_since_last'].fillna(90), 0, 90) * 0.4 +
            (1 - df['condition_score']) * 0.3 +
            (1 - df['digital_access_score']) * 0.3
    )

    # 3. Coverage Confusion Score
    df['score_coverage_confusion'] = (
            df['age_score'] * 0.4 +
            (1 - df['digital_access_score']) * 0.3 +
            df['channel_mismatch_score'] * 0.3
    )

    # 4. Notification Response Score
    df['score_notification_response'] = (
            df['notification_gap_score'] * 0.6 +
            (1 - df['digital_access_score']) * 0.2 +
            df['channel_mismatch_score'] * 0.2
    )

    # Return the final DataFrame with scores
    return df
