import pandas as pd
import numpy as np

def normalize_series(series, min_val, max_val):
    """Vectorized normalization for a Pandas Series"""
    return ((series - min_val) / (max_val - min_val + 1e-9)).clip(lower=0, upper=1)

def score_generator(patients_df, activity_df, income_df):
    # Merge patient and activity data
    df = patients_df.merge(activity_df, left_on='id', right_on='patient_id', how='inner')

    # Convert income grade to numeric
    df['annual_income_grade'] = pd.to_numeric(df['annual_income_grade'], errors='coerce')

    # Fill missing numeric values
    df['supply_days'] = df['supply_days'].fillna(0)
    df['prescribed_medication_days'] = df['prescribed_medication_days'].fillna(df['supply_days'])
    df['refill_reminder_response'] = df['refill_reminder_response'].fillna(False).astype(int)
    df['session_duration'] = df['session_duration'].fillna(0)
    df['attempt_count'] = df['attempt_count'].fillna(1)

    # Derived features
    df['days_diff'] = df['prescribed_medication_days'] - df['supply_days']
    df['short_refill'] = (df['supply_days'] < 0.7 * df['prescribed_medication_days']).astype(int)
    df['short_refill_count'] = df.groupby('patient_id')['short_refill'].transform('sum')

    df['coverage_check'] = df['event_type'].str.lower().eq('coverage_check').astype(int)
    df['coverage_check_attempts'] = df.groupby('patient_id')['coverage_check'].transform('sum')

    df['coverage_check_fail'] = (
        (df['event_type'].str.lower() == 'coverage_check') &
        (df['event_outcome'].str.lower().isin(['failed', 'abandoned']))
    ).astype(int)
    df['coverage_check_fail_rate'] = df.groupby('patient_id')['coverage_check_fail'].transform('mean')

    df['reminder_event'] = df['event_type'].str.lower().eq('reminder').astype(int)
    df['reminder_ignored'] = (
        (df['event_type'].str.lower() == 'reminder') &
        (~df['refill_reminder_response'].astype(bool))
    ).astype(int)
    df['reminder_ignore_rate'] = df.groupby('patient_id')['reminder_ignored'].transform('mean')

    # Handle timestamps
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], errors='coerce')
    df['days_since_last'] = (pd.Timestamp.now() - df['time_stamp']).dt.days
    df['days_since_last'] = df['days_since_last'].fillna(90)

    df['avg_reminder_response_delay'] = df.groupby('patient_id')['days_since_last'].transform('mean')

    # Score calculations (rounded to 2 decimals)
    df['price_sensitivity_score'] = np.round(
        normalize_series(df['annual_income_grade'], 1, 4).rsub(1) * 0.6 +
        normalize_series(df['short_refill_count'], 0, 5) * 0.4,
        2
    )

    df['awareness_score'] = np.round(
        normalize_series(df['days_since_last'], 0, 90) * 0.4 +
        normalize_series(df['session_duration'], 0, 600) * 0.3 +
        df['refill_reminder_response'] * 0.3,
        2
    )

    df['coverage_confusion_score'] = np.round(
        normalize_series(df['coverage_check_attempts'], 0, 5) * 0.5 +
        df['coverage_check_fail_rate'].fillna(0) * 0.5,
        2
    )

    df['refill_reminder_score'] = np.round(
        df['reminder_ignore_rate'].fillna(0) * 0.5 +
        normalize_series(df['avg_reminder_response_delay'].fillna(0), 0, 72) * 0.5,
        2
    )

    # Get latest record per patient
    latest_records = df.sort_values('time_stamp').groupby('patient_id').tail(1)

    # Final dataframe with scores
    final_df = patients_df.merge(
        latest_records[[
            'patient_id',
            'price_sensitivity_score',
            'awareness_score',
            'coverage_confusion_score',
            'refill_reminder_score'
        ]],
        left_on='id', right_on='patient_id', how='left'
    ).drop(columns=['patient_id'])

    return final_df

def normalize(val, min_val, max_val):
    return max(min((val - min_val) / (max_val - min_val + 1e-9), 1), 0) if pd.notnull(val) else 0

def calculate_demo_score(row):
    # Normalize age (older patients may face higher adherence challenges)
    age_score = normalize(row['age'], 18, 90)

    # Normalize income grade (lower grade = lower income = higher potential barrier)
    income_score = normalize(row['annual_income_grade'], 1, 4)

    # Normalize dependents
    dependents_score = normalize(row['no_of_dependant'], 0, 5)

    # Gender score: assume some additional challenges
    gender = str(row['gender']).strip().lower()
    if gender == 'female':
        gender_score = 0.1
    elif gender == 'non-binary':
        gender_score = 0.15
    else:
        gender_score = 0.05

    # State-based adjustment for rural/underserved areas
    rural_states = ['MS', 'WV', 'AR', 'AL', 'KY', 'NM', 'MT', 'WY', 'AK']
    state_score = 0.1 if row['state'] in rural_states else 0.0

    # Final weighted demographic score
    demo_score = (
        0.25 * age_score +
        0.3 * (1 - income_score) +  # Lower income = higher challenge
        0.15 * dependents_score +
        0.2 * gender_score +
        0.1 * state_score
    )

    return round(min(demo_score, 1), 2)

def calculate_adherance_score(df:pd.DataFrame) -> pd.DataFrame :

    df['adherence_score'] = df.apply(lambda row: round(  # 0 = better, 1 = worse
        (1 - row['refill_reminder_score']) * 0.25 +
        (1 - row['price_sensitivity_score']) * 0.2 +
        (1 - row['awareness_score']) * 0.2 +
        (1 - row['coverage_confusion_score']) * 0.15 +
        calculate_demo_score(row) * 0.2,
        2
    ), axis=1)

    return df



