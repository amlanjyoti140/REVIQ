CREATE TABLE activity_log (
    id TEXT PRIMARY KEY,
    patient_id INTEGER,
    event_type TEXT,
    supply_days INTEGER,
    prescribed_medication_days INTEGER,        -- NEW: Days prescribed by provider
    channel TEXT,
    time_stamp TEXT,
    event_outcome TEXT,                        -- NEW: success / failed / abandoned / sent / ignored
    refill_reminder_response BOOLEAN,          -- NEW: True if reminder was responded to
    session_duration INTEGER,                  -- NEW: Time spent in session (in seconds)
    attempt_count INTEGER                      -- NEW: Number of attempts made for this event
);
