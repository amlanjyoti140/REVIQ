import configparser
import logging
import h2o
from h2o.automl import H2OAutoML
from reviq_helper import read_table_from_sqlite

# ---------- STEP 1: Load data from SQLite ----------

# Create a ConfigParser object
config = configparser.ConfigParser()
config.read('config.ini')

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get the logger
logger = logging.getLogger(__name__)

src_dir = config["DEFAULT"]["input_file_dir"]
input_patient_file_nm = config["DEFAULT"]["patient_file_name"]
input_activity_log_file_nm = config["DEFAULT"]["activity_log_file_name"]
sqlite_db_path = config["DEFAULT"]["sqlite_db_path"]

# ---------- STEP 1: Read patient info with score from database ----------

df_patient = read_table_from_sqlite(sqlite_db_path=sqlite_db_path,
                                    table_name="patient_matrix")

# ---- Step 3: Convert to H2OFrame ----
# Initialize H2O cluster
# Start H2O
h2o.init(max_mem_size_GB=4)
hf = h2o.H2OFrame(df_patient)

# ---- Step 4: Define target and features ----
# Identify target and features
target = 'adherence_score'  # The column you're predicting
ignore_cols = ['id', 'name', 'email', 'phone', 'address_line1', 'address_line2']
categorical_cols = ['gender', 'maritial_status', 'occupation', 'state', 'patient_condition']

# Tell H2O which columns are categorical
for col in categorical_cols:
    hf[col] = hf[col].asfactor()

# Define features by excluding target and ignored columns
features = [col for col in hf.columns if col not in ignore_cols + [target]]

# Split data
train, test = hf.split_frame(ratios=[0.8], seed=123)

# Run H2O AutoML
aml = H2OAutoML(max_runtime_secs=300, seed=1, sort_metric='RMSE')  # You can tune this
aml.train(x=features, y=target, training_frame=train)

# Show leaderboard
lb = aml.leaderboard
print(lb.head(rows=10))

# Evaluate model performance
perf = aml.leader.model_performance(test_data=test)
print(perf)

# Optional: Save the model
# model_path = h2o.save_model(model=aml.leader, path="best_model", force=True)

