import os
import pickle
import sys

import numpy as np
import pandas as pd
import utils
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

# Add parent directory to path to import from config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.constants import MOUSE_MOVEMENT_MODEL, SYMBOLS

DATA_FOLDER = "./data"
MODEL_FOLDER = "./models"

def load_and_filter_data(csv_name):
    print(f"Loading {csv_name}...")
    path = os.path.join(DATA_FOLDER, csv_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
        
    df = pd.read_csv(path, sep=";", quotechar="'")
    
    # Load symbols map
    sym_path = os.path.join(DATA_FOLDER, "symbols.csv")
    symbols = pd.read_csv(sym_path, sep=";", quotechar="'")
    symbol_map = {row['symbol_id']: row['latex'] for _, row in symbols.iterrows()}
    
    # Map and filter
    df['char'] = df['symbol_id'].map(symbol_map)
    df_clean = df[df['char'].isin(list(SYMBOLS))].copy()
    
    print(f" > Loaded {len(df_clean)} valid samples for {csv_name}")
    return df_clean

# ------------------------------------------------------------------
# Synthetic Bot Generator (Advanced)
# ------------------------------------------------------------------
def generate_smooth_bot_data(human_strokes_raw: List[List[Dict]]) -> List[List[Dict]]:
    """
    Takes real human strokes and 'robotizes' them:
    1. B-Spline smoothing to remove human tremor/jitter.
    2. Re-sampling to simulate constant velocity (perfect mouse control).
    """
    bot_strokes = []
    
    for stroke in human_strokes_raw:
        if len(stroke) < 4: 
            # Too short to smooth, just copy with constant time
            bot_strokes.append(stroke)
            continue
            
        x = np.array([p['x'] for p in stroke])
        y = np.array([p['y'] for p in stroke])
        
        try:
            # 1. B-Spline Smoothing (removes jitter)
            # s is the smoothing factor. Higher = smoother.
            tck, u = splprep([x, y], s=500) 
            unew = np.linspace(0, 1, len(x)) 
            out = splev(unew, tck)
            smooth_x, smooth_y = out[0], out[1]
            
            # 2. Constant Velocity Re-timing
            # Robots don't accelerate/decelerate like humans at corners
            new_points = []
            start_time = stroke[0]['time']
            # Assume bot draws at constant 10ms intervals
            for i in range(len(smooth_x)):
                new_points.append({
                    'x': smooth_x[i],
                    'y': smooth_y[i],
                    'time': int(start_time + (i * 15)) # Constant 15ms spacing
                })
            bot_strokes.append(new_points)
            
        except Exception:
            # Fallback if spline fails
            bot_strokes.append(stroke)
            
    return bot_strokes

# ------------------------------------------------------------------
# Dataset Construction
# ------------------------------------------------------------------
def build_dataset(df, generate_bots=False):
    X_kin = []  # Kinematic features (for Human vs Bot)
    X_all = []  # All features (for Recognition)
    y_lbl = []  # Character labels
    y_is_human = [] # 1 for human, 0 for bot
    
    print(f"Processing {len(df)} samples (Generating Bots: {generate_bots})...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # 1. Process Real Human Data
        strokes = utils.parse_strokes(row['data'])
        if not strokes: continue
        
        arr = utils.normalize_strokes(strokes)
        feats = utils.extract_features(arr, strokes)
        
        human_kin = [feats[k] for k in utils.FEATURE_NAMES_KINEMATIC]
        human_all = [feats[k] for k in utils.ALL_FEATURE_NAMES]
        
        X_kin.append(human_kin)
        X_all.append(human_all)
        y_lbl.append(row['char'])
        y_is_human.append(1) # Real human
        
        # 2. Generate Synthetic Bot Data (Only for Training)
        if generate_bots:
            bot_strokes = generate_smooth_bot_data(strokes)
            bot_arr = utils.normalize_strokes(bot_strokes)
            bot_feats = utils.extract_features(bot_arr, bot_strokes)
            
            bot_kin = [bot_feats[k] for k in utils.FEATURE_NAMES_KINEMATIC]
            # We don't care about recognition for bots (they can write any letter)
            # We just add them to the "Human vs Bot" dataset
            X_kin.append(bot_kin)
            y_is_human.append(0) # Fake bot

    return np.array(X_kin), np.array(X_all), np.array(y_lbl), np.array(y_is_human)

if __name__ == "__main__":
    # 1. Load Data
    train_df = load_and_filter_data("train-data.csv")
    test_df = load_and_filter_data("test-data.csv")

    # 2. Build Training Sets
    print("\n--- Building Training Set ---")
    X_train_kin, X_train_all, y_train_chars, y_train_hb = build_dataset(train_df, generate_bots=True)

    # 3. Build Test Sets (We don't generate bots for test char recognition, only check human accuracy)
    print("\n--- Building Test Set ---")
    _, X_test_all, y_test_chars, _ = build_dataset(test_df, generate_bots=False)
    
    # Create a separate test set for Human/Bot detection
    # We grab a subset of human test data and generate bots from it to test the detector
    print("--- Building Human vs Bot Validation Set ---")
    subset_test = test_df.head(500) # Use 500 samples for verification
    X_test_kin, _, _, y_test_hb = build_dataset(subset_test, generate_bots=True)

    print(f"\n[Training] Human vs Bot Detector (Samples: {len(X_train_kin)})...")
    hb_model = xgb.XGBClassifier(
        n_estimators=300, 
        max_depth=6, 
        learning_rate=0.1, 
        eval_metric='logloss',
        use_label_encoder=False
    )
    hb_model.fit(X_train_kin, y_train_hb)
    
    # Evaluate
    y_pred_hb = hb_model.predict(X_test_kin)
    acc_hb = accuracy_score(y_test_hb, y_pred_hb)
    print(f" >> Human vs Bot Accuracy: {acc_hb*100:.2f}%")
    print(confusion_matrix(y_test_hb, y_pred_hb))

    print("\n[Saving Model]...")
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    with open(os.path.join(MODEL_FOLDER, MOUSE_MOVEMENT_MODEL), "wb") as f:
        pickle.dump(hb_model, f)
        
    print("Done! Ready for main.py")  
