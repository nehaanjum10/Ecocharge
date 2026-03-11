import os
import numpy as np
import pandas as pd
import scipy.io

BASE_DIR   = os.path.join(os.path.dirname(__file__), '..')
RAW_DIR    = os.path.join(BASE_DIR, 'data', 'raw')
SAVE_PATH  = os.path.join(BASE_DIR, 'data', 'raw', 'battery_data.csv')
BATTERY_IDS = ['B0005', 'B0006', 'B0007', 'B0018']

def load_mat_file(battery_id):
    mat_path = os.path.join(RAW_DIR, f'{battery_id}.mat')
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"File not found: {mat_path}")
    mat = scipy.io.loadmat(mat_path, simplify_cells=True)
    print(f"[INFO] Loaded {battery_id}.mat")
    return mat

def extract_discharge_cycles(battery_id, mat):
    records = []
    try:
        battery_data = mat[battery_id]
        cycles = battery_data['cycle']
        cycle_number = 0
        for cycle in cycles:
            if cycle['type'] != 'discharge':
                continue
            cycle_number += 1
            data = cycle['data']
            try:
                voltage     = np.array(data['Voltage_measured']).flatten()
                current     = np.array(data['Current_measured']).flatten()
                temperature = np.array(data['Temperature_measured']).flatten()
                time        = np.array(data['Time']).flatten()
                capacity    = np.array(data['Capacity']).flatten()
                try:
                    curr_charge = float(np.array(data['Current_charge']).flatten()[-1])
                    volt_charge = float(np.array(data['Voltage_charge']).flatten()[-1])
                except:
                    curr_charge = 0.0
                    volt_charge = 0.0
                cycle_capacity = float(capacity[-1]) if len(capacity) > 0 else np.nan
                records.append({
                    'battery_id':           battery_id,
                    'cycle':                cycle_number,
                    'voltage_measured':     round(float(np.mean(voltage)),     4),
                    'current_measured':     round(float(np.mean(current)),     4),
                    'temperature_measured': round(float(np.mean(temperature)), 4),
                    'current_charge':       round(curr_charge, 4),
                    'voltage_charge':       round(volt_charge, 4),
                    'time':                 round(float(np.max(time)),         2),
                    'capacity':             round(cycle_capacity,              6),
                })
            except Exception as e:
                print(f"  [WARN] Skipping cycle {cycle_number}: {e}")
                continue
    except Exception as e:
        print(f"[ERROR] Could not parse {battery_id}: {e}")
        return pd.DataFrame()
    df = pd.DataFrame(records)
    print(f"  -> {battery_id}: {len(df)} discharge cycles extracted")
    return df

def load_nasa_dataset():
    all_dfs = []
    for battery_id in BATTERY_IDS:
        try:
            mat = load_mat_file(battery_id)
            df  = extract_discharge_cycles(battery_id, mat)
            if len(df) > 0:
                all_dfs.append(df)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
    if not all_dfs:
        raise RuntimeError("No .mat files found in data/raw/")
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n[INFO] Combined: {len(df):,} rows, {df['battery_id'].nunique()} batteries")
    return df

def generate_synthetic_battery_data(n_batteries=10, max_cycles=1000, seed=42):
    np.random.seed(seed)
    records = []
    for b_id in range(1, n_batteries + 1):
        initial_capacity = np.random.uniform(1.85, 1.95)
        degradation_rate = np.random.uniform(0.0003, 0.0007)
        base_temp        = np.random.uniform(22, 30)
        for cycle in range(1, max_cycles + 1):
            capacity    = max(0.0, initial_capacity * np.exp(-degradation_rate * cycle) + np.random.normal(0, 0.005))
            voltage     = np.random.uniform(3.5, 4.0) - (cycle / max_cycles) * 0.3
            current     = np.random.uniform(-2.0, -1.5)
            temperature = base_temp + (cycle / max_cycles) * 5 + np.random.normal(0, 0.5)
            records.append({
                'battery_id':           f'B{b_id:04d}',
                'cycle':                cycle,
                'voltage_measured':     round(voltage, 4),
                'current_measured':     round(current, 4),
                'temperature_measured': round(temperature, 4),
                'current_charge':       round(np.random.uniform(1.4, 1.6), 4),
                'voltage_charge':       round(np.random.uniform(4.18, 4.22), 4),
                'time':                 round(3600 * capacity / abs(current), 2),
                'capacity':             round(capacity, 6),
            })
    return pd.DataFrame(records)

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    mat_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.mat')]
    if mat_files:
        print(f"[INFO] Found .mat files: {mat_files}")
        df = load_nasa_dataset()
    else:
        print("[WARN] No .mat files found. Using synthetic data.")
        df = generate_synthetic_battery_data()
    df.to_csv(SAVE_PATH, index=False)
    print(f"\n[INFO] Saved -> {SAVE_PATH}")
    print(f"Shape: {df.shape}")
    print(f"\nSample:\n{df.head(5).to_string()}")
    print(f"Capacity range: {df['capacity'].min():.4f} - {df['capacity'].max():.4f} Ah")
    print(f"Cycles per battery:\n{df.groupby('battery_id')['cycle'].max()}")

if __name__ == '__main__':
    main()
