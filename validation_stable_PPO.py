import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import PPO

# =========================
# System Parameters (updated)
# =========================
BATTERY_CAPACITY = 5000       # kWh
C_RATE = 1
SOC_MIN = 0.2 * BATTERY_CAPACITY
SOC_MAX = 0.8 * BATTERY_CAPACITY
EFFICIENCY = 0.95

# Hydrogen storage parameters
H2_STORAGE_CAPACITY = 1000.0  # Maximum hydrogen storage in kg
ENERGY_PER_KG_H2 = 32.0       # kWh per kg H2
FUEL_CELL_EFFICIENCY = 0.5    # Fuel cell conversion efficiency

# Emission factor (kWh emission per kWh imported)
EMISSION_FACTOR = 0.5

# Grid battery charging fraction
GRID_CHARGE_FRACTION = 0.5

# Global reference for maximum power (computed from dataset)
self_max_power = None

def compute_max_power(df):
    return max(df['Load'].max(), df['PV'].max(), BATTERY_CAPACITY * C_RATE, 2000)

# =========================
# Feasible Actions Function
# =========================
def get_feasible_actions_new(load, pv, tou_tariff, h2_tariff, soc, h2_storage):
    """
    Expanded action space (0–7):
      0: Do nothing
      1: Battery discharge to meet load
      2: Charge battery using excess PV
      3: Produce hydrogen using excess PV (store produced H2)
      4: Use stored hydrogen directly to meet load
      5: Charge battery using grid power
      6: Convert stored hydrogen to battery energy (H2-to-Battery)
      7: Purchase hydrogen (in kg) from the grid to meet load
    """
    feasible = [0]
    if soc > SOC_MIN + 1e-5:
        feasible.append(1)
    if soc < SOC_MAX - 1e-5:
        feasible.append(2)
        feasible.append(5)
    if pv > 0:
        feasible.append(3)
    if h2_storage > 0:
        feasible.append(4)
        feasible.append(6)
    if load > 0:
        feasible.append(7)
    return feasible

# =========================
# Process Action Function
# =========================
def process_action_new(action, load, pv, tou_tariff, fit, h2_tariff, soc, h2_storage):
    """
    Processes the chosen action and computes flows, cost, emissions, etc.
    Assumes:
      - H2_Tariff is expressed as cost per kg of hydrogen.
      - ENERGY_PER_KG_H2 = 32 kWh/kg, FUEL_CELL_EFFICIENCY = 0.5.
      
    Returns:
      updated soc, updated h2_storage, allocations, grid_cost, pv_revenue, bill, emissions, reward.
      
    The allocations dictionary includes:
      'pv_to_load', 'pv_to_battery', 'pv_to_grid',
      'battery_to_load', 'grid_to_load', 'grid_to_battery',
      'h2_to_load', 'hydrogen_produced', 'h2_to_battery', 'h2_to_load_purchased',
      'H2_Purchased_kg'
    """
    allocations = {
        'pv_to_load': 0.0,
        'pv_to_battery': 0.0,
        'pv_to_grid': 0.0,
        'battery_to_load': 0.0,
        'grid_to_load': 0.0,
        'grid_to_battery': 0.0,
        'h2_to_load': 0.0,
        'hydrogen_produced': 0.0,
        'h2_to_battery': 0.0,
        'h2_to_load_purchased': 0.0,
        'H2_Purchased_kg': 0.0
    }
    H2_purchase_cost = 0
    
    # Use PV to directly supply load.
    allocations['pv_to_load'] = min(pv, load)
    load_remaining = load - allocations['pv_to_load']
    pv_remaining = pv - allocations['pv_to_load']
    
    fc_eff = FUEL_CELL_EFFICIENCY  # 0.5
    
    if action == 0:
        pass
    elif action == 1:
        available_energy = (soc - SOC_MIN) * EFFICIENCY
        allocations['battery_to_load'] = min(load_remaining, available_energy)
        soc -= allocations['battery_to_load'] / EFFICIENCY
        load_remaining -= allocations['battery_to_load']
    elif action == 2:
        allocations['pv_to_battery'] = pv_remaining
        soc += allocations['pv_to_battery'] * EFFICIENCY
        pv_remaining = 0
    elif action == 3:
        energy_used = pv_remaining
        allocations['hydrogen_produced'] = energy_used / ENERGY_PER_KG_H2
        h2_storage = min(h2_storage + allocations['hydrogen_produced'], H2_STORAGE_CAPACITY)
        pv_remaining = 0
    elif action == 4:
        available_h2_energy = h2_storage * ENERGY_PER_KG_H2
        allocations['h2_to_load'] = min(load_remaining, available_h2_energy)
        hydrogen_used = allocations['h2_to_load'] / ENERGY_PER_KG_H2
        h2_storage -= hydrogen_used
        load_remaining -= allocations['h2_to_load']
    elif action == 5:
        available_capacity = SOC_MAX - soc
        energy_to_charge = available_capacity * GRID_CHARGE_FRACTION
        allocations['grid_to_battery'] = energy_to_charge / EFFICIENCY
        soc += energy_to_charge
    elif action == 6:
        available_h2_energy = h2_storage * ENERGY_PER_KG_H2
        battery_capacity_remaining = SOC_MAX - soc
        energy_converted = min(available_h2_energy, battery_capacity_remaining)
        battery_energy_gained = energy_converted * fc_eff
        allocations['h2_to_battery'] = battery_energy_gained
        soc += battery_energy_gained
        hydrogen_used = energy_converted / ENERGY_PER_KG_H2
        h2_storage -= hydrogen_used
    elif action == 7:
        # Purchase hydrogen from grid to meet load.
        hydrogen_required_kg = load_remaining / (ENERGY_PER_KG_H2 * fc_eff)
        allocations['H2_Purchased_kg'] = hydrogen_required_kg
        allocations['h2_to_load_purchased'] = load_remaining
        H2_purchase_cost = hydrogen_required_kg * h2_tariff
        load_remaining = 0

    allocations['grid_to_load'] = load_remaining
    allocations['pv_to_grid'] = pv_remaining
    
    soc = max(SOC_MIN, min(soc, SOC_MAX))
    
    grid_cost = (allocations['grid_to_load'] + allocations['grid_to_battery']) * tou_tariff
    pv_revenue = allocations['pv_to_grid'] * fit
    bill = grid_cost - pv_revenue + H2_purchase_cost
    
    emissions = (allocations['grid_to_load'] + allocations['grid_to_battery']) * EMISSION_FACTOR
    
    # Use cost_weight and emission_weight from the environment.
    composite = (1.0 * bill) + (0.2 * emissions)
    max_possible_bill = self_max_power * max(tou_tariff, h2_tariff)
    if max_possible_bill == 0:
        max_possible_bill = 1
    reward = - composite / max_possible_bill
    
    return soc, h2_storage, allocations, grid_cost, pv_revenue, bill, emissions, reward

# =========================
# Main Validation Script for PPO
# =========================
def main():
    model_path = "ppo_energy_model_finetuned.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PPO model not found at {model_path}")
    model = PPO.load(model_path)
    print(f"Loaded PPO model from {model_path}")

    dataset_path = "dataset.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully.")
    
    global self_max_power
    self_max_power = compute_max_power(df)
    
    allocation_columns = [
        'pv_to_load', 'pv_to_battery', 'pv_to_grid',
        'battery_to_load', 'grid_to_load', 'grid_to_battery',
        'h2_to_load', 'hydrogen_produced', 'h2_to_battery',
        'h2_to_load_purchased', 'H2_Purchased_kg',
        'Purchase', 'Sell', 'Bill', 'Emissions', 'SoC', 'H2_Storage', 'Chosen_Action'
    ]
    for col in allocation_columns:
        df[col] = np.nan

    soc = BATTERY_CAPACITY * 0.5
    h2_storage = 0.0

    print("\nRunning validation with PPO model (expanded action space)...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        load = row['Load']
        pv = row['PV']
        tou_tariff = row['Tou_Tariff']
        fit = row['FiT']
        h2_tariff = row['H2_Tariff']
        day = row['Day']
        hour = row['Hour']

        state = np.array([
            load / self_max_power,
            pv / self_max_power,
            tou_tariff / df['Tou_Tariff'].max(),
            fit / df['FiT'].max(),
            h2_tariff / df['H2_Tariff'].max(),
            soc / BATTERY_CAPACITY,
            h2_storage / H2_STORAGE_CAPACITY,
            day / 6.0,
            hour / 23.0
        ], dtype=np.float32)

        # Use the PPO model to predict an action.
        raw_action, _ = model.predict(state, deterministic=True)
        feasible_actions = get_feasible_actions_new(load, pv, tou_tariff, h2_tariff, soc, h2_storage)
        if raw_action not in feasible_actions:
            raw_action = np.random.choice(feasible_actions)
        
        soc, h2_storage, allocations, purchase, sell, bill, emissions, reward = process_action_new(
            raw_action, load, pv, tou_tariff, fit, h2_tariff, soc, h2_storage
        )

        for key, value in allocations.items():
            df.at[index, key] = value
        df.at[index, 'Purchase'] = purchase
        df.at[index, 'Sell'] = sell
        df.at[index, 'Bill'] = bill
        df.at[index, 'Emissions'] = emissions
        df.at[index, 'SoC'] = (soc / BATTERY_CAPACITY) * 100
        df.at[index, 'H2_Storage'] = (h2_storage / H2_STORAGE_CAPACITY) * 100
        df.at[index, 'Chosen_Action'] = raw_action

    output_csv = "results_ppo_stable_updated.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(df["Purchase"].rolling(24).mean(), label="Purchase")
    plt.plot(df["Sell"].rolling(24).mean(), label="Sell")
    plt.plot(df["Bill"].rolling(24).mean(), label="Net Bill")
    plt.plot(df["Emissions"].rolling(24).mean(), label="Emissions")
    plt.title("24-hour Rolling Average of Financial Metrics (Expanded Actions)")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(df["SoC"], label="Battery SoC")
    plt.plot(df["H2_Storage"], label="Hydrogen Storage (%)")
    plt.axhline(y=20, color="r", linestyle="--", label="Min SoC")
    plt.axhline(y=80, color="r", linestyle="--", label="Max SoC")
    plt.title("Battery SoC & Hydrogen Storage")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("energy_management_results_ppo_stable_updated.png")
    plt.close()
    print("Plots saved to 'energy_management_results_ppo_stable_updated.png'.")

if __name__ == "__main__":
    main()
