# Port-Microgrid

Port-Microgrid is a reinforcement learning project designed to optimize energy management in microgrid systems, with a focus on port-based applications. The project simulates an environment where battery storage, photovoltaic (PV) generation, and hydrogen production/storage interact with grid power to achieve efficient energy management.

## Repository Structure

```
├── dataset.csv                      # CSV file containing simulation data (e.g., Load, PV, tariffs, etc.)
├── train_and_test_dqn_stable.py     # Script to train and test a DQN agent using Stable-Baselines3
├── train_and_test_ppo_stable.py     # Script to train and test a PPO agent using Stable-Baselines3
├── validation_stable.py             # Validation script for the DQN model
├── validation_stable_PPO.py         # Validation script for the PPO model
└── models
    ├── environment.py               # Implementation of the custom energy management environment (EnergyEnv)
    └── __pycache__                  # Compiled Python files (ignore)
```

## Overview

This project leverages reinforcement learning to learn optimal energy management policies in a simulated microgrid environment. The custom environment (implemented in `models/environment.py`) incorporates various energy flows:  
- **Battery Management:** Includes charging/discharging with constraints on state-of-charge (SoC).  
- **PV Utilization:** Directly meets load, charges the battery, or produces hydrogen.  
- **Hydrogen System:** Supports hydrogen production, storage, and utilization via a fuel cell.  
- **Grid Interactions:** Manages grid imports and exports with associated costs and emissions.  

Reinforcement learning agents (using DQN and PPO) are trained on this environment to minimize a composite cost metric that includes both energy bills and emissions.

## Features

- **Custom Energy Environment:** Models a microgrid with dynamic load, PV output, battery storage, and hydrogen production.  
- **Reinforcement Learning Agents:** Implements DQN and PPO agents using the Stable-Baselines3 framework.  
- **Training and Testing Pipelines:** Separate scripts for training agents and running test episodes.  
- **Validation Tools:** Scripts to evaluate agent performance and generate visualizations of energy flows, financial metrics, and system states.  

## Installation

1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/Tayyab-Tahir1/Port-Microgrid
   cd Port-Microgrid
   ```

2. **Install Dependencies:**  
   Ensure you have Python 3.8+ installed, then run:  
   ```bash
   pip install gymnasium stable-baselines3 pandas numpy matplotlib tqdm
   ```  
   Alternatively, if a `requirements.txt` file is provided, run:  
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training and Testing  
**DQN Agent:**  
To train and test the DQN agent:  
```bash
python train_and_test_dqn_stable.py --mode train --timesteps 1752000 --test_episodes 10
```  
Use `--mode both` to perform both training and testing sequentially. Adjust `--timesteps` and `--test_episodes` as needed.  

**PPO Agent:**  
To train and test the PPO agent:  
```bash
python train_and_test_ppo_stable.py --mode train --timesteps 1752000 --test_episodes 10
```  

### Validation  
After training, validate the performance of the models:  

**DQN Validation:**  
```bash
python validation_stable.py
```  

**PPO Validation:**  
```bash
python validation_stable_PPO.py
```  

These scripts process the simulation data, run the trained models over the dataset, and generate plots (saved as PNG files) showing energy flows, financial metrics, and system states.

## Customization

- **Dataset:**  
  The dataset (`dataset.csv`) should contain columns such as `Load`, `PV`, `Tou Tariff`, `FiT`, `H2 Tariff`, `Day`, and `Hour`.  
  A preprocessing step in the training scripts renames some columns (e.g., converting "Tou Tariff" to "Tou_Tariff") to match the expected format.  

- **Environment Parameters:**  
  The `EnergyEnv` class (in `models/environment.py`) sets parameters like battery capacity, hydrogen storage capacity, and efficiencies. Modify these parameters to fit different simulation scenarios.  

- **RL Hyperparameters:**  
  Training scripts allow you to adjust learning rates, batch sizes, exploration settings, and more. Tweak these settings to fine-tune agent performance.  

## Contributing

Contributions to improve the environment, training pipelines, or validation processes are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the **Apache 2.0**.

## Acknowledgments

Port-Microgrid utilizes the **Stable-Baselines3** framework and **Gymnasium** library, along with various open-source Python libraries, to build a comprehensive simulation and training platform for microgrid energy management.
