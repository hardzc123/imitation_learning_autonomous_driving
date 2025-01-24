# Imitation Learning for Autonomous Driving in the CarRacing Simulation

# ![Manual Control Example](images/bc.gif)


This project demonstrates a robust approach to training autonomous driving agents using Imitation Learning (IL) in Gymnasium’s CarRacing-v3 simulation. The methodology employs two distinct sources of expert data—manual human operations and Proximal Policy Optimization (PPO)—to train separate Behavior Cloning (BC) models. These models are later integrated and iteratively refined using Dataset Aggregation (DAgger), a technique designed to improve the adaptability and robustness of the agents.



## Simulation Environment

The CarRacing-v3 simulation, part of the Gymnasium toolkit, is a versatile platform for developing, testing, and refining autonomous driving algorithms. It offers:
- **Complex Track Layouts**: Tracks with varying configurations, including sharp turns and straightaways, to challenge driving strategies.
- **Dynamic Visual Rendering**: High-resolution images from the driver’s perspective for vision-based decision-making.
- **Customizable Conditions**: Adjustable parameters for weather, track friction, and obstacles to create diverse testing scenarios.

The simulation plays a central role across multiple stages of the training pipeline:
- **Data Collection**:
  - **Manual Control**: Human drivers interact with the simulation using a joystick or keyboard, viewing the racetrack through high-resolution images (input) and controlling the car via steering, throttle, and braking actions (output). This process generates a dataset of expert driving decisions under varied conditions.
  - **PPO Expert**: The PPO algorithm uses track images (input) to make driving decisions (output) while optimizing policies through feedback based on simulation rewards.
- **Behavior Cloning (BC) Validation**: After training, the BC models are tested within the simulation to evaluate their driving performance and robustness under different track configurations.
- **DAgger Data Expansion**: During the DAgger phase, the simulation is used to run BC models, identify action discrepancies, and collect additional training data through real-time corrections from manual or PPO experts.



## Implementation Details

### 1. Expert Data Collection
- **From Manual Operations**: A human expert drives the car to generate a dataset capturing diverse and optimal behaviors for navigating challenging track layouts.
- **From PPO Training**: The PPO-trained agent creates an alternative dataset by optimizing its driving strategies through reinforcement learning, focusing on maximizing cumulative rewards.

### 2. Dual Behavior Cloning (BC) Models
- **BC from Manual Data**: A neural network is trained using the human-generated dataset. The input is track images, while the labels correspond to human actions (steering, throttle, braking).
- **BC from PPO Data**: Another neural network is trained on the PPO-generated dataset, leveraging strategic and efficient driving patterns learned by the algorithm.

### 3. Integration and Refinement with DAgger
- The policy starts with predictions from the BC models and is refined iteratively using the DAgger method. Discrepancies between the agent’s actions and those of a composite expert (combining manual and PPO insights) are corrected in real-time, leading to an improved policy.

### 4. Hyperparameters and Training Specifics
- **PPO Hyperparameters**: Configurations include an entropy coefficient of 0.01, a discount factor of 0.99, and a learning rate of 0.0003, with updates occurring every 2048 timesteps.
- **BC Models’ Learning Rates**: Both operate with a learning rate of 0.001, utilizing the Adam optimizer for efficient convergence.
- **DAgger Iterations**: Adjusted dynamically to enhance responsiveness and further align the policy with expert strategies.

### 5. Expected Outcomes and Potential Enhancements
- The dual BC model approach effectively integrates human intuition with algorithmic precision, producing robust driving agents capable of adapting to diverse environments.
- Future advancements could involve more complex neural network architectures, expanded datasets with greater diversity, and extended training periods, further improving the performance and generalizability of the agent.


