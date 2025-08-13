Neural Networks for Differential Equations: PINNs & Classical Approaches
This repository contains two complementary approaches for solving differential equations using neural networks: Physics-Informed Neural Networks (PINNs) with advanced features and Classical Neural Network ODE Solvers.

🔬 Overview
This project demonstrates cutting-edge techniques for solving ordinary and partial differential equations using deep learning, featuring both foundational methods and state-of-the-art implementations with hyperparameter optimization and attention mechanisms.

📁 Repository Structure
text
├── pinns_Advanced.ipynb          # Advanced PINN framework with modern ML techniques
├── NN_ODE_RC_Notebook-1.ipynb    # Classical neural network ODE solver
├── README.md                     # This file
└── requirements.txt              # Dependencies
🚀 Quick Start
Prerequisites
bash
pip install torch torchvision numpy matplotlib scipy optuna tqdm
Running the Notebooks
Advanced PINNs: Open pinns_Advanced.ipynb for sophisticated PINN implementations

Classical Methods: Open NN_ODE_RC_Notebook-1.ipynb for foundational neural ODE solving

📊 Features Comparison
Feature	Advanced PINNs	Classical NN ODE
Hyperparameter Optimization	✅ Optuna integration	❌ Manual tuning
Attention Mechanisms	✅ Feature-wise attention	❌ Standard architecture
Uncertainty Quantification	✅ Ensemble methods	❌ Single model
Boundary Conditions	✅ Flexible (Dirichlet/Neumann)	✅ Trial function approach
User-Defined ODEs	✅ Lambda function support	✅ Hardcoded examples
Computational Efficiency	⚡ Optimized with tuning	⚡ Fast and simple
Learning Curve	🔴 Advanced	🟢 Beginner-friendly
🧠 Advanced PINN Framework
Key Innovations
Hyperparameter Autotuning with Optuna

Automated optimization of learning rates, network architecture, and loss weights

50+ trials for finding optimal configurations

Best achieved loss: ~0.0019

Attention Mechanism Integration

Feature-wise attention gates for enhanced learning

Selective feature weighting during training

Improved convergence and solution quality

Ensemble Uncertainty Quantification

Multiple model training with different random seeds

Statistical analysis of prediction variance

Confidence bounds for solution reliability

Flexible Architecture

python
# Example configuration
main(perform_tuning=True, use_attention=True)
Core Components
PINN Architecture
python
class PINN(nn.Module):
    def __init__(self, layers, activation=nn.Tanh, use_attention=False):
        # Customizable depth, width, and attention integration
        # Support for Tanh, ReLU, and other activation functions
Physics Loss Implementation
python
def physics_loss(model, x, ode_func):
    # Automatic differentiation for derivative computation
    # Residual calculation for differential equation enforcement
Hyperparameter Optimization
python
def objective(trial):
    # Optuna-based parameter space exploration
    # Multi-objective optimization for physics and boundary losses
🔧 Classical Neural Network ODE Solver
Lagaris Method Implementation
Based on the foundational approach by Lagaris et al., this implementation provides:

Trial Function Construction

Automatic boundary condition satisfaction

Custom function design for specific problems

Neural Network Architecture

python
class Fitter(nn.Module):
    # Simple feedforward network
    # Configurable hidden layers and nodes
Applications

General ODE solving

RC circuit analysis

Educational examples with analytical comparisons

Example Problems
General ODE: dy/dx = f(x, y) with initial conditions

RC Circuit: First-order differential equation for circuit analysis

📈 Performance Results
Advanced PINNs
Optimized Loss: 0.00227 (best configuration)

Training Epochs: 3000 (full training)

Hyperparameter Trials: 50 with Optuna

Ensemble Size: 5 models for uncertainty

Classical NN ODE
Final Training Cost: 0.053

MSE vs Analytical: 0.005 (RC circuit)

Training Time: 2.77 seconds (100 epochs)

Architecture: 10 hidden nodes

🎯 Use Cases & Applications
When to Use Advanced PINNs
✅ Complex geometries or boundary conditions

✅ Limited experimental data

✅ Need for uncertainty quantification

✅ Research and development

✅ Multi-physics problems

When to Use Classical NN ODE
✅ Simple 1D problems

✅ Educational purposes

✅ Fast prototyping

✅ Well-defined boundary conditions

✅ Computational efficiency requirements

🛠️ Technical Implementation Details
Automatic Differentiation
Both approaches leverage PyTorch's automatic differentiation for computing derivatives required in differential equations.

Loss Functions
Physics Loss: Enforces differential equation residuals

Boundary Loss: Satisfies initial/boundary conditions

Combined Loss: Weighted sum with tunable parameters

Optimization Strategies
Adam Optimizer: Adaptive learning rates

Learning Rate Scheduling: Dynamic adjustment during training

Ensemble Training: Multiple models for robustness

📚 Mathematical Foundation
PINNs Formulation
text
Total Loss = λ₁ × Physics_Loss + λ₂ × BC_Loss
Physics_Loss = ||∂u/∂t + N[u]||²
BC_Loss = ||u(boundary) - g||²
Classical Approach
text
Trial Function: ψ(x) = A + x × N(x)
Loss = ||dψ/dx - f(x,ψ)||²
🔄 Workflow
Problem Definition: Define ODE and boundary conditions

Model Configuration: Choose architecture and parameters

Training: Optimize physics and boundary losses

Evaluation: Compare with analytical solutions

Uncertainty Analysis: Ensemble predictions (PINNs only)

🤝 Contributing
Contributions are welcome! Please feel free to submit pull requests for:

New differential equation examples

Additional attention mechanisms

Improved optimization strategies

Documentation enhancements

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Physics-Informed Neural Networks methodology by Raissi et al.

Lagaris method for neural network ODE solving

Optuna framework for hyperparameter optimization

PyTorch team for automatic differentiation capabilities

📖 References
Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks

Lagaris, I. E., Likas, A., & Fotiadis, D. I. (1998). Artificial neural networks for solving ordinary and partial differential equations

Optuna: A Next-generation Hyperparameter Optimization Framework

Get Started: Clone this repository and explore both pinns_Advanced.ipynb for cutting-edge techniques and NN_ODE_RC_Notebook-1.ipynb for foundational understanding!
