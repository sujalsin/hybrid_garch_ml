import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Arrow, FancyArrowPatch
import seaborn as sns

def create_architecture_diagram():
    # Set style
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors
    colors = {
        'input': '#2ecc71',
        'garch': '#3498db',
        'ml': '#e74c3c',
        'hybrid': '#9b59b6',
        'output': '#f1c40f'
    }
    
    # Create boxes
    boxes = {
        'input': Rectangle((0.1, 0.6), 0.15, 0.2, facecolor=colors['input'], alpha=0.3),
        'garch': Rectangle((0.35, 0.7), 0.2, 0.2, facecolor=colors['garch'], alpha=0.3),
        'ml': Rectangle((0.35, 0.3), 0.2, 0.2, facecolor=colors['ml'], alpha=0.3),
        'hybrid': Rectangle((0.65, 0.5), 0.15, 0.2, facecolor=colors['hybrid'], alpha=0.3),
        'output': Rectangle((0.9, 0.5), 0.15, 0.2, facecolor=colors['output'], alpha=0.3)
    }
    
    # Add boxes to plot
    for box in boxes.values():
        ax.add_patch(box)
    
    # Add arrows
    arrows = [
        ((0.25, 0.7), (0.35, 0.8)),
        ((0.25, 0.7), (0.35, 0.4)),
        ((0.55, 0.8), (0.65, 0.6)),
        ((0.55, 0.4), (0.65, 0.6)),
        ((0.8, 0.6), (0.9, 0.6))
    ]
    
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, 
                                   arrowstyle='->',
                                   mutation_scale=20,
                                   fc='gray'))
    
    # Add labels
    plt.text(0.125, 0.7, 'Historical\nData', ha='center', va='center')
    plt.text(0.45, 0.8, 'GARCH\nModel', ha='center', va='center')
    plt.text(0.45, 0.4, 'LSTM\nModel', ha='center', va='center')
    plt.text(0.725, 0.6, 'Hybrid\nPredictor', ha='center', va='center')
    plt.text(0.975, 0.6, 'Final\nPrediction', ha='center', va='center')
    
    # Configure plot
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0.2, 1)
    ax.axis('off')
    plt.title('Hybrid Model Architecture', pad=20, size=14)
    
    # Save plot
    plt.savefig('docs/images/hybrid_model_architecture.png', 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

def create_performance_comparison():
    # Set style
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sample data
    models = ['GARCH', 'LSTM', 'Hybrid']
    metrics = {
        'MSE': [0.0018, 0.0016, 0.0014],
        'MAE': [0.0385, 0.0356, 0.0324]
    }
    
    # Create grouped bar plot
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, metrics['MSE'], width, label='MSE', color='#3498db', alpha=0.7)
    ax.bar(x + width/2, metrics['MAE'], width, label='MAE', color='#e74c3c', alpha=0.7)
    
    # Customize plot
    ax.set_ylabel('Error Value')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    # Add value labels
    for i, v in enumerate(metrics['MSE']):
        ax.text(i - width/2, v, f'{v:.4f}', ha='center', va='bottom')
    for i, v in enumerate(metrics['MAE']):
        ax.text(i + width/2, v, f'{v:.4f}', ha='center', va='bottom')
    
    # Save plot
    plt.savefig('docs/images/performance_comparison.png', 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

if __name__ == '__main__':
    create_architecture_diagram()
    create_performance_comparison()
