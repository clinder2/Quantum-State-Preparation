import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def visualize_set_cover_solution(universe, subsets, solution_indices):
    """
    Visualizes the Set Cover problem as an incidence matrix.
    Rows: Elements
    Cols: Subsets
    Cells: Filled if element is in subset.
    Highlight: Columns chosen in solution.
    """
    num_elements = len(universe)
    num_subsets = len(subsets)
    
    # Create Incidence Matrix
    # 0 = Not in subset
    # 1 = In subset (not chosen)
    # 2 = In subset (chosen)
    matrix = np.zeros((num_elements, num_subsets))
    
    for j, subset in enumerate(subsets):
        is_chosen = j in solution_indices
        for element in subset:
            if element in universe:
                i = universe.index(element)
                matrix[i, j] = 2 if is_chosen else 1
                
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Custom colormap: 0=White, 1=LightGray, 2=Green
    cmap = ListedColormap(['white', '#DDDDDD', '#4ECDC4'])
    
    # Show matrix
    # Note: imshow plots [row, col]. 
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect='auto')
    
    # Grid lines
    ax.set_xticks(np.arange(num_subsets) - 0.5, minor=True)
    ax.set_yticks(np.arange(num_elements) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Labels
    ax.set_xticks(np.arange(num_subsets))
    ax.set_xticklabels([f"S{j}\n{list(subsets[j])}" for j in range(num_subsets)])
    ax.set_yticks(np.arange(num_elements))
    ax.set_yticklabels(universe)
    
    # Highlight chosen columns (subsets) with a border or label color
    for j in range(num_subsets):
        if j in solution_indices:
            ax.get_xticklabels()[j].set_color("#2A9D8F")
            ax.get_xticklabels()[j].set_fontweight("bold")
            # Add a checkmark above
            ax.text(j, -0.7, "✓", ha='center', va='center', color="#2A9D8F", fontsize=20, fontweight='bold')
            
    # Add text to cells
    for i in range(num_elements):
        for j in range(num_subsets):
            val = matrix[i, j]
            if val > 0:
                ax.text(j, i, "•", ha='center', va='center', color='black')

    plt.title(f"Exact Cover Solution using Subsets {solution_indices}", fontsize=14, pad=20)
    plt.xlabel("Subsets", fontsize=12)
    plt.ylabel("Universe Elements", fontsize=12)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#DDDDDD', edgecolor='black', label='Contains Element'),
        Patch(facecolor='#4ECDC4', edgecolor='black', label='Chosen in Solution')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    plt.tight_layout()
    output_file = "set_cover_visualization.png"
    plt.savefig(output_file)
    print(f"Saved visualization to {output_file}")
    plt.close()

if __name__ == "__main__":
    # Example from VerifyHamiltonian
    universe = ['A', 'B', 'C']
    subsets = [{'A', 'B'}, {'C'}, {'A'}, {'B', 'C'}]
    
    # Solution 1: {0, 1} -> {A,B} + {C}
    solution_indices = [0, 1]
    
    visualize_set_cover_solution(universe, subsets, solution_indices)
