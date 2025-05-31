import numpy as np
import pandas as pd
from scipy import stats
import itertools
from pysat.formula import CNF, IDPool
from pysat.solvers import Glucose3
import math

class SATCausalDiscovery:
    """
    A SAT-based causal discovery algorithm implementation for discrete data.
    
    This class implements causal discovery using Boolean satisfiability (SAT) solving.
    It encodes the constraints of causal discovery as a SAT problem and uses
    a SAT solver to find valid causal structures. Specifically adapted for
    discrete/categorical data.
    
    The adjacency matrix uses:
    - 1 to represent an arrowhead (→)
    - -1 to represent a tail (−)
    - 0 for no connection
    """
    
    def __init__(self, significance_level=0.05, max_cond_size=3):
        """
        Initialize the SAT-based causal discovery algorithm.
        
        Parameters:
        -----------
        significance_level : float
            The significance level for independence tests.
        max_cond_size : int
            Maximum size of conditioning sets to test.
        """
        self.significance_level = significance_level
        self.max_cond_size = max_cond_size
        self.var_pool = None
        self.cnf = None
        self.column_names = None
        
    def _conditional_independence_test(self, data, x, y, z=None):
        """
        Perform conditional independence test between discrete variables using G-test.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The dataset with variables as columns.
        x : str
            Name of first variable.
        y : str
            Name of second variable.
        z : list or None
            Names of conditioning variables.
            
        Returns:
        --------
        bool
            True if variables are conditionally independent, False otherwise.
        float
            p-value of the test.
        """
        # For marginal independence (no conditioning)
        if z is None or len(z) == 0:
            # Create contingency table
            contingency = pd.crosstab(data[x], data[y])
            # Chi-square test of independence
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            return p_value > self.significance_level, p_value
        else:
            # For conditional independence with discrete variables
            # We'll stratify by all combinations of conditioning variables
            # and perform independence tests within each stratum
            
            # Get all unique combinations of values for conditioning vars
            z_df = data[z]
            z_combinations = z_df.drop_duplicates().to_dict('records')
            
            if not z_combinations:
                return False, 0.0
                
            # Initialize weights for combining p-values
            weights = []
            p_values = []
            
            # For each combination of conditioning values
            for z_comb in z_combinations:
                # Create mask for this combination
                mask = pd.Series(True, index=data.index)
                for var, val in z_comb.items():
                    mask = mask & (data[var] == val)
                
                # Skip if too few samples
                if mask.sum() < 5:
                    continue
                    
                # Create contingency table for this stratum
                contingency = pd.crosstab(data.loc[mask, x], data.loc[mask, y])
                
                # Skip if the contingency table has less than 2 rows or columns
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue
                
                # Chi-square test within this stratum
                try:
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                    weights.append(mask.sum())
                    p_values.append(p_value)
                except:
                    # Handle numerical issues
                    continue
            
            # If we couldn't perform any valid tests
            if not p_values:
                return False, 0.0
                
            # Combine p-values (weighted average)
            combined_p = sum(w * p for w, p in zip(weights, p_values)) / sum(weights)
            
            return combined_p > self.significance_level, combined_p
    
    def _create_edge_variables(self, n_vars):
        """
        Create SAT variables for edges in the graph.
        
        Parameters:
        -----------
        n_vars : int
            Number of variables/nodes.
            
        Returns:
        --------
        dict
            Mapping from edge tuples to SAT variable indices.
        """
        self.var_pool = IDPool()
        edge_vars = {}
        
        # Create variables for each possible edge i→j
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:  # No self-loops
                    edge_vars[(i, j)] = self.var_pool.id(f"edge_{i}_{j}")
        
        return edge_vars
    
    def _add_acyclicity_constraints(self, n_vars, edge_vars):
        """
        Add constraints to ensure the graph is acyclic.
        
        Parameters:
        -----------
        n_vars : int
            Number of variables/nodes.
        edge_vars : dict
            Mapping from edge tuples to SAT variable indices.
        """
        # For all paths of length 2: if i→j and j→k, then not i→k
        for i, j, k in itertools.permutations(range(n_vars), 3):
            # If a path i→j→k exists, then i→k must be false
            self.cnf.append([-edge_vars[(i, j)], -edge_vars[(j, k)], -edge_vars[(k, i)]])
        
        # No cycles of length 2 (no bidirectional edges)
        for i, j in itertools.permutations(range(n_vars), 2):
            self.cnf.append([-edge_vars[(i, j)], -edge_vars[(j, i)]])
    
    def _add_independence_constraints(self, data, n_vars, edge_vars):
        """
        Add constraints based on conditional independence tests.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The dataset with variables as columns.
        n_vars : int
            Number of variables/nodes.
        edge_vars : dict
            Mapping from edge tuples to SAT variable indices.
        """
        independence_constraints = []
        edges_removed = 0
        edges_forced = 0
        
        # Track which edges have been found independent
        independent_edges = set()
        
        # Limit conditioning set size for efficiency
        max_cond_size = min(self.max_cond_size, n_vars - 2)  
        valid_cond = []
        for cond_size in range(1, max_cond_size + 1):
            for z_indices in itertools.combinations(list(range(n_vars)), cond_size):
                z_names = [self.column_names[k] for k in z_indices]
                other_v = [k for k in range(n_vars) if k not in z_indices]
                count = 0
                total_num = math.comb(len(other_v), 2)
                for xy_indices in itertools.combinations(other_v, 2):
                    i, j = xy_indices
                    x = self.column_names[i]
                    y = self.column_names[j]
                    is_independent, p_value = self._conditional_independence_test(data, x, y, z_names)
                    if not is_independent:
                        count += 1
                if count >= total_num // 2:
                    valid_cond.append(z_names)

        print(f"Found {len(valid_cond)} valid conditioning sets")

        # First pass: test marginal independence and mark independent edges
        for i, j in itertools.combinations(range(n_vars), 2):
            x = self.column_names[i]
            y = self.column_names[j]
                
            # Test marginal independence
            is_independent, p_value = self._conditional_independence_test(data, x, y)
            
            if is_independent:
                # If X and Y are independent, there should be no direct edge
                independence_constraints.append([-edge_vars[(i, j)]])
                independence_constraints.append([-edge_vars[(j, i)]])
                edges_removed += 2
                independent_edges.add((i, j))
                independent_edges.add((j, i))
                print(f"Variables {x} and {y} are marginally independent (p={p_value:.3f})")
            
            # Test conditional independence with various conditioning sets
            for z_names in valid_cond:
                is_independent, p_value = self._conditional_independence_test(data, x, y, z_names)
                
                if is_independent:
                    # For simplicity, if X and Y are conditionally independent given Z,
                    # we enforce no direct edge between them
                    independence_constraints.append([-edge_vars[(i, j)]])
                    independence_constraints.append([-edge_vars[(j, i)]])
                    edges_removed += 2
                    independent_edges.add((i, j))
                    independent_edges.add((j, i))
                    conditioning_vars = ', '.join(z_names)
                    print(f"Variables {x} and {y} are conditionally independent given {conditioning_vars} (p={p_value:.3f})")

        # Second pass: add positive constraints for edges that were never found independent
        for i, j in itertools.combinations(range(n_vars), 2):
            if (i, j) not in independent_edges and (j, i) not in independent_edges:
                # If neither direction was found independent, force at least one edge to exist
                independence_constraints.append([edge_vars[(i, j)], edge_vars[(j, i)]])
                edges_forced += 1
                print(f"Variables {self.column_names[i]} and {self.column_names[j]} are dependent in all tests, forcing an edge")

        print(f"Total edges removed through independence tests: {edges_removed}")
        print(f"Total edges forced through dependence: {edges_forced}")
        for clause in independence_constraints:
            self.cnf.append(clause)
    
    def discover(self, data):
        """
        Discover causal structure from data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The dataset with discrete variables as columns.
            
        Returns:
        --------
        pandas.DataFrame
            Adjacency matrix of the discovered causal graph with variable names.
            Uses -1 for tails and 1 for arrowheads.
        """
        # Store column names for variable mapping
        self.column_names = list(data.columns)
        n_vars = len(self.column_names)
        
        # Verify data is discrete/integer
        for col in self.column_names:
            if not pd.api.types.is_integer_dtype(data[col]) and not data[col].dtype.name == 'category':
                print(f"Warning: Column '{col}' might not be discrete. Consider converting to integer type.")
        
        # Create SAT variables for edges
        edge_vars = self._create_edge_variables(n_vars)
        
        # Initialize CNF formula
        self.cnf = CNF()
        
        # Add acyclicity constraints
        self._add_acyclicity_constraints(n_vars, edge_vars)
        
        # Add constraints from independence tests
        self._add_independence_constraints(data, n_vars, edge_vars)
        
        # Solve the SAT problem
        solver = Glucose3()
        for clause in self.cnf:
            solver.add_clause(clause)
            
        if solver.solve():
            # Extract the model (solution)
            model = solver.get_model()
            #print("---Model:----")
            #print(model)
            
            # Initialize adjacency matrix with zeros
            adjacency_matrix = pd.DataFrame(0, 
                                           index=self.column_names, 
                                           columns=self.column_names)
            
            # Construct adjacency matrix with -1 for tails and 1 for arrowheads
            edges_found = 0
            for (i, j), var_id in edge_vars.items():
                if var_id in model:  # Positive literal in the model
                    edges_found += 1
                    from_var = self.column_names[i]
                    to_var = self.column_names[j]
                    # i → j means i has a tail (-1) and j has an arrowhead (1)
                    adjacency_matrix.loc[from_var, to_var] = 1  # Arrowhead at j
                    adjacency_matrix.loc[to_var, from_var] = -1  # Tail at i
            
            print(f"Found {edges_found} edges in the solution")
            return adjacency_matrix.to_numpy()
        else:
            print("No solution found. The constraints may be too restrictive.")
            adjacency_matrix = pd.DataFrame(0, 
                                           index=self.column_names, 
                                           columns=self.column_names)
            return adjacency_matrix.to_numpy()
    
    def plot_graph(self, adjacency_matrix):
        """
        Plot the discovered causal graph using networkx and matplotlib.
        
        Parameters:
        -----------
        adjacency_matrix : pandas.DataFrame
            The adjacency matrix with -1 for tails and 1 for arrowheads.
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            # Create directed graph
            G = nx.DiGraph()
            
            # Add nodes
            G.add_nodes_from(adjacency_matrix.index)
            
            # Add edges based on arrowheads (value 1)
            for i, row in enumerate(adjacency_matrix.index):
                for j, col in enumerate(adjacency_matrix.columns):
                    if adjacency_matrix.loc[row, col] == 1:
                        # This means col → row (col causes row)
                        G.add_edge(col, row)
            
            # Plot the graph
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                   node_size=1500, arrowsize=20, font_size=12,
                   font_weight='bold', arrows=True)
            plt.title("Discovered Causal Graph")
            plt.show()
            
        except ImportError:
            print("Plotting requires networkx and matplotlib. Install them with:")
            print("pip install networkx matplotlib")



# def main():
#     # Use the complex model with 5 variables
#     print("Generating synthetic data from complex model: X→Y→Z, X→W→V, V→Y")
#     data = generate_complex_synthetic_data(n_samples=2000)
    
#     print("Sample of generated data:")
#     print(data.head())
    
#     # Run causal discovery
#     discovery = SATCausalDiscovery(significance_level=0.01, max_cond_size=2)
#     adjacency_matrix = discovery.discover(data)
    
#     print("\nDiscovered causal graph adjacency matrix (-1 for tails, 1 for arrowheads):")
#     print(adjacency_matrix)
    
#     # Try to plot the graph if libraries are available
#     discovery.plot_graph(adjacency_matrix)

# if __name__ == "__main__":
#     main()