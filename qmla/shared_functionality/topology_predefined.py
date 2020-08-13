r"""
A number of GridTopology instances predefined for convenience in growth rules. 

"""
from qmla.shared_functionality.topology import GridTopology


# 1D: chains
_2_site_chain = GridTopology(
    dimension=1, num_sites = 2
)
_3_site_chain = GridTopology(
    dimension=1, num_sites = 3
)
_3_site_chain_fully_connected = GridTopology(
    dimension=1, num_sites = 3,
    all_sites_connected=True
)
_4_site_chain = GridTopology(
    dimension=1, num_sites = 4
)
_5_site_chain = GridTopology(
    dimension=1, num_sites = 5
)
_6_site_chain = GridTopology(
    dimension=1, num_sites = 6
)


# 2D 

# Fully connected
_3_site_lattice_fully_connected = GridTopology(
    dimension=2, num_sites=3, all_sites_connected=True
) 
_4_site_lattice_fully_connected = GridTopology(
    dimension=2, num_sites=4, all_sites_connected=True
) 
_5_site_lattice_fully_connected = GridTopology(
    dimension=2, num_sites=5, all_sites_connected=True
)

# Squares and Grids
_4_site_square = GridTopology(dimension=2, num_sites=4)
_6_site_grid = GridTopology(
    dimension=2, num_sites = 6
)
