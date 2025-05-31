import numpy as onp
import jax.numpy as np
import json



def non_uniform_mesh(node_turning, num_long_elem, num_short_elem, dim=1):
    """
    Mesh generator for non-uniform segments
    --- Inputs ---
    node_turning: 1D numpy array of known nodes
    num_long_elem: Number of elements in each long segment
    num_short_elem: Number of elements in each short segment
    dim: Problem dimension (default is 1 for 1D)
    
    --- Outputs ---
    XY: nodal coordinates (nnode, dim)
    Elem_nodes: elemental nodes (nelem, nodes_per_elem)
    nelem: number of elements
    nnode: number of nodes
    dof_global: global degrees of freedom
    """
    
    node_turning = onp.sort(node_turning)  # Ensure the nodes are in increasing order
    num_segments = len(node_turning) - 1  # Number of segments
    nodes_per_elem = 2  # For 1D 2-node linear elements
    
    # Initialize lists to store the coordinates and connectivity
    XY_list = []
    Elem_nodes_list = []
    
    node_counter = 0
    
    for seg in range(num_segments):
        segment_start = node_turning[seg]
        segment_end = node_turning[seg + 1]
        segment_length = segment_end - segment_start
        
        # Determine the number of elements for this segment
        if seg % 2 == 0:  # Long segment
            nelem = num_long_elem
        else:  # Short segment
            nelem = num_short_elem
        
        # Calculate the spacing between nodes for this segment
        dx = segment_length / nelem
        
        for i in range(nelem + 1):
            x = segment_start + i * dx
            if i == 0 and seg > 0:
                continue  # Skip the first node if it's coincident with the previous segment end node
            
            XY_list.append([x])
            if i < nelem:
                Elem_nodes_list.append([node_counter, node_counter + 1])
            
            node_counter += 1
    
    # Convert lists to numpy arrays
    XY = onp.array(XY_list)
    Elem_nodes = onp.array(Elem_nodes_list, dtype=onp.int32)
    
    nnode = len(XY)
    nelem = len(Elem_nodes)
    dof_global = nnode * dim
    
    return XY, Elem_nodes, nelem, nnode, dof_global

def non_uniform_mesh_list(node_turning, num_long_elem, num_short_elem, dim=1):
    """
    Mesh generator for non-uniform segments
    --- Inputs ---
    node_turning: 1D numpy array of known nodes
    num_long_elem: Number of elements in each long segment
    num_short_elem: Number of elements in each short segment
    dim: Problem dimension (default is 1 for 1D)
    
    --- Outputs ---
    XY: nodal coordinates (nnode, dim)
    Elem_nodes: elemental nodes (nelem, nodes_per_elem)
    nelem: number of elements
    nnode: number of nodes
    dof_global: global degrees of freedom
    """
    
    node_turning = onp.sort(node_turning)  # Ensure the nodes are in increasing order
    num_segments = len(node_turning) - 1  # Number of segments
    nodes_per_elem = 2  # For 1D 2-node linear elements
    
    # Initialize lists to store the coordinates and connectivity
    XY = []; Elem_nodes = []
    for seg in range(num_segments):
        XY_list = []
        Elem_nodes_list = []
        node_counter = 0
        segment_start = node_turning[seg]
        segment_end = node_turning[seg + 1]
        segment_length = segment_end - segment_start
        
        # Determine the number of elements for this segment
        if seg % 2 == 0:  # Long segment
            nelem = num_long_elem
        else:  # Short segment
            nelem = num_short_elem
        
        # Calculate the spacing between nodes for this segment
        dx = segment_length / nelem
        
        for i in range(nelem + 1):
            x = segment_start + i * dx
            if i == 0 and seg > 0:
                continue  # Skip the first node if it's coincident with the previous segment end node
            
            XY_list.append([x])
            if i < nelem:
                Elem_nodes_list.append([node_counter, node_counter + 1])
            
            node_counter += 1
        
        XY.append(onp.array(XY_list))
        Elem_nodes.append(onp.array(Elem_nodes_list, dtype=onp.int32))
    # Convert lists to numpy arrays
    
    return XY, Elem_nodes

def uniform_mesh_new(L, nelem_x):
    """ Mesh generator
    --- Inputs ---
    L: length of the domain
    nelem_x: number of elements in x-direction
    dim: problem dimension
    nodes_per_elem: number of nodes in one elements
    elem_type: element type
    --- Outputs ---
    XY: nodal coordinates (nnode, dim)
    Elem_nodes: elemental nodes (nelem, nodes_per_elem)
    connectivity: elemental connectivity (nelem, node_per_elem*dim)
    nnode: number of nodes
    dof_global: global degrees of freedom
    """
    

    dim = 1
    nelem = nelem_x
    nnode = nelem+1 # number of nodes
    dof_global = nnode*dim    
    
    ## Nodes ##
    XY = onp.zeros([nnode, dim], dtype=onp.double)
    dx = L/nelem # increment in the x direction

    n = 0 # This will allow us to go through rows in NL
    for i in range(1, nelem+2):
        if i == 1 or i == nelem+1: # boundary nodes
            XY[n,0] = (i-1)*dx
        else: # inside nodes
            XY[n,0] = (i-1)*dx
        n += 1
        
    ## elements ##
    nodes_per_elem = 2
    Elem_nodes = onp.zeros([nelem, nodes_per_elem], dtype=onp.int32)
    for j in range(1, nelem+1):
        Elem_nodes[j-1, 0] = j-1
        Elem_nodes[j-1, 1] = j 
                   
    return XY, Elem_nodes

def uniform_mesh(L, nelem_x, dim, nodes_per_elem, elem_type, non_uniform_mesh_bool=False):
    """ Mesh generator
    --- Inputs ---
    L: length of the domain
    nelem_x: number of elements in x-direction
    dim: problem dimension
    nodes_per_elem: number of nodes in one elements
    elem_type: element type
    --- Outputs ---
    XY: nodal coordinates (nnode, dim)
    Elem_nodes: elemental nodes (nelem, nodes_per_elem)
    connectivity: elemental connectivity (nelem, node_per_elem*dim)
    nnode: number of nodes
    dof_global: global degrees of freedom
    """
    
    if elem_type == 'D1LN2N': # 1D 2-node linear element
        nelem = nelem_x
        nnode = nelem+1 # number of nodes
        dof_global = nnode*dim    
        
        ## Nodes ##
        XY = onp.zeros([nnode, dim], dtype=onp.double)
        dx = L/nelem # increment in the x direction
    
        n = 0 # This will allow us to go through rows in NL
        for i in range(1, nelem+2):
            if i == 1 or i == nelem+1: # boundary nodes
                XY[n,0] = (i-1)*dx
            else: # inside nodes
                XY[n,0] = (i-1)*dx
                if non_uniform_mesh_bool:
                     XY[n,0] += onp.random.normal(0,0.2,1)*dx# for x values
            n += 1
            
        ## elements ##
        Elem_nodes = onp.zeros([nelem, nodes_per_elem], dtype=onp.int32)
        for j in range(1, nelem+1):
            Elem_nodes[j-1, 0] = j-1
            Elem_nodes[j-1, 1] = j 
                   
    return XY, Elem_nodes, nelem, nnode, dof_global


def write_json_from_data(filename, vtk_filenames, times):
    """
    Writes a JSON file with the provided filenames and corresponding times.
    
    :param filename: str - The path of the JSON file to write to.
    :param vtk_filenames: list of str - List of filenames.
    :param times: list of float - List of times corresponding to the filenames.
    
    # Example usage:
    # vtk_files = ["foo1.vtk", "foo2.vtk", "foo3.vtk"]
    # times = [0, 5.5, 11.2]
    # write_json_from_data('output.json', vtk_files, times)
    """
    if len(vtk_filenames) != len(times):
        raise ValueError("The length of vtk_filenames must be equal to the length of times")

    # Create the list of file dictionaries
    files_list = [{"name": name, "time": time} for name, time in zip(vtk_filenames, times)]
    
    # Define the data structure for JSON
    data = {
        "file-series-version": "1.0",
        "files": files_list
    }
    
    # Write the data to the JSON file
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)  # Pretty print with indentation