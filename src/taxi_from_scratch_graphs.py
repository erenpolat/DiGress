import os
import pandas as pd
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

def round_coordinates(pair):
    coordinates = pair.strip('()').split(', ')
    rounded_coords = tuple(round(float(coord), 1) for coord in coordinates)
    return rounded_coords

def visualize_from_pyg(pyg_graph):
  G = to_networkx(pyg_graph, node_attrs=["x"], edge_attrs=["edge_attr"])
  pos = nx.circular_layout(G) 
  plt.figure(figsize=(6, 4))
  nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_weight='bold', arrows=True)

  # Add edge labels (edge weights) to the visualization
  edge_labels = nx.get_edge_attributes(G, "edge_attr")
  nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

  plt.title("Directed Graph Visualization")
  plt.show()

def load_taxi_data():

  cwd = os.getcwd()
  print("Current Directory is: ", cwd)

  # find the parent directory
  parent = os.path.abspath(os.path.join(cwd, os.pardir))
  ''''folder = "pygeometric_taxi"
  for idx, tensor in enumerate(pyg_graphs):
      torch.save(tensor, f"{folder}/tensor{idx}.pt")'''
  df = pd.read_csv(parent + '/rearranged_nyc_taxi_data.csv')

  print(df.head)

  df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

  # Filter data between 8 am and 9 am
  df = df[
      (df['pickup_datetime'].dt.hour == 8) &  # Check for hour being 8
      (df['pickup_datetime'].dt.minute >= 0) &  # Minutes >= 0 (start of the hour)
      (df['pickup_datetime'].dt.minute < 60)  # Minutes < 60 (end of the hour)
  ]

  print(df.head)

  tables = []

  df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

  # Extracting date from 'pickup_datetime' column
  df['pickup_date'] = df['pickup_datetime'].dt.date

  # Get unique dates from the 'pickup_date' column
  unique_dates = df['pickup_date'].unique()

  # Creating separate tables for each date
  for date in unique_dates:
      # Filter data for the current date
      filtered_data = df[df['pickup_date'] == date]

      # Group by 'pickup_pair' and 'dropoff_pair', then calculate the sum of 'flow_value'
      sum_flow_values = filtered_data.groupby(['pickup_pair', 'dropoff_pair'])['flow_value'].sum().reset_index()

      # Display or save the table for the current date
      tables.append(sum_flow_values)

  print(len(tables))

  for i, df in enumerate(tables):
      # Round 'pickup_pair' and 'dropoff_pair' coordinates to 1 decimal point
      df['pickup_pair_rounded'] = df['pickup_pair'].apply(round_coordinates)
      df['dropoff_pair_rounded'] = df['dropoff_pair'].apply(round_coordinates)

      # Group by the rounded pickup and dropoff pairs, then calculate the sum of 'flow_value'
      tables[i] = df.groupby(['pickup_pair_rounded', 'dropoff_pair_rounded'])['flow_value'].sum().reset_index()

  for i, df in enumerate(tables):
    # Filter rows where 'dropoff_pair_rounded' is not equal to 'pickup_pair_rounded'
    df = df[df['dropoff_pair_rounded'] != df['pickup_pair_rounded']]

    # Displaying the filtered DataFrame
    tables[i] = df

  for i, df in enumerate(tables):
    # Get unique coordinates
    unique_coords = list(set(df['pickup_pair_rounded']).union(set(df['dropoff_pair_rounded'])))

    # Create a mapping dictionary for unique coordinates
    coord_mapping = {coord: i for i, coord in enumerate(unique_coords)}

    # Replace pickup/dropoff pairs with unique values
    df = df.drop(df[df.flow_value >= 30].index)
    df['pickup_pair_unique'] = df['pickup_pair_rounded'].map(coord_mapping)
    df['dropoff_pair_unique'] = df['dropoff_pair_rounded'].map(coord_mapping)

    # Display the DataFrame with unique values for pickup/dropoff pairs
    tables[i] = df
    #print(df[['pickup_pair_unique', 'dropoff_pair_unique', 'flow_value']])

  print(tables[0])

  '''#Create the graphs array of nx objects
  import networkx as nx
  import matplotlib.pyplot as plt
  import torch_geometric
  from torch_geometric.data import Data
  from torch_geometric.utils import from_networkx, to_networkx

  graphs = []
  for df in tables:
    G = nx.DiGraph()
    # Add edges with flow values as weights
    for _, row in df.iterrows():
        G.add_edge(row['pickup_pair_unique'], row['dropoff_pair_unique'], weight=row['flow_value'])
        graphs.append(G)

  # Draw the graph with node labels and edge labels
  G = graphs[0]
  plt.figure(figsize=(6, 4))
  nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_weight='bold', arrows=True)

  # Add edge labels (edge weights) to the visualization
  edge_labels = nx.get_edge_attributes(G, 'weight')
  nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

  plt.title("Directed Graph Visualization")
  plt.show()'''

  pyg_graphs = []
  for df in tables:
    # Create a list of tuples representing directed edges from pickup to dropoff
    edges = list(zip(df['pickup_pair_unique'], df['dropoff_pair_unique']))

    # Create edge indices (transpose to make it 2 x num_edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    # Edge attributes
    edge_attr = torch.tensor(df['flow_value'].to_numpy(dtype=float).reshape(edge_index.shape[-1]), dtype=torch.float)

    # Get the maximum node ID to define the number of nodes
    num_nodes = max(edge_index.max().item(), max(df['dropoff_pair_unique']))

    # Node attr
    node_attr = tensor = torch.arange(num_nodes)

    # Construct a PyTorch Geometric Data object representing the directed graph
    data = Data(edge_index=edge_index, x = node_attr, edge_attr=edge_attr, num_nodes=num_nodes)
    pyg_graphs.append(data)

  pyg_graph = pyg_graphs[0]
  print(pyg_graph)
  print(pyg_graph.x)
  print(pyg_graph.edge_attr)
  print(pyg_graph.edge_index)

  visualize_from_pyg(pyg_graph)


  #NOT DONE, label the node attributes
  '''
  for i, data in enumerate(pyg_graphs[0]):
    # Get the edge index (assuming a directed graph)
    edge_index = data.edge_index

    # Create a boolean mask to identify nodes with outgoing edges
    has_outgoing_edges = torch.zeros(num_nodes, dtype=torch.bool)
    has_outgoing_edges[edge_index[0]] = 1

    # Create a boolean mask to identify nodes with incoming edges
    has_incoming_edges = torch.zeros(num_nodes, dtype=torch.bool)
    has_incoming_edges[edge_index[1] - 1] = 1

    # Nodes with no outgoing edges are labeled as sink nodes
    sink_nodes = (~has_outgoing_edges).nonzero(as_tuple=False).squeeze()

    # Nodes with no incoming edges are labeled as source nodes
    source_nodes = (~has_incoming_edges).nonzero(as_tuple=False).squeeze()

    # Nodes with both incoming and outgoing edges are labeled as normal nodes
    normal_nodes = (has_incoming_edges & has_outgoing_edges).nonzero(as_tuple=False).squeeze()

    # Labeling the nodes based on categories
    node_attr = torch.zeros(num_nodes, dtype=torch.long)  # Initialize with zeros (normal nodes)
    node_attr[sink_nodes] = 1  # Label sink nodes as 1
    node_attr[source_nodes] = 2  # Label source nodes as 2

    # Creating a new graph with node labels
    data_with_labels = Data(edge_index=data.edge_index, edge_attr=data.edge_attr, x=node_attr)
    pyg_graphs[i] = data_with_labels

  visualize_from_pyg(pyg_graphs[0])'''

  


if __name__ == '__main__':
    load_taxi_data()
