import plotly.graph_objects as go


# Tree Plot https://plotly.com/python/tree-plots/
def plot_decision_tree(root):
    global trace_nodes, trace_edges
    trace_nodes = []
    trace_edges = []

    def add_trace_node(x, y, info):
        trace_nodes.append(go.Scatter(
            x=[x], y=[y], text=[info],
            mode='markers+text',
            textposition="bottom center",
            marker=dict(symbol='circle',
                        size=35,
                        color='#156082',
                        line=dict(color='rgb(50,50,50)', width=1)
                        ),
            showlegend=False,
            hoverinfo='text'
            )
        )

    def add_trace_edge(x_from, y_from, x_to, y_to):
        trace_edges.append(go.Scatter(
            x=[x_from, x_to], y=[y_from, y_to],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
			hoverinfo='none'
            )
        )

    def traverse(node, x=0, y=0, dx=1.5, dy=-1, parent_pos=None):
        """Perform a recursive depth first, pre order traversal of a binary tree to create a visual representation of the tree using Plotly"""
        info = f"Depth: {node.depth}<br>{'Label: ' if node.is_leaf else ''}{node.label if node.is_leaf else f'{node.feature} <= {node.value}'}"
        add_trace_node(x, y, info)

        if parent_pos:
            add_trace_edge(parent_pos[0], parent_pos[1], x, y)

        spacing = dx * 2 ** (-y)
        if node.left:
            traverse(node.left, x - spacing, y + dy, dx, dy, (x, y))
        if node.right:
            traverse(node.right, x + spacing, y + dy, dx, dy, (x, y))

    # Start the traversal with the root node
    traverse(root)

    # hide axis line, grid, ticklabels and  title
    axis = dict(showline=False, zeroline=False, showgrid=False, showticklabels=False)
    
    # Setup plot layout
    layout = go.Layout(height=700,
                       width=900,
                       plot_bgcolor='white',
                       xaxis=axis,
                       yaxis=axis,
                       margin={'l': 50, 'r': 50, 't': 50, 'b': 50},
                       hovermode='closest')

    # Create figure and display, draw edges before nodes
    fig = go.Figure(data=trace_edges + trace_nodes, layout=layout)
    fig.show()

    # Saving to PDF for vector format
    file_name = "plots/Decision_Tree.pdf"
    fig.write_image(file_name)
    print(f"Figure saved to {file_name}")
