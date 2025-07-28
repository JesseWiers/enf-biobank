import jax
import jax.numpy as jnp
import flax.linen as nn

def segment_mean(data, segment_ids, num_segments):
    """
    Computes the mean within segments of an array.
    """
    # Sum the data within each segment
    segment_sums = jax.ops.segment_sum(data, segment_ids, num_segments)
    # Compute the size of each segment
    segment_sizes = jax.ops.segment_sum(jnp.ones_like(data), segment_ids, num_segments)
    segment_means = segment_sums / segment_sizes
    return segment_means


def compute_radial(edge_index, x):
    """
    Compute x_i - x_j and ||x_i - x_j||^2.
    """
    senders, receivers = edge_index
    x_i, x_j = x[senders], x[receivers]
    distance = jnp.sum((x_i - x_j) ** 2, axis=1, keepdims=True)
    return distance

def custom_xavier_uniform_init(gain=0.001):
    """
    Low variance initialization used in positional MLPs
    """

    def init(key, shape, dtype=jnp.float32):
        std = gain * jnp.sqrt(2.0 / shape[0])
        return jax.random.uniform(key, shape, dtype, -std, std)

    return init


def build_fn(hidden_dim, act_fn):
    """
    EGNN primitives as functions
        1. message function (eq. 3)
        2. message aggregation + node update (eq. 5,6)
        3. message aggregation + positional update (eq. 4)
    """

    def message_fn(edge_index, h, dist, edge_attr):
        """
        Message: m_ij = phi_e(h_i^l, h_j^l, ||x_i^l - x_j^l||^2, a_ij)
        """
        phi_e = nn.Sequential(
            [
                nn.Dense(hidden_dim), 
                act_fn, 
                nn.Dense(hidden_dim), 
                act_fn
            ]
        )

        senders, receivers = edge_index
        h_i, h_j = h[senders], h[receivers]
        out = jnp.concatenate([h_i, h_j, dist, edge_attr], axis=1)

        return phi_e(out)

    def agg_update_fn(edge_index, h_i, m_ij):
        """
        Aggregation: m_i = sum_{j!=i} m_ij

        Node update: h_i^{l+1} = phi_h(h_i^l, m_i)
        """
        phi_h = nn.Sequential(
            [
                nn.Dense(hidden_dim), 
                act_fn, 
                nn.Dense(hidden_dim)
                ]
        )

        senders, _ = edge_index
        m_i = jax.ops.segment_sum(m_ij, senders, num_segments=h_i.shape[0])
        out = jnp.concatenate([h_i, m_i], axis=1)

        return h_i + phi_h(out)

    def pos_agg_update_fn(edge_index, x, m_ij):
        """
        Positional update: x_i^{l+1} = x_i^l + mean_{j!=i} (x_i^l - x_j^l) phi_x(m_ij)
        """
        phi_x = nn.Sequential(
            [
                nn.Dense(hidden_dim),
                act_fn,
                nn.Dense(1, kernel_init=custom_xavier_uniform_init(gain=0.001)),
            ]
        )

        senders, receivers = edge_index
        x_i, x_j = x[senders], x[receivers]
        x_ij = (x_i - x_j) * phi_x(m_ij)

        return x + segment_mean(x_ij, senders, num_segments=x.shape[0])

    return message_fn, agg_update_fn, pos_agg_update_fn


class EGNN_layer(nn.Module):
    hidden_dim: int
    act_fn: callable

    @nn.compact
    def __call__(self, edge_index, h, x, edge_attr):
        # get primitives
        message_fn, agg_update_fn, pos_agg_update_fn = build_fn(
            self.hidden_dim, self.act_fn
        )
        # compute the distance between connected nodes
        dist = compute_radial(edge_index, x)
        # message -> aggregation -> node update, position update
        m_ij = message_fn(edge_index, h, dist, edge_attr)
        x = pos_agg_update_fn(edge_index, x, m_ij)
        h = agg_update_fn(edge_index, h, m_ij)
        return h, x


class EGNN(nn.Module):
    hidden_dim: int
    out_dim: int
    num_layers: int
    act_fn: callable = jax.nn.silu

    @nn.compact
    def __call__(self, edge_index, h, x, edge_attr):
        
        # edge_index is a tuple of (senders, receivers) -> index of the connected nodes
        # h is a tensor of shape (num_nodes, hidden_dim) -> features of each node
        # x is a tensor of shape (num_nodes, 3) -> position of each node
        # edge_attr is a tensor of shape (num_edges, 1) -> edge features

        h = nn.Dense(self.hidden_dim)(h)

        for _ in range(self.num_layers):
            h, x = EGNN_layer(self.hidden_dim, self.act_fn)(edge_index, h, x, edge_attr)

        h = nn.Dense(self.out_dim)(h)

        return h, x
    
    

class EGNNClassifier(nn.Module):
    """
    EGNN-based classifier for latent point clouds.
    """
    hidden_dim: int
    num_layers: int
    num_classes: int
    act_fn: callable = jax.nn.silu
    k_neighbors: int = 8  # number of nearest neighbors for graph construction

    @nn.compact
    def __call__(self, p_0, c_0, g_0):
        batch_size = p_0.shape[0]
        
        # 1. Construct edges (k-nearest neighbors in 4D space)
        def get_edges(positions):
            # Compute pairwise distances
            diff = positions[:, None, :] - positions[None, :, :]
            dist = jnp.sum(diff ** 2, axis=-1)
            
            # Get k nearest neighbors for each point
            _, indices = jax.lax.top_k(-dist, k=self.k_neighbors + 1)
            
            # Create edge index (excluding self-loops)
            senders = jnp.repeat(jnp.arange(positions.shape[0]), self.k_neighbors)
            receivers = indices[:, 1:].reshape(-1)  # exclude self
            
            return jnp.stack([senders, receivers])

        # Process each batch item separately (since they're independent graphs)
        def process_batch_item(inputs):
            p, c, g = inputs
            
            # Construct edges
            edge_index = get_edges(p)
            
            # Create edge attributes (could be distances or other features)
            edge_attr = jnp.zeros((edge_index.shape[1], 1))  
            
            # Apply EGNN
            h_out, x_out = EGNN(
                hidden_dim=self.hidden_dim,
                out_dim=self.hidden_dim,  # intermediate output
                num_layers=self.num_layers,
                act_fn=self.act_fn
            )(edge_index, c, p, edge_attr)
            
            # Global pooling
            return jnp.mean(h_out, axis=0)

        # Process batch
        features = jax.vmap(process_batch_item)((p_0, c_0, g_0))
        
        # Final classification layer
        return nn.Dense(self.num_classes)(features)
    
    
