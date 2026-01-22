
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.embeddings import get_timestep_embedding




class FourierFeatures(nn.Module):
    def __init__(self, output_size=64, learnable=False):
        super().__init__()
        self.output_size = output_size
        self.learnable = learnable
        
        if self.learnable:
            self.w = nn.Parameter(torch.randn(output_size // 2, 1) * 0.2)
        else:
            half_dim = output_size // 2
            f = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            f = torch.exp(torch.arange(half_dim) * -f)
            self.register_buffer('f', f)
    
    def forward(self, x):
        """Apply Fourier features to input."""
        if self.learnable:
            f = 2 * torch.pi * x @ self.w.T
        else:
            f = x * self.f
        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)

class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()
    def forward(self,x):
        return x



class MLP(nn.Module):
    """
    MLP Model (PyTorch Implementation)
    """
    def __init__(self,
                 input_dim,
                 action_dim,     
                 hidden_dim=(512, 512, 512, 512),
                 activations=nn.GELU,       
                 activate_final=False, 
                 layer_norm = False,
                ):            
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        layers = []
        for h_dim in hidden_dim:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(activations()) 
            if layer_norm:
                layers.append(nn.LayerNorm(h_dim)) 
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, action_dim))
        
        if activate_final:
            layers.append(activations())
            
        self.model = nn.Sequential(*layers)
        
        # self.to(self.device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
            
        return self.model(x)

class ResnetStack(nn.Module):
    """ResNet stack module."""
    def __init__(self, in_channels, num_features, num_blocks, max_pooling=True):
        super().__init__()
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.max_pooling = max_pooling

        self.conv_in = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1)
        
        nn.init.xavier_uniform_(self.conv_in.weight)
        if self.conv_in.bias is not None:
            nn.init.zeros_(self.conv_in.bias)

        if max_pooling:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'conv1': nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
                'conv2': nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
            })
            for _ in range(num_blocks)
        ])
        
        for block in self.blocks:
            for name, layer in block.items():
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.output_dim = num_features

    def forward(self, x):
        conv_out = self.conv_in(x)

        if self.max_pooling:
            conv_out = self.max_pool(conv_out)

        # Residual Blocks
        for block in self.blocks:
            block_input = conv_out
            
            out = F.relu(conv_out)
            out = block['conv1'](out)
            out = F.relu(out)
            out = block['conv2'](out)
            
            conv_out = out + block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""
    def __init__(self, 
                 input_shape,  # (C, H, W)
                 width=1, 
                 stack_sizes=(16, 32, 32), 
                 num_blocks=2, 
                 dropout_rate=None, 
                 mlp_hidden_dims=(512,), 
                 layer_norm=False):
        super().__init__()
        
        self.width = width
        self.stack_sizes = stack_sizes
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm

        # 1. Build Stacks
        self.stacks = nn.ModuleList()
        current_channels = input_shape[0]
        
        for i, size in enumerate(stack_sizes):
            out_channels = size * width
            stack = ResnetStack(
                in_channels=current_channels,
                num_features=out_channels,
                num_blocks=num_blocks,
                max_pooling=True
            )
            self.stacks.append(stack)
            current_channels = out_channels

        # 2. Dropout
        if dropout_rate is not None and dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

        # 3. Calculate Flatten Dim (Dummy Pass)
        # 这是 PyTorch 中最稳健的做法，不需要手动计算卷积公式
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_out = self._forward_conv(dummy_input)
            self.flatten_dim = dummy_out.reshape(1, -1).shape[1]

        # 4. Layer Norm
        if layer_norm:
            self.ln = nn.LayerNorm(current_channels)
        else:
            self.ln = None

        # 5. MLP
        # 修正：删除了这里原本重复的第一次初始化，只保留这一个
        self.mlp = MLP(
            input_dim=self.flatten_dim,
            action_dim=mlp_hidden_dims[-1], # 最后一层作为 Output
            hidden_dim=mlp_hidden_dims[:-1] if len(mlp_hidden_dims) > 1 else (),
            activate_final=True # 对应 JAX 中的逻辑
        )
        
        self.output_dim = mlp_hidden_dims[-1]

    def _forward_conv(self, x):
        # Normalize
        x = x.float() / 255.0
        for stack in self.stacks:
            x = stack(x)
            if self.dropout is not None:
                x = self.dropout(x)
        return x

    def forward(self, x, train=True, cond_var=None):
        # 注意：PyTorch 的 dropout 不需要 train 参数，它是通过 model.train() / model.eval() 控制的
        conv_out = self._forward_conv(x)
        conv_out = F.relu(conv_out)
        
        if self.ln is not None:
            # (N, C, H, W) -> (N, H, W, C) for LayerNorm
            conv_out = conv_out.permute(0, 2, 3, 1)
            conv_out = self.ln(conv_out)
            out = conv_out.reshape(conv_out.size(0), -1)
        else:
            out = conv_out.reshape(conv_out.size(0), -1)

        out = self.mlp(out)
        return out

class Value(nn.Module):
    """
    Value/Critic Network (PyTorch implementation).
    可以作为 V(s) 网络 (当 action_dim=0 或 None)
    也可以作为 Q(s, a) 网络 (当 action_dim > 0)
    
    特点:
    - 包含 num_ensembles 个独立的 MLP 网络 (Ensemble)。
    - 输出形状为 (num_ensembles, batch_size)，方便计算 min/mean。
    """
    def __init__(self, 
                 observation_dim, 
                 action_dim=None,          # 如果是 V 网络，不传或传 0
                 hidden_dim=(512, 512, 512, 512),    # 对应 JAX 中的 hidden_dims
                 num_ensembles=2,          # 默认通常是 2 个 Q 网络
                 encoder=None,             # 可选的 encoder module
                 layer_norm=True,          # JAX default is True
                ):
        super(Value, self).__init__()
        
        self.num_ensembles = num_ensembles
        self.encoder = encoder
        if self.encoder is not None:
            self.input_dim = self.encoder.output_dim
        else:
            self.input_dim = observation_dim
        
        if action_dim is not None and action_dim > 0:
            self.input_dim += action_dim
            
        # 2. 构建 Ensemble
        # 使用 nn.ModuleList 创建多个独立的 MLP 实例
        # 这里的 MLP 是你刚才定义的类，输出维度设为 1 (代表 Value 值)
        self.nets = nn.ModuleList([
            MLP(input_dim=self.input_dim,
                action_dim=1,           
                hidden_dim=hidden_dim,
                activations=nn.GELU,    
                activate_final=False,
                layer_norm=layer_norm,  # Add layer_norm support
                )
            for _ in range(num_ensembles)
        ])
        
        
            
    def forward(self, observations, actions=None):
        """
        Args:
            observations: (batch_size, obs_dim)
            actions: (batch_size, action_dim) [可选]
            
        Returns:
            values: (num_ensembles, batch_size)
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
            
        if actions is not None:
            inputs = torch.cat([inputs, actions], dim=-1)
            
        
        outputs = []
        for net in self.nets:
            out = net(inputs)
            outputs.append(out)
            
        # [batch_size, 1] * N -> [num_ensembles, batch_size, 1]
        outputs = torch.stack(outputs, dim=0)
        
        # 去掉最后一个维度 -> [num_ensembles, batch_size]
        return outputs.squeeze(-1)
    
class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Supports optional time conditioning with configurable time encoders.

    Args:
        observation_dim: Dimension of observations
        action_dim: Dimension of actions
        hidden_dim: Hidden layer dimensions for MLP
        encoder: Optional observation encoder
        time_encoder: Type of time encoding. Options:
            - None: No time conditioning (use_time=False)
            - "sinusoidal": Sinusoidal positional embeddings (default)
            - "fourier": Random Fourier features
            - "positional": Learnable positional embeddings
        time_encoder_dim: Dimension of time embeddings (default: 64)
        use_time: Whether to use time conditioning (deprecated, use time_encoder=None instead)
    """
    def __init__(self, observation_dim, action_dim, hidden_dim=(512,512,512,512),
                 encoder=None, time_encoder="sinusoidal", time_encoder_dim=64,
                 use_fourier_features=False, fourier_feature_dim=64,
                 use_time=True, layer_norm=False):
        super(ActorVectorField, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.encoder = encoder

        # Handle legacy parameters
        if not use_time:
            time_encoder = None
        elif use_fourier_features and time_encoder == "sinusoidal":
            time_encoder = "fourier"
            time_encoder_dim = fourier_feature_dim

        self.time_encoder_type = time_encoder
        self.use_time = time_encoder is not None

        if self.encoder is not None:
            self.input_dim = self.encoder.output_dim
        else:
            self.input_dim = observation_dim

        # Add action dimension
        self.input_dim += self.action_dim

        # Setup time encoder
        if self.use_time:
            self.time_encoder = get_timestep_embedding(time_encoder, time_encoder_dim)
            self.input_dim += time_encoder_dim
        else:
            self.time_encoder = None

        self.mlp = MLP(self.input_dim, self.action_dim, hidden_dim=hidden_dim, layer_norm=layer_norm)
    
    def forward(self, o, x_t, t=None, is_encoded=False):
        """Forward pass.

        Args:
            o: observations
            x_t: noisy actions
            t: time (optional, only used if use_time=True)
            is_encoded: whether observations are already encoded
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(o)
        else:
            observations = o

        if self.use_time and t is not None:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=observations.device).float()

            # Ensure t has correct shape for time encoder
            # Time encoder expects (batch_size,) shape
            if t.dim() == 0:
                # Scalar: expand to match batch size
                batch_size = observations.shape[0]
                t = t.unsqueeze(0).expand(batch_size)
            elif t.dim() == 1:
                # Already 1D, check if it needs expansion
                if t.shape[0] == 1 and observations.shape[0] > 1:
                    t = t.expand(observations.shape[0])
            elif t.dim() == 2 and t.shape[1] == 1:
                # (batch, 1) -> (batch,)
                t = t.squeeze(1)
            elif t.dim() > 1:
                # Multi-dimensional, squeeze to 1D
                t = t.squeeze()
                if t.dim() == 0:
                    batch_size = observations.shape[0]
                    t = t.unsqueeze(0).expand(batch_size)

            # Apply time encoder
            time_emb = self.time_encoder(t)

            inputs = torch.cat([observations, x_t, time_emb], dim=-1)
        else:
            # No time input - just concat observations and actions
            inputs = torch.cat([observations, x_t], dim=-1)

        v = self.mlp(inputs)
        return v


class MeanActorVectorField(nn.Module):
    """Actor vector field with t_begin and t_end time conditioning (for JVP-based flow matching).

    Args:
        observation_dim: Dimension of observations
        action_dim: Dimension of actions
        hidden_dim: Hidden layer dimensions for MLP
        encoder: Optional observation encoder
        time_encoder: Type of time encoding. Options:
            - "sinusoidal": Sinusoidal positional embeddings (default)
            - "fourier": Random Fourier features
            - "positional": Learnable positional embeddings
        time_encoder_dim: Dimension of time embeddings (default: 64)
    """

    def __init__(self, observation_dim, action_dim, hidden_dim=(512,512,512,512),
                 encoder=None, time_encoder="sinusoidal", time_encoder_dim=64,
                 use_fourier_features=False, fourier_feature_dim=64, layer_norm=False):
        super(MeanActorVectorField, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.encoder = encoder

        # Handle legacy parameters
        if use_fourier_features and time_encoder == "sinusoidal":
            time_encoder = "fourier"
            time_encoder_dim = fourier_feature_dim

        self.time_encoder_type = time_encoder

        if self.encoder is not None:
            self.input_dim = self.encoder.output_dim
        else:
            self.input_dim = observation_dim

        # Create time encoders for t_begin and t_end
        self.time_encoder_begin = get_timestep_embedding(time_encoder, time_encoder_dim)
        self.time_encoder_end = get_timestep_embedding(time_encoder, time_encoder_dim)

        # Input: observations + actions + time_begin_emb + time_end_emb
        self.input_dim += self.action_dim + time_encoder_dim * 2

        self.mlp = MLP(self.input_dim, self.action_dim, hidden_dim=hidden_dim, layer_norm=layer_norm)

    def forward(self, observations, x_t, t_begin, t_end, is_encoded=False):
        """Forward pass.

        Args:
            observations: Observation tensor
            x_t: Noisy actions
            t_begin: Begin time for JVP computation
            t_end: End time for JVP computation
            is_encoded: Whether observations are already encoded
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        if not isinstance(t_begin, torch.Tensor):
            t_begin = torch.tensor(t_begin, device=observations.device).float()
        if not isinstance(t_end, torch.Tensor):
            t_end = torch.tensor(t_end, device=observations.device).float()

        # Ensure times have correct shape for time encoder
        # Time encoder expects (batch_size,) shape
        batch_size = observations.shape[0]

        # Process t_begin
        if t_begin.dim() == 0:
            t_begin = t_begin.unsqueeze(0).expand(batch_size)
        elif t_begin.dim() == 1:
            if t_begin.shape[0] == 1 and batch_size > 1:
                t_begin = t_begin.expand(batch_size)
        elif t_begin.dim() == 2 and t_begin.shape[1] == 1:
            t_begin = t_begin.squeeze(1)
        elif t_begin.dim() > 1:
            t_begin = t_begin.squeeze()
            if t_begin.dim() == 0:
                t_begin = t_begin.unsqueeze(0).expand(batch_size)

        # Process t_end
        if t_end.dim() == 0:
            t_end = t_end.unsqueeze(0).expand(batch_size)
        elif t_end.dim() == 1:
            if t_end.shape[0] == 1 and batch_size > 1:
                t_end = t_end.expand(batch_size)
        elif t_end.dim() == 2 and t_end.shape[1] == 1:
            t_end = t_end.squeeze(1)
        elif t_end.dim() > 1:
            t_end = t_end.squeeze()
            if t_end.dim() == 0:
                t_end = t_end.unsqueeze(0).expand(batch_size)

        # Apply time encoders
        t_begin_emb = self.time_encoder_begin(t_begin)
        t_end_emb = self.time_encoder_end(t_end)

        inputs = torch.cat([observations, x_t, t_begin_emb, t_end_emb], dim=-1)

        v = self.mlp(inputs)
        return v





