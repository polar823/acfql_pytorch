"""Visual observation encoders for image-based tasks.

Ported from much-ado-about-noising repository.
Supports ResNet-based visual encoders and multi-modal observations.
"""

import copy
from collections.abc import Callable

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as ttf


def at_least_ndim(x, ndim):
    """Ensure tensor has at least ndim dimensions by adding trailing dimensions."""
    while x.ndim < ndim:
        x = x.unsqueeze(-1)
    return x


class BaseEncoder(nn.Module):
    """Base class for all encoders.

    All encoders should inherit from this class and implement the forward method
    with signature: forward(condition, mask) -> encoded_condition
    """

    def __init__(self):
        super().__init__()

    def forward(self, condition: torch.Tensor | dict, mask: torch.Tensor = None):
        """Encode the condition.

        Args:
            condition: Input condition (tensor or dict of tensors)
            mask: Optional mask tensor

        Returns:
            Encoded condition tensor
        """
        raise NotImplementedError("Subclasses must implement forward method")


class IdentityEncoder(BaseEncoder):
    """Identity encoder that passes through the input with optional dropout."""

    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.dropout = dropout

    def forward(self, condition: torch.Tensor | dict, mask: torch.Tensor = None):
        """
        Args:
            condition: (batch, *shape) tensor or dict of tensors
            mask: (batch,) mask tensor or None
        Returns:
            condition with mask applied
        """
        # Handle dict input by concatenating all tensors
        if isinstance(condition, dict):
            keys = sorted(condition.keys())
            tensors = [condition[k] for k in keys]
            flattened = [t.reshape(t.shape[0], -1) for t in tensors]
            condition = torch.cat(flattened, dim=-1)

        # Apply mask
        if mask is None:
            if self.training:
                mask = (torch.rand(condition.shape[0], device=condition.device) > self.dropout).float()
            else:
                mask = 1.0

        mask = at_least_ndim(mask, condition.ndim)
        return condition * mask


class MLPEncoder(BaseEncoder):
    """MLP encoder for low-dimensional observations."""

    def __init__(
        self,
        obs_dim: int,
        emb_dim: int,
        To: int,
        hidden_dims: list[int],
        act=nn.LeakyReLU(),
        dropout: float = 0.25,
    ):
        super().__init__()
        self.dropout = dropout
        self.To = To
        self.emb_dim = emb_dim

        hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims

        # Build MLP
        layers = []
        in_dim = obs_dim * To
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                act,
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, emb_dim * To))

        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor | dict, mask: torch.Tensor = None):
        """
        Args:
            obs: (batch, To, obs_dim) tensor or dict of tensors
            mask: (batch,) mask tensor or None
        Returns:
            (batch, To, emb_dim) encoded observations
        """
        # Handle dict input
        if isinstance(obs, dict):
            keys = sorted(obs.keys())
            obs_list = [obs[k] for k in keys]
        else:
            obs_list = [obs]

        # Flatten and concatenate
        flattened = [t.reshape(t.shape[0], -1) for t in obs_list]
        obs = torch.cat(flattened, dim=-1)

        # Apply mask
        if mask is None:
            if self.training:
                mask = (torch.rand(obs.shape[0], device=obs.device) > self.dropout).float()
            else:
                mask = 1.0

        mask = at_least_ndim(mask, obs.ndim)
        emb_features = self.mlp(obs) * mask
        return emb_features.reshape(obs.shape[0], self.To, self.emb_dim)


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """Replace submodules matching predicate with func output."""
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    return root_module


def get_resnet(name, weights=None, **kwargs):
    """name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m".
    """
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet


class CropRandomizer(nn.Module):
    """Randomly crop images during training, center crop during eval."""

    def __init__(
        self,
        input_shape,
        crop_height,
        crop_width,
        num_crops=1,
        pos_enc=False,
    ):
        super().__init__()
        assert len(input_shape) == 3  # (C, H, W)
        assert crop_height < input_shape[1]
        assert crop_width < input_shape[2]

        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def forward(self, inputs):
        """Apply random crops during training, center crop during eval."""
        if self.training:
            # Random crop
            B, C, H, W = inputs.shape
            # Simple random crop implementation
            h_start = torch.randint(0, H - self.crop_height + 1, (B,), device=inputs.device)
            w_start = torch.randint(0, W - self.crop_width + 1, (B,), device=inputs.device)

            crops = []
            for i in range(B):
                crop = inputs[i:i+1, :, h_start[i]:h_start[i]+self.crop_height,
                             w_start[i]:w_start[i]+self.crop_width]
                crops.append(crop)
            return torch.cat(crops, dim=0)
        else:
            # Center crop
            return ttf.center_crop(inputs, (self.crop_height, self.crop_width))


class MultiImageObsEncoder(BaseEncoder):
    """Multi-modal observation encoder supporting both RGB and low-dim inputs.

    Supports pretrained vision encoders (e.g., ImageNet-pretrained ResNet) and
    optional freezing of encoder parameters for sample-efficient training.
    """

    def __init__(
        self,
        shape_meta: dict,
        rgb_model_name: str,
        emb_dim: int = 256,
        resize_shape: tuple[int, int] | dict[str, tuple] | None = None,
        crop_shape: tuple[int, int] | dict[str, tuple] | None = None,
        random_crop: bool = True,
        use_group_norm: bool = False,
        share_rgb_model: bool = False,
        imagenet_norm: bool = False,
        use_seq=False,
        keep_horizon_dims=False,
        pretrained: bool = True,
        freeze_rgb_encoder: bool = True,
    ):
        """Initialize multi-modal observation encoder.

        Args:
            shape_meta: Shape metadata dict
            rgb_model_name: ResNet model name ('resnet18', 'resnet34', 'resnet50')
            emb_dim: Output embedding dimension
            resize_shape: Optional resize shape for images
            crop_shape: Optional crop shape for images
            random_crop: Use random crop during training
            use_group_norm: Replace BatchNorm with GroupNorm
            share_rgb_model: Share RGB encoder across multiple cameras
            imagenet_norm: Use ImageNet normalization
            use_seq: Handle sequential observations
            keep_horizon_dims: Keep horizon dimensions in output
            pretrained: Load pretrained ImageNet weights (default: True)
            freeze_rgb_encoder: Freeze RGB encoder parameters (default: True)
        """
        super().__init__()
        rgb_keys = []
        low_dim_keys = []
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = {}

        self.pretrained = pretrained
        self.freeze_rgb_encoder = freeze_rgb_encoder

        # Get RGB model with optional pretrained weights
        if "resnet" in rgb_model_name:
            weights = 'IMAGENET1K_V1' if pretrained else None
            rgb_model = get_resnet(rgb_model_name, weights=weights)
            if pretrained:
                print(f"✓ Loaded pretrained {rgb_model_name} weights from ImageNet")
        else:
            raise ValueError(f"Unsupported rgb_model: {rgb_model_name}")

        # Handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map["rgb"] = rgb_model

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            key_shape_map[key] = shape

            if type == "rgb":
                rgb_keys.append(key)
                # Configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        this_model = rgb_model[key]
                    else:
                        this_model = copy.deepcopy(rgb_model)

                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features // 16,
                                num_channels=x.num_features,
                            ),
                        )
                    key_model_map[key] = this_model

                # Configure transforms
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (shape[0], h, w)

                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False,
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(size=(h, w))

                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )

                this_transform = nn.Sequential(
                    this_resizer, this_randomizer, this_normalizer
                )
                key_transform_map[key] = this_transform

            elif type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.use_seq = use_seq
        self.keep_horizon_dims = keep_horizon_dims

        # Freeze RGB encoder if requested
        if freeze_rgb_encoder:
            self._freeze_rgb_encoder()
            print(f"✓ Froze RGB encoder parameters (only MLP projection will be trained)")

        # MLP to project concatenated features to emb_dim
        # This MLP is always trainable, even when RGB encoder is frozen
        self.mlp = nn.Sequential(
            nn.Linear(self.output_shape(), emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # Set output_dim based on use_seq and keep_horizon_dims
        if use_seq and not keep_horizon_dims:
            # When flattening sequence, output_dim is emb_dim * seq_len
            # But we don't know seq_len here, so we set it to emb_dim
            # The actual output will be (batch, seq_len * emb_dim)
            self.output_dim = emb_dim  # Base dimension per timestep
            self._base_output_dim = emb_dim
        else:
            self.output_dim = emb_dim
            self._base_output_dim = emb_dim

    def _freeze_rgb_encoder(self):
        """Freeze all RGB encoder parameters."""
        for key in self.rgb_keys:
            if key in self.key_model_map:
                for param in self.key_model_map[key].parameters():
                    param.requires_grad = False
        if self.share_rgb_model and "rgb" in self.key_model_map:
            for param in self.key_model_map["rgb"].parameters():
                param.requires_grad = False

    def unfreeze_rgb_encoder(self):
        """Unfreeze all RGB encoder parameters (for fine-tuning)."""
        for key in self.rgb_keys:
            if key in self.key_model_map:
                for param in self.key_model_map[key].parameters():
                    param.requires_grad = True
        if self.share_rgb_model and "rgb" in self.key_model_map:
            for param in self.key_model_map["rgb"].parameters():
                param.requires_grad = True
        print("✓ Unfroze RGB encoder parameters")

    def multi_image_forward(self, obs_dict):
        """Process multi-modal observations."""
        batch_size = None
        seq_len = None
        features = []

        # Process RGB inputs
        if self.share_rgb_model:
            imgs = []
            for key in self.rgb_keys:
                img = obs_dict[key]
                if self.use_seq:
                    if batch_size is None:
                        batch_size = img.shape[0]
                        seq_len = img.shape[1]
                    img = img.reshape(batch_size * seq_len, *img.shape[2:])
                else:
                    if batch_size is None:
                        batch_size = img.shape[0]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            feature = self.key_model_map["rgb"](imgs)
            num_keys = len(self.rgb_keys)
            feature_dim = feature.shape[-1]
            feature = feature.view(
                num_keys,
                batch_size if not self.use_seq else batch_size * seq_len,
                feature_dim,
            )
            feature = torch.moveaxis(feature, 0, 1)
            feature = feature.reshape(
                batch_size if not self.use_seq else batch_size * seq_len, -1
            )
            features.append(feature)
        else:
            for key in self.rgb_keys:
                img = obs_dict[key]
                if self.use_seq:
                    if batch_size is None:
                        batch_size = img.shape[0]
                        seq_len = img.shape[1]
                    img = img.reshape(batch_size * seq_len, *img.shape[2:])
                else:
                    if batch_size is None:
                        batch_size = img.shape[0]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)

        # Process low-dim inputs
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if self.use_seq:
                if batch_size is None:
                    batch_size = data.shape[0]
                    seq_len = data.shape[1]
                data = data.reshape(batch_size * seq_len, *data.shape[2:])
            else:
                if batch_size is None:
                    batch_size = data.shape[0]
            features.append(data)

        # Concatenate all features
        features = torch.cat(features, dim=-1)
        return features, batch_size, seq_len

    def forward(self, obs_dict, mask=None):
        """
        Args:
            obs_dict: Dict of observations
            mask: Optional mask
        Returns:
            Encoded observations
        """
        features, batch_size, seq_len = self.multi_image_forward(obs_dict)
        result = self.mlp(features)

        if self.use_seq:
            if self.keep_horizon_dims:
                result = result.view(batch_size, seq_len, -1)
            else:
                result = result.view(batch_size, -1)
        return result

    @torch.no_grad()
    def output_shape(self):
        """Compute output shape by running a forward pass."""
        example_obs_dict = {}
        obs_shape_meta = self.shape_meta["obs"]
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            prefix = (batch_size, 1) if self.use_seq else (batch_size,)
            this_obs = torch.zeros(prefix + shape, dtype=self.dtype, device=self.device)
            example_obs_dict[key] = this_obs
        example_output, _, _ = self.multi_image_forward(example_obs_dict)
        return example_output.shape[1]

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype






