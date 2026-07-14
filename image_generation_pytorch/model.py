import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PolynomialGenerator2D(nn.Module):
    """
    Polynomial generator for 2D point data.

    Input:
        z with shape [batch_size, z_dim]

    Output:
        generated points with shape [batch_size, 2]
        Each row is one point: [x_coordinate, y_coordinate]
    """

    def __init__(
        self,
        z_dim=16,
        hidden_dim=128,
        num_orders=3,
        activation_fn=False,
        bound_output=False
    ):
        super().__init__()

        # z_dim:
        # Number of random values in one noise vector.
        # This is not the number of circle coordinates.
        # Example: one z vector has shape [16].
        self.z_dim = z_dim

        # hidden_dim:
        # Number of internal features used by the generator.
        # 128 is a design choice, not a property of the circle.
        self.hidden_dim = hidden_dim

        # num_orders:
        # Maximum polynomial order created by repeated multiplication.
        # Starting representation is first order.
        # Every additional injection can increase the order by one.
        if num_orders < 1:
            raise ValueError("num_orders must be at least 1")

        self.num_orders = num_orders

        # ReLU can improve practical training.
        #
        # activation_fn=False:
        # The generator remains a strict polynomial function,
        # apart from an optional output Tanh.
        #
        # activation_fn=True:
        # The generator becomes piecewise polynomial because of ReLU.
        self.activation = (
            nn.ReLU(inplace=False)
            if activation_fn
            else nn.Identity()
        )

        # First transformation:
        #
        # [batch, z_dim] -> [batch, hidden_dim]
        #
        # Mathematically:
        # h_1 = W_1 z + b_1
        self.first_layer = nn.Linear(
            in_features=z_dim,
            out_features=hidden_dim,
            bias=True
        )

        # These layers repeatedly transform the ORIGINAL noise z.
        #
        # Each layer performs:
        # a_n = A_n z + c_n
        #
        # Shape:
        # [batch, z_dim] -> [batch, hidden_dim]
        #
        # ModuleList registers all these layers as trainable parameters.
        self.inject_layers = nn.ModuleList([
            nn.Linear(
                in_features=z_dim,
                out_features=hidden_dim,
                bias=True
            )
            for _ in range(num_orders - 1)
        ])

        # These layers transform the current hidden representation.
        #
        # Each performs:
        # h = S_n h + b_n
        #
        # Shape:
        # [batch, hidden_dim] -> [batch, hidden_dim]
        self.state_layers = nn.ModuleList([
            nn.Linear(
                in_features=hidden_dim,
                out_features=hidden_dim,
                bias=True
            )
            for _ in range(num_orders - 1)
        ])

        # Final transformation:
        #
        # [batch, hidden_dim] -> [batch, 2]
        #
        # The two outputs are the generated x and y coordinates.
        self.output_layer = nn.Linear(
            in_features=hidden_dim,
            out_features=2,
            bias=True
        )

        # Tanh restricts generated coordinates to approximately [-1, 1].
        #
        # This can be useful for a unit-circle dataset.
        # However, Tanh means the complete generator is no longer
        # strictly one global polynomial.
        self.output_activation = (
            nn.Tanh()
            if bound_output
            else nn.Identity()
        )

    def forward(self, z):
        """
        z shape: [batch_size, z_dim]
        """

        if z.ndim != 2:
            raise ValueError(
                f"Expected z to have shape [batch, {self.z_dim}], "
                f"but received {tuple(z.shape)}"
            )

        if z.size(1) != self.z_dim:
            raise ValueError(
                f"Expected each noise vector to contain {self.z_dim} "
                f"values, but received {z.size(1)}"
            )

        # ---------------------------------------------------------
        # First-order representation
        # ---------------------------------------------------------
        #
        # h = W_1 z + b_1
        #
        # Shape:
        # [batch, z_dim] -> [batch, hidden_dim]
        h = self.first_layer(z)
        h = self.activation(h)

        # ---------------------------------------------------------
        # Higher-order polynomial interactions
        # ---------------------------------------------------------
        for inject_layer, state_layer in zip(
            self.inject_layers,
            self.state_layers
        ):
            # Transform the original noise vector again.
            #
            # a = A_n z + c_n
            #
            # Shape:
            # [batch, z_dim] -> [batch, hidden_dim]
            a = inject_layer(z)
            a = self.activation(a)

            # Element-wise/Hadamard multiplication.
            #
            # h and a both have shape:
            # [batch, hidden_dim]
            #
            # Therefore:
            # h[i, j] = h[i, j] * a[i, j]
            #
            # This multiplication creates higher-order terms in z.
            h = h * a

            # Apply the S[n] transformation.
            #
            # h = S_n h + b_n
            #
            # For point data, S[n] is an nn.Linear layer.
            # For DCGAN image data, S[n] was a ConvTranspose2d block.
            h = state_layer(h)
            h = self.activation(h)

        # Convert the final hidden representation into [x, y].
        point = self.output_layer(h)

        # Optional restriction to [-1, 1].
        point = self.output_activation(point)

        return point


class Discriminator2D(nn.Module):
    """
    Discriminator for 2D point data.

    Input:
        points with shape [batch_size, 2]

    Output:
        logits with shape [batch_size, 1]

    One score is produced for every [x, y] point.
    """

    def __init__(
        self,
        input_dim=2,
        hidden_dim=128,
        use_spectral_norm=False
    ):
        super().__init__()

        # input_dim=2 because every sample contains:
        # [x_coordinate, y_coordinate]
        self.input_dim = input_dim

        # hidden_dim=128 controls discriminator capacity.
        # It does not mean the circle has 128 raw features.
        self.hidden_dim = hidden_dim

        # Helper for optionally adding spectral normalization.
        #
        # Spectral normalization can stabilize GAN discriminator
        # training by controlling the size of the layer weights.
        def make_linear(in_features, out_features):
            layer = nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=True
            )

            if use_spectral_norm:
                layer = spectral_norm(layer)

            return layer

        self.main = nn.Sequential(
            # First discriminator transformation:
            #
            # [batch, 2] -> [batch, 128]
            make_linear(input_dim, hidden_dim),

            # LeakyReLU keeps a small gradient for negative values.
            # 0.2 is a common GAN discriminator choice.
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=False
            ),

            # Second hidden transformation:
            #
            # [batch, 128] -> [batch, 128]
            make_linear(hidden_dim, hidden_dim),

            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=False
            ),

            # Final real/fake score:
            #
            # [batch, 128] -> [batch, 1]
            make_linear(hidden_dim, 1)
        )

    def forward(self, points):
        """
        points shape: [batch_size, 2]
        """

        if points.ndim != 2:
            raise ValueError(
                f"Expected points to have shape [batch, 2], "
                f"but received {tuple(points.shape)}"
            )

        if points.size(1) != self.input_dim:
            raise ValueError(
                f"Expected every point to contain {self.input_dim} "
                f"coordinates, but received {points.size(1)}"
            )

        # No Sigmoid here.
        #
        # The output is a raw logit. This is preferred when using:
        #
        # nn.BCEWithLogitsLoss()
        logits = self.main(points)

        return logits