from __future__ import annotations

import torch

from models import GaussianRenderParams


class GaussianRenderer:
    """
    Differentiable renderer for 2D Gaussians.

    When ``use_bbox=True``, each Gaussian is only evaluated on the pixel
    patch within its 3-sigma bounding box.  This gives a large speed-up on
    CPU (~9x for 1000 Gaussians at 128x128) but is slower on CUDA due to
    ``index_put_`` overhead.  The default ``use_bbox="auto"`` selects the
    bbox path for CPU and the full-grid path for CUDA.

    The bbox path also falls back to the full-grid path automatically when
    the largest Gaussian covers the entire image (e.g. early in training
    with random initialisation).
    """

    def __init__(
        self,
        image_size: int,
        bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        use_anisotropic: bool = False,
        use_alpha: bool = False,
        use_bbox: bool | str = "auto",
    ) -> None:
        self.image_size = image_size
        self.bg_color = bg_color
        self.use_anisotropic = use_anisotropic
        self.use_alpha = use_alpha
        self.use_bbox = use_bbox

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def render(self, params: GaussianRenderParams) -> torch.Tensor:
        want_bbox = self.use_bbox
        if want_bbox == "auto":
            want_bbox = not params.centers.is_cuda
        if want_bbox:
            return self._render_bbox(params)
        return self._render_full(params)

    # ------------------------------------------------------------------
    # full-grid renderer (original)
    # ------------------------------------------------------------------

    def _render_full(self, params: GaussianRenderParams) -> torch.Tensor:
        centers = params.centers
        scales = params.scales
        rotations = params.rotations
        alphas = params.alphas
        colors = params.colors

        device = centers.device
        grid = self._build_pixel_grid(device=device)  # [H, W, 2]
        bg_color = torch.tensor(self.bg_color, dtype=centers.dtype, device=device)

        # Compare every Gaussian against every pixel.
        # diff shape: [N, H, W, 2]
        diff = grid.unsqueeze(0) - centers[:, None, None, :]

        if self.use_anisotropic:
            cos_theta = rotations[:, None, None, 0]
            sin_theta = rotations[:, None, None, 1]
            local_x = cos_theta * diff[..., 0] + sin_theta * diff[..., 1]
            local_y = -sin_theta * diff[..., 0] + cos_theta * diff[..., 1]

            sigma_x2 = scales[:, None, None, 0] ** 2
            sigma_y2 = scales[:, None, None, 1] ** 2
            mahalanobis = (local_x * local_x) / sigma_x2 + (local_y * local_y) / sigma_y2
            support_mask = mahalanobis <= 9.0
            base_weights = torch.exp(-0.5 * mahalanobis) * support_mask.to(centers.dtype)
        else:
            dist2 = (diff * diff).sum(dim=-1)  # [N, H, W]
            sigma2 = scales[:, None, None, 0] ** 2
            support_mask = dist2 <= 9.0 * sigma2
            base_weights = torch.exp(-dist2 / (2.0 * sigma2)) * support_mask.to(centers.dtype)

        if self.use_alpha:
            weights = alphas[:, None, None, 0] * base_weights
        else:
            weights = base_weights

        weighted_rgb = torch.einsum("nhw,nc->hwc", weights, colors)  # [H, W, 3]

        image = bg_color.view(1, 1, 3) + weighted_rgb

        return image

    # ------------------------------------------------------------------
    # bounding-box renderer
    # ------------------------------------------------------------------

    def _render_bbox(self, params: GaussianRenderParams) -> torch.Tensor:
        centers = params.centers          # [N, 2]
        scales = params.scales            # [N, 2]
        rotations = params.rotations      # [N, 2]
        alphas = params.alphas            # [N, 1]
        colors = params.colors            # [N, 3]

        device = centers.device
        dtype = centers.dtype
        H = self.image_size
        bg = torch.tensor(self.bg_color, dtype=dtype, device=device)
        image_rgb = torch.zeros(H, H, 3, dtype=dtype, device=device)

        # 3-sigma bounding-box radius in pixel units
        max_scale = scales.max(dim=1).values          # [N]
        radius_px = (3.0 * max_scale * (H - 1))      # [N]

        # If the largest bbox already covers the full image, fall back to
        # the vectorised full-grid path which has lower overhead.
        max_radius = radius_px.max().item()
        if max_radius >= H:
            return self._render_full(params)

        # Pixel-coordinate centres
        cx_px = centers[:, 0] * (H - 1)              # [N]
        cy_px = centers[:, 1] * (H - 1)              # [N]

        # Integer patch half-size per Gaussian
        half = torch.ceil(radius_px).to(torch.int32)  # [N]

        # Patch size (side length) per Gaussian – used to batch by equal size
        patch_side = (2 * half + 1).clamp(max=H)      # [N]

        # Build the full pixel grid once
        grid = self._build_pixel_grid(device=device)   # [H, H, 2]

        # Group Gaussians by patch_side so each batch can be vectorised
        unique_sizes = torch.unique(patch_side)

        for ps in unique_sizes:
            ps_val = ps.item()
            mask = patch_side == ps
            idx = mask.nonzero(as_tuple=True)[0]       # indices into N
            B = idx.shape[0]

            # Integer bbox corners: [B]
            h = half[idx]
            x0 = (cx_px[idx] - h.to(dtype)).floor().to(torch.int64).clamp(0, H - ps_val)
            y0 = (cy_px[idx] - h.to(dtype)).floor().to(torch.int64).clamp(0, H - ps_val)

            # Gather patches from grid: build [B, ps, ps, 2]
            # Use a single offset vector for the patch row/col offsets
            off = torch.arange(ps_val, device=device)  # [ps]

            # Row and column indices for every Gaussian in this batch
            col_idx = x0[:, None] + off[None, :]       # [B, ps]
            row_idx = y0[:, None] + off[None, :]        # [B, ps]
            col_idx = col_idx.clamp(0, H - 1)
            row_idx = row_idx.clamp(0, H - 1)

            # Extract sub-grids: [B, ps, ps, 2]
            # grid[row, col] -> need grid[row_idx[:,r], col_idx[:,c]]
            sub_grid = grid[
                row_idx[:, :, None].expand(B, ps_val, ps_val),
                col_idx[:, None, :].expand(B, ps_val, ps_val),
            ]  # [B, ps, ps, 2]

            # diff from each Gaussian centre
            b_centers = centers[idx]                    # [B, 2]
            diff = sub_grid - b_centers[:, None, None, :]  # [B, ps, ps, 2]

            b_scales = scales[idx]                      # [B, 2]

            if self.use_anisotropic:
                b_rot = rotations[idx]                  # [B, 2]
                cos_t = b_rot[:, None, None, 0]
                sin_t = b_rot[:, None, None, 1]
                local_x = cos_t * diff[..., 0] + sin_t * diff[..., 1]
                local_y = -sin_t * diff[..., 0] + cos_t * diff[..., 1]
                sx2 = b_scales[:, None, None, 0] ** 2
                sy2 = b_scales[:, None, None, 1] ** 2
                maha = (local_x * local_x) / sx2 + (local_y * local_y) / sy2
            else:
                dist2 = (diff * diff).sum(dim=-1)
                sigma2 = b_scales[:, None, None, 0] ** 2
                maha = dist2 / sigma2

            smask = maha <= 9.0
            base_w = torch.exp(-0.5 * maha) * smask.to(dtype)

            if self.use_alpha:
                b_alpha = alphas[idx]                   # [B, 1]
                w = b_alpha[:, None, None, 0] * base_w
            else:
                w = base_w

            # weighted colour per patch pixel: [B, ps, ps, 3]
            b_colors = colors[idx]                      # [B, 3]
            contrib = w.unsqueeze(-1) * b_colors[:, None, None, :]

            # Scatter-add back into image_rgb
            # Expand row/col indices to [B, ps, ps]
            ri = row_idx[:, :, None].expand(B, ps_val, ps_val)
            ci = col_idx[:, None, :].expand(B, ps_val, ps_val)

            # Flatten for index_put_
            ri_flat = ri.reshape(-1)
            ci_flat = ci.reshape(-1)
            contrib_flat = contrib.reshape(-1, 3)

            image_rgb.index_put_(
                (ri_flat, ci_flat),
                contrib_flat,
                accumulate=True,
            )

        image = bg.view(1, 1, 3) + image_rgb
        return image

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_pixel_grid(self, device: torch.device) -> torch.Tensor:
        coords = torch.linspace(0.0, 1.0, self.image_size, device=device)
        ys, xs = torch.meshgrid(coords, coords, indexing="ij")
        return torch.stack([xs, ys], dim=-1)
