from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class ImageProcessor:
    """
    Raw image (H, W, C) ko PyTorch model-ready tensor (B, C, H, W) banata hai.

    Pipeline:
        load → channel fix → CHW → float [0,1] → resize → normalize → batch dim → device

    Parameters
    ----------
    size : (int, int), default (255, 255)
        Target (height, width).
    mean, std : sequence of float
        Per-channel normalization stats. Default = ImageNet stats.
    out_channels : {1, 3}, default 3
        3 = force RGB, 1 = force grayscale.
    keep_aspect : bool, default False
        True  -> aspect ratio preserve karke letterbox padding karta hai.
        False -> seedha stretch/squash kar deta hai.
    pad_value : float, default 0.0
        Letterbox padding ki value (0-1 scale me, normalize se pehle).
    interpolation : {'bilinear', 'bicubic', 'nearest', 'area'}
    device, dtype : output tensor ka device aur dtype.
    """

    def __init__(
        self,
        size: tuple[int, int] = (255, 255),
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        out_channels: int = 3,
        keep_aspect: bool = False,
        pad_value: float = 0.0,
        interpolation: str = "bilinear",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if out_channels not in (1, 3):
            raise ValueError("out_channels must be 1 or 3")
        if len(size) != 2 or any(s <= 0 for s in size):
            raise ValueError("size must be (height, width) with positive ints")

        self.size = tuple(int(s) for s in size)
        self.out_channels = out_channels
        self.keep_aspect = keep_aspect
        self.pad_value = float(pad_value)
        self.interpolation = interpolation
        self.device = torch.device(device)
        self.dtype = dtype

        # mean/std ko (C, 1, 1) shape me rakho taaki broadcasting seedhi ho
        mean_t = torch.as_tensor(mean, dtype=dtype)
        std_t = torch.as_tensor(std, dtype=dtype)
        if out_channels == 1 and mean_t.numel() == 3:      # RGB stats -> gray
            mean_t, std_t = mean_t.mean().reshape(1), std_t.mean().reshape(1)
        if mean_t.numel() != out_channels:
            raise ValueError(f"mean/std must have {out_channels} values")
        if torch.any(std_t == 0):
            raise ValueError("std me 0 nahi ho sakta (divide by zero)")

        self.mean = mean_t.view(-1, 1, 1).to(self.device)
        self.std = std_t.view(-1, 1, 1).to(self.device)

        # last processed image ka metadata (debug / inverse ke liye)
        self.last_shape_: tuple | None = None
        self.last_scale_: float | None = None
        self.last_pad_: tuple[int, int, int, int] | None = None

    # ------------------------------------------------------------- STEP 1
    def _load(self, image) -> torch.Tensor:
        """Path / np.ndarray / PIL.Image / Tensor → raw torch tensor."""
        if isinstance(image, (str, Path)):
            from PIL import Image                      # lazy import
            image = np.array(Image.open(image))

        if "PIL" in type(image).__module__:
            image = np.array(image)

        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                t = torch.from_numpy(image.copy())
            else:
                t = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
        elif torch.is_tensor(image):
            t = image.detach().clone()
        else:
            raise TypeError(f"Unsupported input type: {type(image).__name__}")

        if t.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D image, got shape {tuple(t.shape)}")
        return t

    # ------------------------------------------------------------- STEP 2
    def _to_chw(self, t: torch.Tensor) -> torch.Tensor:
        """(H,W) ya (H,W,C) ya already (C,H,W) → (C,H,W)."""
        if t.ndim == 2:                                  # grayscale (H, W)
            t = t.unsqueeze(-1)                          # (H, W, 1)

        # HWC vs CHW detect karo: channel axis chhota (<=4) hota hai
        if t.shape[-1] <= 4 and t.shape[0] > 4:
            t = t.permute(2, 0, 1)                       # HWC → CHW  ✅
        elif t.shape[0] > 4:
            raise ValueError(f"Channel layout samajh nahi aaya: {tuple(t.shape)}")

        c = t.shape[0]
        if c == 4:                                       # RGBA → RGB (alpha drop)
            t = t[:3]
            c = 3

        if self.out_channels == 3 and c == 1:            # gray → RGB
            t = t.repeat(3, 1, 1)
        elif self.out_channels == 1 and c == 3:          # RGB → gray (luma)
            w = torch.tensor([0.299, 0.587, 0.114], dtype=t.dtype).view(3, 1, 1)
            t = (t * w).sum(0, keepdim=True)

        if t.shape[0] != self.out_channels:
            raise ValueError(f"Channel fix fail hua: got {t.shape[0]}")
        return t

    # ------------------------------------------------------------- STEP 3
    def _to_float(self, t: torch.Tensor) -> torch.Tensor:
        """uint8 [0,255] → float [0,1]. Float already ho to sirf cast."""
        if t.dtype == torch.uint8:
            return t.to(self.dtype).div_(255.0)
        t = t.to(self.dtype)
        if t.max() > 1.5:                                # 0-255 float aaya tha
            t = t / 255.0
        return t

    # ------------------------------------------------------------- STEP 4
    def _resize(self, t: torch.Tensor) -> torch.Tensor:
        """(C,H,W) → (C, size[0], size[1])."""
        th, tw = self.size
        c, h, w = t.shape
        t = t.unsqueeze(0)                               # interpolate ko 4D chahiye

        align = None if self.interpolation in ("nearest", "area") else False
        anti = self.interpolation in ("bilinear", "bicubic")

        if not self.keep_aspect:
            self.last_scale_, self.last_pad_ = None, (0, 0, 0, 0)
            t = F.interpolate(t, size=(th, tw), mode=self.interpolation,
                              align_corners=align, antialias=anti)
            return t.squeeze(0)

        # ---- letterbox: aspect ratio preserve + padding ----
        scale = min(th / h, tw / w)
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        t = F.interpolate(t, size=(nh, nw), mode=self.interpolation,
                          align_corners=align, antialias=anti).squeeze(0)

        pad_t = (th - nh) // 2
        pad_b = th - nh - pad_t
        pad_l = (tw - nw) // 2
        pad_r = tw - nw - pad_l
        t = F.pad(t, (pad_l, pad_r, pad_t, pad_b), value=self.pad_value)

        self.last_scale_, self.last_pad_ = scale, (pad_t, pad_b, pad_l, pad_r)
        return t

    # ------------------------------------------------------------- STEP 5
    def _normalize(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.mean) / self.std

    # ---------------------------------------------------------- PUBLIC API
    def process(self, image, add_batch: bool = True) -> torch.Tensor:
        """Single image → (1, C, H, W) tensor."""
        raw = self._load(image)
        self.last_shape_ = tuple(raw.shape)

        t = self._to_chw(raw)
        t = self._to_float(t)
        t = t.to(self.device)
        t = self._resize(t)
        t = self._normalize(t)

        if add_batch:
            t = t.unsqueeze(0)                           # (C,H,W) → (1,C,H,W) ✅

        expected = ((1, self.out_channels, *self.size) if add_batch
                    else (self.out_channels, *self.size))
        assert tuple(t.shape) == expected, f"{tuple(t.shape)} != {expected}"
        return t.contiguous()

    def process_batch(self, images: Iterable) -> torch.Tensor:
        """List of images → (N, C, H, W) — alag-alag sizes chalengi."""
        images = list(images)
        if not images:
            raise ValueError("Empty batch")
        return torch.stack([self.process(im, add_batch=False) for im in images])

    def denormalize(self, t: torch.Tensor) -> torch.Tensor:
        """Normalized tensor → [0,1] range wapas (visualization ke liye)."""
        if t.ndim == 4:
            return (t * self.std.unsqueeze(0) + self.mean.unsqueeze(0)).clamp(0, 1)
        return (t * self.std + self.mean).clamp(0, 1)

    def to_tensor_array(self, t: torch.Tensor, keep_normalized: bool = True) -> torch.Tensor:
        """
        CNN-ready channel-first tensor lauta deta hai — (C, H, W) ya batched
        (N, C, H, W). Koi permute/numpy conversion NAHI hoti, isliye shape
        hamesha (..., C, H, W) hi rehti hai, jo CNN input ke liye correct hai.

        Parameters
        ----------
        t : torch.Tensor
            `process()` / `process_batch()` se aaya (C,H,W) ya (N,C,H,W) tensor.
        keep_normalized : bool, default True
            True  -> mean/std normalized values hi rakhta hai (default; CNN
                     training/inference ke liye yahi sahi input hai).
            False -> `denormalize()` karke [0,1] range me le aata hai, lekin
                     channel-first hi rehta hai (numpy/HWC me convert NAHI
                     karta — visualization ke liye nahi, encoding ke liye hai).

        Returns
        -------
        torch.Tensor, same ndim as input, contiguous memory.
        """
        if not torch.is_tensor(t):
            raise TypeError(f"Expected torch.Tensor, got {type(t).__name__}")
        if t.ndim not in (3, 4):
            raise ValueError(f"Expected (C,H,W) or (N,C,H,W), got shape {tuple(t.shape)}")

        out = t if keep_normalized else self.denormalize(t)
        return out.to(self.dtype).contiguous()

    # ------------------------------------------------------------ niceties
    def __call__(self, image, **kw) -> torch.Tensor:
        return self.process(image, **kw)

    def __repr__(self) -> str:
        return (f"ImageProcessor(size={self.size}, out_channels={self.out_channels}, "
                f"keep_aspect={self.keep_aspect}, device='{self.device}')")
        
        


@dataclass
class BatchResult:
    """Processed batch + uske saath ka metadata."""
    tensor: torch.Tensor                       # (N, C, H, W)
    paths: list[str] = field(default_factory=list)
    original_shapes: list[tuple] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)   # (path, error)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return (f"BatchResult(tensor={tuple(self.tensor.shape)}, "
                f"loaded={len(self)}, failed={len(self.failed)})")

    def name_of(self, i: int) -> str:
        return Path(self.paths[i]).name

    def summary(self) -> str:
        lines = [f"✅ Loaded : {len(self)}", f"❌ Failed : {len(self.failed)}",
                 f"📐 Shape  : {tuple(self.tensor.shape)}"]
        for p, e in self.failed[:5]:
            lines.append(f"   ↳ skip {Path(p).name}: {e}")
        if len(self.failed) > 5:
            lines.append(f"   ↳ ...aur {len(self.failed) - 5} aur")
        return "\n".join(lines)


class ImagePathProcessor(ImageProcessor):
    """
    ImageProcessor + smart path handling.

    process_path() accept karta hai:
        • single file   : "img.jpg"
        • folder        : "data/images/"          (recursive optional)
        • glob pattern  : "data/**/*.png"
        • URL           : "https://.../cat.jpg"
        • list of above : ["a.jpg", "folder/", "*.png"]

    Parameters
    ----------
    recursive : bool, default True
        Folder ke andar sub-folders bhi scan kare.
    on_error : {'skip', 'raise'}, default 'skip'
        Corrupt / unreadable file milne par kya kare.
    sort : bool, default True
        Files ko naam se sort kare (deterministic order).
    limit : int | None
        Max kitni images load karni hain (testing ke liye useful).
    """

    SUPPORTED_EXTS = frozenset({
        ".jpg", ".jpeg", ".png", ".bmp", ".webp",
        ".tif", ".tiff", ".ppm", ".pgm", ".gif",
    })

    def __init__(self, *args, recursive: bool = True, on_error: str = "skip",
                 sort: bool = True, limit: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if on_error not in {"skip", "raise"}:
            raise ValueError("on_error must be 'skip' or 'raise'")
        self.recursive = recursive
        self.on_error = on_error
        self.sort = sort
        self.limit = limit

    # ------------------------------------------------------- path discovery
    def _is_image_file(self, p: Path) -> bool:
        return p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS

    def resolve_paths(self, source) -> list[str]:
        """Kisi bhi input ko concrete image-paths ki flat list me badal deta hai."""
        if isinstance(source, (str, Path)):
            sources = [source]
        elif isinstance(source, Sequence):
            sources = list(source)
        else:
            raise TypeError(f"Unsupported source type: {type(source).__name__}")

        found: list[str] = []
        for src in sources:
            s = str(src)

            if s.startswith(("http://", "https://")):          # URL
                found.append(s)
                continue

            p = Path(s)

            if p.is_dir():                                     # folder
                it = p.rglob("*") if self.recursive else p.glob("*")
                found += [str(f) for f in it if self._is_image_file(f)]

            elif p.is_file():                                  # exact file
                if self._is_image_file(p):
                    found.append(str(p))
                elif self.on_error == "raise":
                    raise ValueError(f"Unsupported extension: {p.suffix}")

            elif any(ch in s for ch in "*?["):                 # glob pattern
                matches = _glob.glob(s, recursive=True)
                found += [m for m in matches if self._is_image_file(Path(m))]
                if not matches and self.on_error == "raise":
                    raise FileNotFoundError(f"Glob matched nothing: {s}")

            else:
                if self.on_error == "raise":
                    raise FileNotFoundError(f"Path exists nahi karta: {s}")

        # duplicates hatao, order preserve karo
        found = list(dict.fromkeys(found))
        if self.sort:
            found.sort()
        if self.limit is not None:
            found = found[: self.limit]

        if not found and self.on_error == "raise":
            raise FileNotFoundError(f"Koi valid image nahi mili: {source}")
        return found

    # ------------------------------------------------------------ main API
    def process_path(self, source, return_meta: bool = True):
        """
        Path(s) → (N, C, H, W) tensor.

        return_meta=True  -> BatchResult (tensor + paths + failures)
        return_meta=False -> sirf tensor
        """
        paths = self.resolve_paths(source)
        tensors, ok_paths, shapes, failed = [], [], [], []

        for p in paths:
            try:
                tensors.append(self.process(p, add_batch=False))
                ok_paths.append(p)
                shapes.append(self.last_shape_)
            except Exception as e:                              # noqa: BLE001
                if self.on_error == "raise":
                    raise RuntimeError(f"Failed on '{p}': {e}") from e
                failed.append((p, f"{type(e).__name__}: {e}"))

        if not tensors:
            raise RuntimeError(
                f"Ek bhi image load nahi hui. {len(failed)} failed. "
                f"Pehla error: {failed[0][1] if failed else 'no files found'}"
            )

        batch = torch.stack(tensors)
        if not return_meta:
            return batch
        return BatchResult(batch, ok_paths, shapes, failed)

    # ------------------------------------- memory-safe chunked iteration
    def iter_batches(self, source, batch_size: int = 32) -> Iterator[BatchResult]:
        """Hazaaron images? RAM bachane ke liye chunks me do."""
        paths = self.resolve_paths(source)
        for i in range(0, len(paths), batch_size):
            yield self.process_path(paths[i : i + batch_size])

    # ------------------------------------------------------------ niceties
    def scan(self, source) -> dict:
        """Load kiye bina folder ka quick summary."""
        paths = self.resolve_paths(source)
        by_ext: dict[str, int] = {}
        total_mb = 0.0
        for p in paths:
            if not p.startswith("http"):
                f = Path(p)
                by_ext[f.suffix.lower()] = by_ext.get(f.suffix.lower(), 0) + 1
                total_mb += f.stat().st_size / 1e6
        return {"count": len(paths), "by_extension": by_ext,
                "total_mb": round(total_mb, 2), "first_5": paths[:5]}

    def __call__(self, source, **kw):
        """Smart dispatch: array/PIL → process(), path-ish → process_path()."""
        if isinstance(source, (str, Path)) or (
            isinstance(source, Sequence) and not isinstance(source, (bytes, bytearray))
            and source and isinstance(source[0], (str, Path))
        ):
            return self.process_path(source, **kw)
        return self.process(source, **kw)