# neural_ed

Lightweight encoders/scalers for turning pandas `Series` and images into
model-ready tensors and arrays — tokenization (LLM + classic NLP), categorical
and unique-value encoding, numeric scaling, and image preprocessing.

## Install

```bash
pip install tiktoken numpy pandas torch scikit-learn joblib pillow
```

## Package layout

```
neural_ed/
├── SeriesEncoder.py     # pandas Series -> padded BPE token tensors
├── LLmEncoder.py         # text list -> causal-LM training batches
├── NlpEncoder.py         # text list -> TF-IDF over BPE tokens
├── OrdinalEncoder.py     # categorical Series -> integer codes
├── UniqueEncoder.py      # unique-valued Series -> integer codes
├── Normalized.py         # numeric Series scalers (MinMax/Standard/Robust/...)
├── ImageEncoder.py       # images/paths -> normalized tensors
└── __init__.py
```

---

## `SeriesEncoder`

Encodes a single `pandas.Series` of text into a padded batch of BPE token ids
(`o200k_base`), ready for `torch`.

```python
from neural_ed import SeriesEncoder

enc = SeriesEncoder(df["comments"])   # raises on empty Series / DataFrame input
out = enc.series_encode()
```

**Constructor**
- `SeriesEncoder(series: pd.Series)` — NaNs are filled with `""` and everything
  is cast to `str`. Raises `TypeError` if given a `DataFrame`, `ValueError` if
  the Series is empty.

**Methods**
- `series_encode() -> dict` — returns:
  - `padded` (`np.ndarray[int64]`, shape `(N, max_len)`)
  - `mask` (`np.ndarray[bool]`, same shape) — `True` where real tokens are
  - `input_ids` / `attention_mask` — the same two arrays as `torch.Tensor`
  - `vocab_size`, `pad_id`
- `corpus_to_ids(ids: list[int]) -> list[int]` — truncates to `MAX_LENGTH - 1`
  and appends the end-of-text token.
- `ids_to_text(ids) -> str` — decodes ids back to text, dropping `PAD_ID`.
- `max_length() -> int` — longest encoded row in the Series, plus 1.

**Attributes:** `MAX_LENGTH`, `PAD_ID`, `VOCAB_SIZE`, `end_id`, `bpe`.

---

## `LLmEncoder`

Turns a list of raw strings into next-token-prediction training batches
(inputs, shifted labels, padding mask, attention mask).

```python
from neural_ed import LLmEncoder

enc = LLmEncoder(max_length=512, encoding="o200k_base")
batch = enc.llm_encode(["hello world", "a second document"], device="cuda")
# batch: inputs, labels, keep, pad_mask, attn_mask
```

**Constructor**
- `LLmEncoder(max_length: int = 512, encoding: str = "o200k_base")`

**Methods**
- `corpus_to_ids(text: str) -> list[int]` — BPE-encodes, appends EOT, truncates
  to `max_length`.
- `with_batch(texts: list[str]) -> (ids, keep)` — right-pads a list of texts
  into a rectangular `np.ndarray` (`ids`) plus a boolean `keep` mask.
- `to_pairs(ids, keep) -> (inputs, labels)` — shifts by one token for causal
  LM training; padding positions in `labels` are set to `PAD_LABEL = -100`.
- `llm_encode(texts: list[str], device=None, causal: bool = True) -> dict` —
  full pipeline. Returns a dict of `torch.Tensor`s:
  `inputs`, `labels`, `keep`, `pad_mask`, `attn_mask` (a 4D boolean mask,
  causal + padding-aware when `causal=True`).
- `ids_to_text(ids) -> str` — decode ids back to text.
- `learnable(inputs, labels) -> dict` — convenience decode of a training pair
  back into `{"corpus": ..., "learn": ...}` strings for inspection.

**Attributes:** `bpe`, `max_length`, `end_id`, `pad_id`, `vocab_size`.

---

## `NlpEncoder`

Classic TF-IDF, but computed over BPE token ids instead of raw words (so
vocabulary/segmentation matches your LLM encoders).

```python
from neural_ed import NlpEncoder

enc = NlpEncoder(ngram_range=(1, 2))
matrix = enc.nlp_encode(["I love NLP", "NLP loves me"])   # dense np.ndarray
enc.top_terms("I love NLP", k=3)
```

**Constructor**
- `NlpEncoder(ngram_range=(1, 2), min_df=1, max_features=None)`

**Methods**
- `fit(docs) -> self` / `transform(docs) -> scipy.sparse matrix`
- `nlp_encode(docs) -> np.ndarray` — fit + transform in one call, returns a
  dense array (note: this always refits on `docs`).
- `feature_names() -> np.ndarray` — the fitted vocabulary (space-joined token
  ids as strings).
- `readable(feature: str) -> str` — decodes a vocabulary entry back to text.
- `top_terms(doc: str, k: int = 5) -> list[tuple[str, float]]` — top-`k`
  human-readable (term, TF-IDF score) pairs for a single document.
- `save(path)` / `load(path)` — persist/restore the underlying
  `TfidfVectorizer` via `joblib`.

Module-level helper: `encode_corpus(docs: list[str]) -> list[str]` — converts
raw text to space-joined `cl100k_base` token ids.

---

## `OrdinalEncoder`

Scikit-learn-style categorical → integer encoder with explicit
unknown/missing handling. Use this for a `Series` with **repeated**
category labels (e.g. `"A"`, `"B"`, `"C"` appearing many times) — each
distinct category always maps to the same code, wherever it appears.

```python
from neural_ed import OrdinalEncoder

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
codes = enc.fit_transform(df["category"])
enc.inverse_transform(codes)
```

**Constructor**
- `OrdinalEncoder(categories="auto", start=0, handle_unknown="error", unknown_value=None, encode_missing_as=np.nan, dtype=np.float64)`
  - `handle_unknown="error"` raises on unseen categories at transform time;
    `"use_encoded_value"` maps them to `unknown_value` instead.

**Methods**
- `fit(series) -> self` — learns categories (sorted, or in given order if
  `categories` is an explicit sequence).
- `transform(series) -> pd.Series` — maps categories to integer codes.
- `inverse_transform(series) -> pd.Series` — maps codes back to categories.
- `fit_transform(series) -> pd.Series` (inherited from `BaseEncoder`).
- `get_params() -> dict`
- `len(encoder)` — number of learned categories.

**Attributes (post-fit):** `categories_`, `mapping_`, `inverse_mapping_`.

---

## `UniqueEncoder`

Scikit-learn-style encoder for a `Series` whose values are expected to be
**unique** (e.g. an ID column, a primary key, a slug) — a 1:1 mapping between
each distinct value and an integer code, with explicit unknown/missing
handling. Unlike `OrdinalEncoder`, this is not meant for repeated category
labels: by default it enforces uniqueness at fit time and raises if it finds
duplicates, since a repeat would mean the column isn't actually unique-valued
as assumed — use `OrdinalEncoder` for that case instead. Codes are assigned
by first-appearance order (there is no natural ordering among unique
identifiers to sort by).

```python
from neural_ed import UniqueEncoder

enc = UniqueEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
codes = enc.fit_transform(df["user_id"])
enc.inverse_transform(codes)
```

**Constructor**
- `UniqueEncoder(start=0, handle_unknown="error", unknown_value=None, encode_missing_as=np.nan, enforce_unique=True, dtype=np.int64)`
  - `handle_unknown="error"` raises on unseen values at transform time;
    `"use_encoded_value"` maps them to `unknown_value` instead.
  - `enforce_unique=True` raises `DuplicateValueError` at fit time if any
    non-null value repeats; set to `False` to silently dedupe instead.

**Methods**
- `fit(series) -> self` — learns each distinct value's code, in order of
  first appearance.
- `transform(series) -> pd.Series` — maps values to integer codes.
- `inverse_transform(series) -> pd.Series` — maps codes back to values.
- `fit_transform(series) -> pd.Series` (inherited from `BaseEncoder`).
- `get_params() -> dict`
- `len(encoder)` — number of learned values.

**Attributes (post-fit):** `categories_`, `mapping_`, `inverse_mapping_`.

Also exported: `BaseEncoder` (abstract base with the `fit`/`transform`/
`inverse_transform`/`fit_transform` contract), `NotFittedError`, and
`DuplicateValueError` (raised when `enforce_unique=True` and the fitted
Series contains repeated values).

---

## `Normalized` — numeric scalers

Template-method scalers over a single `pandas.Series`, all sharing the same
`fit` / `transform` / `inverse_transform` / `fit_transform` interface, NaN
passthrough, optional output clipping, and zero-division protection (falls
back to `1.0` with a `RuntimeWarning` on constant input).

```python
from neural_ed import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, UnitNormScaler, ScalerFactory

scaler = ScalerFactory.create("standard")   # or minmax / zscore / robust / maxabs / unit
scaled = scaler.fit_transform(df["price"])
original = scaler.inverse_transform(scaled)
scaler.params_     # learned stats, e.g. {"mean_": ..., "std_": ...}
```

| Class | Formula | Key options |
|---|---|---|
| `MinMaxScaler` | `(x - min) / (max - min)` rescaled to `feature_range` | `feature_range=(lo, hi)` |
| `StandardScaler` | `(x - mean) / std` | `ddof`, `with_mean`, `with_std` |
| `RobustScaler` | `(x - median) / IQR` | `quantile_range=(q_lo, q_hi)` |
| `MaxAbsScaler` | `x / max(|x|)` | — |
| `UnitNormScaler` | `x / ‖x‖_p` over the whole Series | `norm="l1" \| "l2" \| "max"` |

Common constructor kwargs (all scalers): `with_nan=True` (allow NaNs through),
`clip=False` (clip transformed output to the fitted output's min/max),
`copy_name=True` (keep the Series name on output).

**Common methods:** `fit(series)`, `transform(series)`, `inverse_transform(series)`,
`fit_transform(series)`, `params_` (property, learned stats), `__repr__`.

**`ScalerFactory`**
- `ScalerFactory.create(method: str, **kwargs) -> BaseScaler` — build by name:
  `"minmax"`, `"standard"`/`"zscore"`, `"robust"`, `"maxabs"`, `"unit"`.
- `ScalerFactory.register(name: str, klass: type)` — register a custom
  `BaseScaler` subclass under a new name.

`BaseScaler` is also exported for writing your own scaler (implement
`_learn`, `_apply`, `_revert`).

---

## `ImageEncoder`

### `ImageProcessor`

Turns a single image (path / `np.ndarray` / `PIL.Image` / `torch.Tensor`)
into a normalized, model-ready `(1, C, H, W)` tensor.

Pipeline: `load → channel fix → CHW → float [0,1] → resize → normalize → batch dim → device`.

```python
from neural_ed import ImageProcessor

proc = ImageProcessor(size=(224, 224), out_channels=3, keep_aspect=True)
tensor = proc("cat.jpg")          # __call__ == process()
proc.last_shape_, proc.last_scale_, proc.last_pad_   # debug metadata
```

**Constructor**
- `ImageProcessor(size=(255, 255), mean=IMAGENET_MEAN, std=IMAGENET_STD, out_channels=3, keep_aspect=False, pad_value=0.0, interpolation="bilinear", device="cpu", dtype=torch.float32)`
  - `out_channels`: `3` forces RGB (grayscale is repeated), `1` forces
    grayscale (RGB is luma-converted). RGBA drops the alpha channel.
  - `keep_aspect=True` letterbox-pads instead of stretching.

**Methods**
- `process(image, add_batch: bool = True) -> torch.Tensor` — full single-image
  pipeline; returns `(1, C, H, W)` (or `(C, H, W)` if `add_batch=False`).
- `process_batch(images: Iterable) -> torch.Tensor` — stacks `process` over
  multiple in-memory images into `(N, C, H, W)`.
- `denormalize(t: torch.Tensor) -> torch.Tensor` — reverses the mean/std
  normalization, clamped back to `[0, 1]`. Accepts either `(C, H, W)` or a
  batched `(N, C, H, W)` tensor. Useful before visualizing a processed image.
- `to_numpy_image(t: torch.Tensor) -> np.ndarray` — denormalizes, scales to
  `[0, 255]`, and permutes to `uint8` `(H, W, C)` — ready for `matplotlib`,
  `cv2`, or `PIL`. Accepts `(C, H, W)` or `(1, C, H, W)` (batch dim is
  dropped); a single grayscale channel is squeezed out.
- `__call__(image, **kw)` — alias for `process`.

**Attributes:** `size`, `mean`, `std`, `out_channels`, `keep_aspect`, `device`,
`dtype`; post-call debug info: `last_shape_`, `last_scale_`, `last_pad_`.

### `ImagePathProcessor(ImageProcessor)`

Adds smart path/folder/glob/URL resolution and batch loading on top of
`ImageProcessor`.

```python
from neural_ed import ImagePathProcessor

proc = ImagePathProcessor(size=(224, 224), on_error="skip", limit=1000)
result = proc.process_path("data/images/")     # folder, glob, URL, or list of these
print(result.summary())
```

**Constructor (extra kwargs)**
- `recursive=True` — recurse into sub-folders.
- `on_error="skip" | "raise"` — behavior on unreadable/unsupported files.
- `sort=True` — deterministic (sorted) file order.
- `limit: int | None` — cap on number of images resolved.
- Supported extensions: `.jpg .jpeg .png .bmp .webp .tif .tiff .ppm .pgm .gif`.

**Methods**
- `resolve_paths(source) -> list[str]` — normalizes a file / folder / glob /
  URL / list of these into a flat, de-duplicated list of paths.
- `process_path(source, return_meta: bool = True) -> BatchResult | torch.Tensor` —
  loads and stacks every resolved image into `(N, C, H, W)`. With
  `return_meta=True` (default), returns a `BatchResult`.
- `iter_batches(source, batch_size: int = 32) -> Iterator[BatchResult]` —
  memory-safe chunked loading for large sources.
- `scan(source) -> dict` — quick summary without loading:
  `{"count", "by_extension", "total_mb", "first_5"}`.
- `__call__(source, **kw)` — dispatches to `process_path` for path-like input,
  or `process` for an already-loaded image.

**`BatchResult`** (returned by `process_path` / `iter_batches`)
- Fields: `tensor` (`(N, C, H, W)`), `paths`, `original_shapes`, `failed`
  (list of `(path, error_message)`).
- `len(result)` — number of successfully loaded images.
- `name_of(i) -> str` — filename for index `i`.
- `summary() -> str` — human-readable load/failure/shape report.

---

## Notes

- All scalers, `OrdinalEncoder`, `UniqueEncoder`, and `SeriesEncoder`/
  `LLmEncoder` operate on a single `pandas.Series` (or list of strings) at a
  time — encode column-by-column / field-by-field for a full DataFrame.
- Use `OrdinalEncoder` for repeated category labels and `UniqueEncoder` for
  columns where every value should be distinct (e.g. an ID column) — picking
  the wrong one either loses the categorical grouping or raises a
  `DuplicateValueError`.
- `LLmEncoder` and `SeriesEncoder` both use `tiktoken`; make sure the intended
  `encoding` (`o200k_base` by default) matches the model you're training or
  targeting.
- `PAD_LABEL = -100` (in `LLmEncoder.py`) matches PyTorch's
  `CrossEntropyLoss(ignore_index=-100)` default, so `labels` can be passed
  straight to a standard LM loss without extra configuration.
- Every class above is importable straight from `neural_ed` (see the
  package's `__init__.py`); `__all__` also exposes the underlying method
  names (`fit_transform`, `process_path`, `to_numpy_image`, `summary`, etc.)
  for tooling that introspects the package.