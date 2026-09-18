import numpy as np
import pandas as pd
import tiktoken
import torch


class SeriesEncoder:

    def __init__(self, series: pd.Series):
        if isinstance(series, pd.DataFrame):
            raise TypeError(
                f"SeriesEncoder expects a pandas Series, got a DataFrame with "
                f"columns {series.columns.tolist()}. This usually means your "
                f"source DataFrame has duplicate column names — check with "
                f"df.columns[df.columns.duplicated()]."
            )

        if series.empty:
            raise ValueError("SeriesEncoder received an empty Series.")

        self.bpe = tiktoken.get_encoding("o200k_base")
        self.end_id = self.bpe.eot_token
        self.PAD_ID = self.bpe.n_vocab  # guaranteed outside real vocab, no collision with end_id
        self.VOCAB_SIZE = self.bpe.n_vocab + 1  # +1 to account for PAD_ID

        # fill NaNs and force everything to string before tokenizing
        self.SERIES = series.fillna("").astype(str)

        # encode once, reuse everywhere (avoids double BPE pass)
        self._encoded = self.SERIES.apply(self.bpe.encode)

        self.MAX_LENGTH = self.max_length()


    def corpus_to_ids(self, ids: list[int]) -> list[int]:
        ids = ids[: self.MAX_LENGTH - 1]
        ids.append(self.end_id)
        return ids


    def ids_to_text(self, ids) -> str:
        ids = [
            int(i)
            for i in np.asarray(ids).ravel()
            if int(i) != self.PAD_ID
        ]
        return self.bpe.decode(ids)


    def max_length(self) -> int:
        max_length = self._encoded.apply(len).max()
        return int(max_length) + 1


    def series_encode(self) -> dict:

        seqs = self._encoded.apply(self.corpus_to_ids).tolist()

        padded = np.full(
            (len(seqs), self.MAX_LENGTH),
            self.PAD_ID,
            dtype=np.int64
        )

        mask = np.zeros(
            (len(seqs), self.MAX_LENGTH),
            dtype=bool
        )

        for r, s in enumerate(seqs):
            s = s[: self.MAX_LENGTH]
            padded[r, : len(s)] = s
            mask[r, : len(s)] = True

        input_ids = torch.from_numpy(padded)
        attention_mask = torch.from_numpy(mask)

        return {
            "padded": padded,
            "mask": mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "vocab_size": self.VOCAB_SIZE,
            "pad_id": self.PAD_ID,
        }