
import functools

import tiktoken
import numpy as np
import torch

PAD_LABEL = -100

@functools.lru_cache(maxsize=1)
def load_bpe(name: str = "o200k_base") -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)
    
    
class LLmEncoder:
    
    def __init__(self, max_length: int = 512, encoding:str = "o200k_base"):
        self.bpe = load_bpe(encoding)
        self.max_length = max_length
        self.end_id = self.bpe.eot_token
        self.pad_id = self.end_id
        self.vocab_size = self.bpe.n_vocab
        
    def corpus_to_ids(self, text:str) -> list[int]:
        ids = self.bpe.encode(text)
        ids.append(self.end_id)
        return ids[: self.max_length]
    
    def ids_to_text(self, ids) -> str:
        return self.bpe.decode([int(i) for i in np.asarray(ids).ravel()])
    
    def with_batch(self, texts: list[str]):
        rows = [self.corpus_to_ids(t) for t in texts]
        length = max(len(r) for r in rows)
        
        ids = np.full((len(rows), length), self.pad_id, dtype=np.int64)
        keep = np.zeros((len(rows), length), dtype=bool)
        
        for i, row in enumerate(rows):
            ids[i, : len(row)] = row
            keep[i, : len(row)] = True

        return ids, keep
    
    def learnable(self, inputs, labels):
        corpus = self.ids_to_text(inputs)
        learn = self.ids_to_text(labels)
        
        return {
            "corpus" : corpus,
            "learn" : learn
        }
    
    def llm_encode(self, texts: list[str], device=None, causal: bool = True):
        
        ids, keep = self.with_batch(texts)
        inputs, labels = self.to_pairs(ids, keep)
        keep = keep[:, :-1]
        
        inputs = torch.from_numpy(np.ascontiguousarray(inputs))
        labels = torch.from_numpy(np.ascontiguousarray(labels))
        keep = torch.from_numpy(np.ascontiguousarray(keep))
        
        if device is not None:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            keep = keep.to(device, non_blocking=True)
            
        B, L = keep.shape
        dev = keep.device
        
        attn_mask = keep[:, None, None, :].expand(B, 1, L, L)
        
        if causal:
            attn_mask = attn_mask & torch.ones(
                L, L, dtype=torch.bool, device=dev
            ).tril()

        attn_mask = attn_mask | torch.eye(L, dtype=torch.bool, device=dev)
        
        return {
            "inputs": inputs,
            "labels": labels,
            "keep": keep,
            "pad_mask": ~keep,
            "attn_mask": attn_mask,
        }
        
    
    def to_pairs(self, ids: np.ndarray, keep: np.ndarray):
        inputs = ids[:, :-1]
        labels = ids[:, 1:].copy()
        labels[~keep[:, 1:]] = PAD_LABEL
        return inputs, labels
      