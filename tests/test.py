
import numpy as np

from neural_ed.src.neural_ed.ImageEncoder import ImageProcessor
from neural_ed.src.neural_ed.LLmEncoder import LLmEncoder
from neural_ed.src.neural_ed.Normalized import ScalerFactory
from neural_ed.src.neural_ed.SeriesEncoder import SeriesEncoder
from neural_ed.src.neural_ed.NlpEncoder import NlpEncoder
from neural_ed.UniqueEncoder import OrdinalEncoder
import pandas as pd


# corpus = [
#         "i am fine.",
#         "he is bad."
#     ]

# raw = NlpEncoder(ngram_range=(1, 2), min_df=1)
# X = raw.nlp_encode(corpus)

# # print(X)

# s = pd.Series(["low", "high", "medium", "low", None, "high"], name="risk")
# enc = OrdinalEncoder(s.unique(), dtype="Int64")
# codes = enc.fit_transform(s)

# # print(codes)






# s = pd.Series([10, 20, 30, 40, np.nan, 500], name="income")

# for name in ["minmax", "standard", "robust", "maxabs", "unit"]:
#     sc = ScalerFactory.create(name)
#     out = sc.fit_transform(s)
#     print(f"{name:9s} {np.round(out.values, 3)}  ->  {sc}")         



img = np.random.randint(0, 256, (456, 543, 3), dtype=np.uint8)
proc = ImageProcessor(size=(25, 25))
out = proc(img)

print(out)
print(out.shape)