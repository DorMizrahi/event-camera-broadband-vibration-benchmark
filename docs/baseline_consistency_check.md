# Baseline consistency

Before reviewer analyses, the original baseline summary script was rerun on the 30 saved per-experiment outputs in a separate directory. Both summary JSON files produced SHA-256 ee39a2315f37e954532198ea1253436f08c6068700d8af83a5561edc14fbfb10. The frozen inputs passed all 122 minimal-release manifest entries.

K=10 was independently reconstructed from the centroids and the unchanged extraction and matching code. Each stationary peak list and every Chirp ridge matched the archived baseline. The 237 matches, 258 reported peaks and 861 references, 0.354519-Hz macro MAE, 0.367494-Hz pooled MAE and median Chirp RMSE of 22.067668 Hz agree. Recomputed ON displacements and full-record x error metrics agree with the frozen output. No saved baseline result was overwritten.
