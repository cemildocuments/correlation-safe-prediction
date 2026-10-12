# Dataset attribution

The source archives retain their UCI content. Each dataset is distributed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). The MIT license applies to the experiment code. Dataset attribution and source terms apply to the data.

| Archive | Dataset and creators | Year | Source DOI |
|---|---|---|---|
| `letter.zip` | Letter Recognition; Slate, David | 1991 | [10.24432/C5ZP40](https://doi.org/10.24432/C5ZP40) |
| `pendigits.zip` | Pen-Based Recognition of Handwritten Digits; Alpaydin, E. and Alimoglu, Fevzi | 1996 | [10.24432/C5MG6K](https://doi.org/10.24432/C5MG6K) |
| `drybean.zip` | Dry Bean; repository citation without a named creator | 2020 | [10.24432/C50S4B](https://doi.org/10.24432/C50S4B) |
| `spambase.zip` | Spambase; Hopkins, Mark and Reeber, Erik and Forman, George and Suermondt, Jaap | 1999 | [10.24432/C53G6X](https://doi.org/10.24432/C53G6X) |
| `satellite.zip` | Statlog (Landsat Satellite); Ashwin Srinivasan | 1993 | [10.24432/C55887](https://doi.org/10.24432/C55887) |
| `waveform.zip` | Waveform Database Generator (Version 1); Breiman, L. and Stone, C. J. | 1984 | [10.24432/C5CS3C](https://doi.org/10.24432/C5CS3C) |

Preprocessing removes exact duplicate feature-label rows before splitting, standardizes features using training observations, and introduces controlled block outages. No source archive is altered. Waveform is a generated-data benchmark. The supplied Pen Digits and Satellite partitions are used only in the source-shift analysis.
