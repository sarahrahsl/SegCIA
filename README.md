# SegCIA
Segmentation for Cancer Invasion Analysis of nerve and vessel
## Extracting and analyzing 3D histomorphometric. bfeatures related to perineural and lymphovascular invasion in prostate cancer

### 1. Nerve Segmentation (Ground Truth Creation)
- Generate binary nerve masks from IHC channel images using classical computer vision methods.
- Example Notebook: ```Segment_Nerve.ipynb```
- Example input data provided in the folder: ```ihc_example/```

### 2. Chunking data for nnU-Net input
- Divide large 3D volumes into smaller spatial chunks to reduces memory load.
- Example Notebook: ```Chunking_Upenn.ipynb```
### 3. Volume Blending
- Blend overlapping inferred chunks into unified 3D volumes.
- Example Notebook: ```BlendingUpenn.ipynb```
### 4. Feature Extraction
- Extract cancer invasion features relative to segmentation masks.
Feature can be extracted by Level-by-level (slice-wise) analysis, or Chunk-by-chunk analysis. 
- Output: structured CSV files with extracted features
- Example Notebook: ```Extract_feature_CSV.ipynb```

##
Helper functions:
```
utils/
├── data_handling.py
└── feature_extractor.py
```

## Dependencies
Typical dependencies include:
- Python 3.9+
- numpy
- nibabel
- scipy
- pandas
- scikit-image
- opencv-python
- matplotlib
- tqdm