# Interior Design Visual Retrieval System.

An image retrieval system that finds visually similar interior design images using three approaches: CLIP, ResNet50 and RGB color histograms

## Overview of the project.

On uploading an image of a room or furniture, and selecting a tool among the three:CLIP, ResNet-50 and RGB Colour histogram, the application retrieves visually similar images from a dataset of interior design pictures. The project compares learned image representations and evaluates the retrieval quality of the three models against human judgment.

## Research question: How do pre-trained CLIP, ResNet50 and RGB color histograms compare with one another and retrieve images closest to human perception.

## Features:

- Upload an image through a Streamlit web app interface will retrieve visually similar images using CLIP, ResNet50 or RGB color histogram.

- Choose how many results from 3 to 10 you would like to display.

- Evaluate retrieval quality using human relevance labels, Precision@5, confidence intervals, and paired statistical tests.

## Methods:

Method |Representation | Expected strength
|:---|:---:|---:|
Clip | 512-dimensional image embeddings| semantic and visual similarity 
ResNet | 2048-dimensional feature embeddings| learned visual patterns and object features 
RGB color histogram | 96-dimensional color representation| dominant color distribution

## Data set 
The dataset contains four one three nine metadata rows covering four room categories: 
- bathroom, bedroom, kitchen, living room. 
It includes five style labels: 
- boho, industrial, minimalist, modern, and scandinavian. 
An exact file duplicate audit found 3789 unique file contents among the 4139 rows, with 288 duplicate groups. 
The live application filters exact duplicates from retrieval results without deleting the original dataset rows or changing the saved embeddings.

## Evaluation 
The human evaluation used 30 queries and 450 candidate results. 
Relevance judgments focused on overall visual and aesthetic similarity. 
A separate Room consistency metric measured whether retrieved images matched the query's drone category.

## Visual precision@5

Method | Precision@5 | 95% confidence interval
|:---|:---:|---:|
CLIP | 87.3% | 79.3%–95.4%
ResNet-50 | 71.3% | 60.9%–81.7%
RGB color histogram | 46.0% | 33.2%–58.8%

## Room consistent precision@5
Method | Precision@5 | 95% confidence interval
|:---|:---:|---:|
CLIP | 66.0% | 54.7%–77.3%
ResNet-50 | 42.7% | 31.1%–54.2%
RGB color histogram | 15.3% | 8.4%–22.3%

Paired with Wilcoxon tests with Holm correction found statistically significant differences between all three pairs of methods in original evaluation (adjusted p-values below 0.01). 

## Interpretation 

CLIP achieved the highest visual precision at the rate five in the evaluated sample, ResNet-50 performed better than the color histogram baseline. 

The color histogram's weaker results illustrate the limitations of using color distribution without semantic image information.

These results describe the original evaluation pipeline. The final application normalizes embeddings or cosine similarity ranking. The reported matrices should not be presented as measurements of that updated ranking pipeline until the evaluation is confirmed to use the same normalization.

## Project structure
```
visual-retrieval-system/
├── app.py
├── data/
│   ├── embeddings/
│   ├── evaluation/
│   ├── processed/
│   └── raw/
├── notebooks/
├── outputs/
├── reports/
├── scripts/
├── src/
│   ├── models/
│   ├── retrieval/
│   └── dataset_loader.py
├── requirements.txt
└── README.md
```
## Setup

Use a Python environment with the packages listed in requirements.txt.

From the project root, install dependencies:

`python -m pip install -r requirements.txt`

If Streamlit or the model dependencies are missing, install them with:

`python -m pip install streamlit torch torchvision open_clip_torch`

Ensure the processed metadata and embedding files exist in the expected data/ directories.

## Run the Application

From the project root:

`python -m streamlit run app.py`

Upload a room or furniture image, select a retrieval method, choose desired number of results, and inspect the returned matches.

## Limitations

- Human judgments of visual similarity can be subjective.

- The dataset contains exact duplicates and some noisy or ambiguous images.

- Aesthetic similarity, object-category similarity, and room consistency are different objectives.

- The evaluation covers a limited sample of queries and does not guarantee performance on all external images.

- Exact-file deduplication does not detect every visually identical image saved with different compression or dimensions.

- The reported evaluation metrics were obtained using the original evaluation pipeline and require a consistency check against the final application's cosine-similarity ranking.

## Future Work

- Evaluate the final application using the same normalization and ranking logic as the evaluation pipeline.

- Expand testing with external Pinterest images and additional object categories.

- Evaluate object matching separately from overall aesthetic similarity.

- Investigate near-duplicate images and improve dataset quality.

- Add automated tests and publish representative retrieval examples.

## Conclusion

This project compares three image-retrieval representations in an interior-design setting. In the original human evaluation, CLIP achieved the highest visual Precision@5, followed by ResNet-50 and RGB color histograms. The comparison demonstrates the value of learned image representations over a simple color-based baseline while highlighting the importance of clear relevance definitions, dataset quality, and reproducible evaluation.
