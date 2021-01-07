[Accuracy for different perplexities of t-SNE. Also shows how our feature embedding method achieved superior resutls using Euclidean and JS distance metrics.](20ng_accuracy.png)

**Figure**: Accuracy for different perplexities of t-SNE using the transformed 20Newsgroups dataset. Also shows how our feature embedding method achieved superior results using Euclidean and JS distance metrics.

**What is feature embedding?**

It is a feature re-organization method that my teammates and I developed to get better performance out of CNN (ConvNet) for non-image data. The high-dimensional data visualization technique (also, dimensionality reduction technique) t-SNE, is at the heart of the feature embedding method. t-SNE is generally used to group similar samples in a dataset. We engineer this method to group similar features. More details of this method can be found in the corresponding ACM paper (link below). 



**Transformation**: Each raw document -> a function using GloVe word embeddings -> Scaled between [0,255] to imitate black and white images -> features reorganized using the Feature Embedding method


**Directions**:
Please go to https://dl.acm.org/doi/10.1145/3299815.3314429 for details of the feature embedding method. Once done, use the files in the following order.

How it works?

1. get a non-image dataset that satisfies the constraints of the above paper
2. generate distance_matrix using js_geodesic.py
3. generate t-SNE mappings for Jensen-Shannon (JS) using the distance matrix, and for Euclidean distance use the training set;
4. find the array representation using generate_tsne_mapped_dataset_ecl.py or generate_tsne_mapped_dataset_js.py
5. use the datasets with CNN; I used PyTorch; You may use TensorFlow. It doesn't matter.

If the performance doesn't improve, use Hungarian method. 

To measure entropy, use the quantize_dataset_singlefile_bucket.py file. Note that entropy can be measured in many ways. Here we have tried to divide the entire range of feature values into buckets, and use the bucket index as a patch label.
