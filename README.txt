README file by Yuchen Wang(yw3890)

Whole Logic:

Iris Localization: 
1. We first identifies the pupil's approximate location in an eye image by finding the darkest region, presumed to be the pupil, after applying a bilateral filter for smoothing.
2. Then, for a reasonable initial guess of the pupil's location, we refines the center coordinates using the Hough Circle Transform on a blurred region around the guess.
3. If the initial guess is out of bounds, we uses the Hough Circle Transform on the whole eye image to detect the pupil.
4. Next, after pupil detection, the iris edges are detected using the Canny edge detector on a median-blurred image, avoiding the pupil area.
5. Finally, the Hough Circle Transform detects the iris boundary, and the result is adjusted if it deviates significantly from the pupil center.

IrisNormalization:
1. For Normalization, we maps the iris region from the original eye image to a dimensionally consistent, pseudo-cylindrical coordinate system using the pupil and iris boundaries.
2. It interpolates points between the pupil and iris boundaries along radial lines divided into equal angular intervals to populate the normalized iris pattern.
3. After normalization, we inverts the grayscale values to facilitate further processing, like feature extraction in iris recognition systems.

IrisEnhancement:
1. We applied histogram equalization using a disk-shaped structural element to enhance local contrasts in the normalized iris image like in Li ma's paper.
2. The top 48 rows of the matrix are selected by us as the region of interest (ROI) to exclude potential eyelash and eyelid interference, resulting in an ROI with dimensions of 48 by 512.

IrisFeatureExtraction:
1. We first defined functions to create and apply Gabor filters, which are used for texture analysis in images, to regions of an iris image for feature extraction.
2. Gabor filters with predefined spatial constants and frequencies are generated to target specific features in the iris patterns.
3. ROI from an iris image is convolved with these filters, producing filtered images that highlight different aspects of the iris texture.
4. From each filtered image, a feature vector is constructed by calculating the mean and average absolute deviation within non-overlapping blocks of the image.
5. Finally, the resulting vectors from all filters are concatenated to form a final feature vector representing the unique aspects of the iris, which can be used for further analysis such as iris recognition.

IrisMatching:
1. In IrisMatching, we calculated the distances between feature vectors of iris images using different distance metrics and applies circular shifts to account for rotational differences.
2. It evaluates the performance of the matching process by comparing predicted labels against true labels and calculates the rate of correct matches.
3. We applied the dimensionality reduction techniques to reduce the feature space before applying a k-NN classifier to improve the matching process, with the best match rate being determined.
4. We selected a random subset of test samples to evaluated against the training set for validation purposes.
5. Finally, we computed a Receiver Operating Characteristic (ROC) curve by assessing false match and false non-match rates at various threshold levels.

IrisPerformanceEvaluation:
1. we first used function to calculate how often the iris recognition system correctly identifies individuals using different distance measures (L1, L2, cosine) and a reduced feature set through dimensionality reduction.
2. Then we generated Receiver Operating Characteristic curves to evaluate the system's ability to distinguish between genuine matches and impostors by plotting false match rates against false non-match rates across various thresholds.
3. Then, to get the CRR table,we used the original feature set against a dimensionally reduced feature set for different similarity measures.
4. Next, the impact of the dimensionality of feature vectors on the system's correct recognition rate is plotted, showing how the recognition performance changes with varying numbers of features.
5. We tried to write the function to output the table  4, providing false match and non-match rates at different threshold levels.
6. Finally, we wrote the main functions to execute these evaluations, including generating tables and plots for a comprehensive performance assessment of the iris recognition system.

IrisRecoginition:
1. This the main function for our whole model. We did all functions for the rest of six files. This file is for applying those functions and get tables and figures. 
2. We seperated the training and testing set based on requirement. And we read all image from our database for 108 sample with applying functions in localization, nomalization, enhancement, feature extraction, matching and performance evaluation. And we calculated the time it takes to run all loops, which is around 5-10 mins.
3. We printed the table 3, figure 10 and the Roc curve using the function in IrisPermormanceEvaluation.py.

Limitations:
1. In our model, the highest accuracy rate achieved by the algorithm stands at 91.67%, but most of the others are below 90%. 
To improve it, optimizing the parameters within IrisLocalization.py presents an opportunity to improve this rate. Exploring different dimensions for PCA and LDA might also yield enhanced accuracy rates. 

2. For table 4 which requires generating False Match Rate (FMR) and False Non-Match Rate (FNMR) in the given three thresholds, we did not run out the result since the bootstrap takes too long to process the table. 
To improve it, we can try different approaches that do not need bootstrap, or we can try to decrease the time for a bootstrap running, but it might decrease the accuracy a little bit.

3. Our database only contains 108 samples with 7 images for each sample. As a result, the final result for dimensionality might not be accurate.
To improve it, we should have more samples like 200 to further test our model to get the best final result.

Peer Evaluation: 
Sally Wang: She did the Localization and Normalization part code and the comment.
Hongjun Zeng: He did the Enhancement and the feature extraction part code and the comment.
Yuchen Wang: I did the IrisMatching, IrisPerformanceEvaluation, and IrisRecognition part code and the comment.
When we were encountering the error, we did the debugging together and tried to improve our result. And we all participated in searching supplementary materials and tuning models.