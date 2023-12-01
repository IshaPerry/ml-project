import streamlit as st
import pandas as pd
import base64
from PIL import Image

st.title("Classifying Images: Real Vs AI Generated")
st.caption("Members: Sydney Defelice, Claire Matheny, Isha Perry, Ariane Richard, Emily Wu")
st.divider()

st.title("Introduction/Background")
st.markdown("Artificial Intelligence (AI) generated images lead to a mistrust in the reliability of photographs, which oftentimes serve as proof of an event. In fact, research shows that humans misclassify real images from AI-generated ones at a rate of 38.7% (Lu et al., 2023). AI-generated images can have widespread negative impacts by creating false alibis, taking creative reins from real people, amplifying stereotypes, and attempting to falsify historical and current events (Partadiredja et al., 2020; Ragot et al., 2020; Tiku et al., 2023; Verma, 2023). It is important, more than ever, that people are able to quickly tell an AI-generated image from a real one, especially in an age where data reliability and authentication are essential (Bird & Lotfi, 2023). The world is already a chaotic place, rife with political tensions and obsessions over social media, and the last thing the world needs is fake images spreading misinformation.")
st.markdown (" The dataset utilized for this project is obtained from https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/data and contains 120,000 images. Its features consist of 60,000 synthetically generated (fake) images made with Stable Diffusion and 60,000 real images obtained from CIFAR-10 (Krizhevsky & Hilton, 2023; Bird & Lotfi, 2023). 10,000 images of each category make up the testing data, while the remaining are the training data")
st.divider()

st.title("Problem Definition")
st.markdown("AI-generated images allow for the manipulation of reality. The motivation of this project is to identify images as real (non-AI-generated) or fake (AI-generated) to a level of accuracy higher than human classification (61.3%) and ideally higher than existing AI model accuracy (~87%) (Lu et al., 2023).")
st.divider()

st.title("Data Preprocessing")
st.write("Prior to working with our unsupervised and supervised learning models, it was necessary to clean and pre-process the data to run the models successfully. As described above, our dataset was pre-sorted into real and fake images which was further divided into training and testing datasets. Additionally, the dataset indicated that it was pre-cleaned and contained no duplicates. Therefore, we performed the following tasks to pre-process our dataset for CNN:")

tab1, tab2, tab3 = st.tabs(["Resize the Images", "Normalize the Images", "Generate Grayscale and Color (RGB) Versions of the Images"])
tab1.write("To ensure that all the images have the same dimensions, we resized the images to be (32x32x3). This will help ensure that our dataset can properly run in our models. To do so, when importing our images using the image_dataset_from_directory method from the tensorflow keras package, we changed the size of the image by specifying the image_size metric to be (32, 32). Below is a snippet of our code:")
tab1.code("tf_train_data = image_dataset_from_directory(directory = train_dir, label_mode = 'binary', image_size = (32, 32), shuffle=True)", language="python")
tab1.write("The image below displays 9 random images from our training data set and how they have been resized. The label above each image indicates their predetermined label.")
image = Image.open('resizedImages.png')
tab1.image(image, caption='Resized Images')
tab2.write("The images were then normalized to have each pixel value be in the range of [0,1] compared to their original range of [0, 255]. "
           "This step aims to help speed up model learning when running our images through our learning models. "
           "For normalization, we utilized the ImageDataGenerator method from keras.preprocessing.image package. "
           "Below is the code we used on our training and testing dataset to do so:")
tab2.code("TRAIN_IMAGE_GENERATOR = ImageDataGenerator(rescale = 1./255)", language="python")
tab2.code("TEST_IMAGE_GENERATOR = ImageDataGenerator(rescale = 1./255)", language="python")
tab3.write("We are interested in seeing if the accuracy of our models changes if the image is grayscale or colored. "
           "Hence, we converted all of the images to have a grayscale version that we could then run through our models. "
           "Utilizing the flow_from_directory method from keras.preprocessing package, "
           "we modified the color_mode metric to change the images to grayscale. Below is a snippet of the code:")
with tab3:
      tab3.code("TRAIN_GENERATOR_GRAY = TRAIN_IMAGE_GENERATOR.flow_from_directory(train_dir, target_size = (32, 32), class_mode = 'binary', color_mode = 'grayscale',shuffle = False)", language="python")
      tab3.code("TRAIN_GENERATOR_RGB = TRAIN_IMAGE_GENERATOR.flow_from_directory(train_dir, target_size = (32, 32), class_mode = 'binary', color_mode = 'rgb', shuffle = False)", language="python")
      tab3.write("Based on our results from our methods, we will be able to determine if the color scale of the image needs to "
                 "change in order to be able to accurately distinguish between a real image and an AI-generated image.")
      col1, col2 = st.columns(2)
      image = Image.open('otherBird.png')
      col1.image(image)
      image = Image.open('greenBird.png')
      col2.image(image)
      tab3.write("The two images above show the original, colored image and its grayscale version. Both of these images will go through our models.")

st.divider()

st.title("Methods")
pca, cnn, nb, lr = st.tabs(["Unsupervised Learning: Principal Component Analysis (PCA)", "Supervised Learning: Convolutional Neural Network (CNN)", "Supervised Learning: Naive Bayes Classifier", "Supervised Learning: Logistic Regression"])
pca.write("For our unsupervised learning model, we decided to use Principal Component Analysis (PCA). PCA is used for dimensionality reduction and feature extraction, lending itself well to our input of high-dimensional images. It can also help filter noisy data by emphasizing principal components where data exhibits the most variance. Thus, in addition to analyzing PCA on its own, we can also treat it as a type of data pre-processing to feed into our Convolutional Neural Network. ")
pca.write("We loaded, flattened, centered, and standardized the images and applied PCA to our dataset using the SciKit-Learn library. As part of our PCA implementation, we tested all possible principal components against the cumulative expected variance, iterating from 1 component to 3072 components (the number of original features in the matrix). Lastly, we plotted a graph comparing the number of principal components to the cumulative explained variance for further analysis. ")
pca.write("Below is a snippet of the PCA code. ")
pca.code("n_components = data_matrix.shape[1]\npca = PCA(n_components=n_components)\npca.fit(data_matrix)", language="python")
cnn.write("For our supervised learning model, we decided to use a Convolutional Neural Network (CNN). CNNs are designed for image recognition and recognition-based tasks, making it ideal for our goal of classifying images as either real or AI-generated. This model sequentially uses inputs from the previous layer to learn patterns in data adaptively.")
cnn.write("In order to implement CNN, we leveraged Keras (a library built on top of Tensorflow). Our neural network consists of three convolutional layers with max-pooling layers in between. Max pooling improves computational complexity by reducing spatial dimensions, thus enabling transition invariance between layers. The first convolutional layer uses 32 filters that correspond with the input shape of (32, 32, 1) for grayscale images and (32, 32, 3) for RGB images. Additionally, we use the ReLu activation function to introduce non-linearity in the data. In the next convolutional layer, we increase the number of filters to 64 in order to learn even more complex patterns. The last convolutional layer applies 64 filters and ReLu activation another time. The sequence of each our layers is shown below:")
image = Image.open('flowchart_updated.jpeg')
cnn.image(image)
cnn.write("At first, we ran the model using 5 epochs. However, we noticed that the accuracy of testing data for both grayscale and RGB images was higher than the accuracy of training data. This result goes against our intuition that, initially, training should be more accurate than testing. Additionally, the accuracy of these models fluctuates greatly for all datasets. Thus, we decided to fine-tune our number of epochs to 10. With this adjustment, the accuracy levels for both grayscale and RGB images leveled off acceptably. ")
cnn.write("Along with modifying the number of epochs and running CNN on both grayscale and RGB images, we ran CNN with PCA-processed data as well. This is essential, as reducing the dimensionality and noise in a dataset can create a more generalizable model; this is because a model that is trained on lower-dimensional data can be less likely to overfit. However, pre-processing the CNN data using PCA can have drawbacks, as PCA may not maintain the image’s spatial relationships when it reduces the dimensionality. This can hinder the accuracy of CNN, as CNN is designed to exploit the spatial structure of data. Furthermore, CNN is a non-linear model, and applying PCA as a linear transformation may remove complex non-linear relationships in the data set. ")
cnn.write("Thus, we will compare the accuracy of applying CNN on both the grayscale image and RGB image data sets before and after PCA is performed in preprocessing. ")
nb.write("Given that we are solving a classification problem, we decided to implement Naive Bayes. Naive Bayes is based on the assumption that all features of the input data are probabilistically independent of each other. While this assumption is not always the case in real life, it provides an efficient and simple approach to supervised learning. Since our project deals with classifying images as either real or AI-generated, we safely assume that the predictors of each image (i.e. RGB channels, shape, pixels) used are independent of each other. All in all, Naive Bayes provides a simple yet informative baseline classification.")
nb.write(" In order to implement Naive Bayes, we leveraged GaussianNB from the Scikit-learn library. We used a gaussian distribution since features are continuous and normally distributed. First, we trained the Naive Bayes classifier on RGB and grayscale images separately. To reiterate, our input data was processed by loading, flattening, and reshaping. After training the models, we predicted the classification label of each image. Finally, we used more Scikit metric classes to generate precision, recall, and f1 scores, in addition to a confusion matrix to better understand our results. ")
nb.write(" Below is a snippet of our logistic regression code. ")
nb.code("nb_gray = BernoulliNB()\np_rgb = BernoulliNB()\ny_pred_gray = nb_gray.fit(X_reshaped_gray, y_gray).predict(X_reshaped_gray)\ny_pred_rgb = nb_rgb.fit(X_reshaped_rgb, y_rgb).predict(X_reshaped_rgb)", language="python")
lr.write("To further investigate how additional supervised models might perform when classifying real versus AI generated images, we also decided to implement logistic regression. In general, the discriminatory nature of logistic regression makes it well suited for finding links between features and labels, further making it a popular method for classification problems. However, the performance of this method is often limited when faced with highly non-linear problems and/or intricate relationships within input data. Hence, it's important to complement its usage with feature engineering and tuning for diverse problem sets. Thus, we found that it would be useful to compare different preprocessing techniques (grayscale versus RGB) for this method as well. ")
lr.write("In order to implement logistic regression, we decided to use the Scikit-learn library, which aligns well with our task given that it functions primarily as a binary classifier. In addition, by default the logistic regression class of this library applies regularization, which helps improve numerical stability. This parameter is also adjustable to allow model fine-tuning, and we found that values of 0.01 worked best for both the grayscale data and the RGB data. This class also provides several solvers for optimization, but in both cases, we found that the default solver (the Broyden–Fletcher–Goldfarb–Shanno algorithm) yielded the best results. This quasi-Newtonian solver utilizes ℓ2 (aka ridge) regularization to iteratively approximate the second derivative, and hence descent direction, for unconstrained optimization problems. ")
lr.write("After fine-tuning our regression parameters, we fit our preprocessed data to the model and used built-in Scikit functions to generate various evaluation metrics, such as accuracy. Afterwards, we used more Scikit metric classes to generate precision, recall, and f1 scores, in addition to a confusion matrix to better understand our results.")
lr.write("Below is a snippet of our logistic regression code. ")
lr.write( "logreg = LogisticRegression(C = 0.15)\nresults = logreg.fit(x_train, y_train)\ny_pred = logreg.predict(x_test)\nscore = logreg.score(x_test, y_test)\n print('logistic regression accuracy: ' + str(score))" )



st.divider()
st.title("Results and Discussions:")
pca, cnn, cnnwpca, nb, lr = st.tabs(["Unsupervised Learning: Principal Component Analysis (PCA)", "Supervised Learning: Convolutional Neural Network (CNN)", "Supervised Learning: Convolutional Neural Network with PCA Processed Data", "Supervised Learning: Naive Bayes Classifier", "Supervised Learning: Logistic Regression"])
pca.write("Principal Component Analysis (PCA) is an unsupervised learning method that reduces the number of features (dimensions) in a data set while preserving dataset trends and patterns. The benefits of performing PCA include reducing the complexity of the model and thus reducing the risk of overfitting the data. However, PCA can result in information loss if the cumulative expected variance is too low. A low cumulative expected variance means that the selected components do not relate to the original data’s variability, making the loss of information a negative consequence. ")
pca.write("For our dataset specifically, the trade-off between 95% and 99% cumulative expected variance must be considered. On one hand, choosing the number of components that retain 99% of the cumulative expected variance ensures that the finer details are preserved in the image. This is valuable when distinguishing between AI and real images, as these details are imperceptible to the human eye. Specifically, research shows that humans misclassify real images from AI-generated ones at a rate of 38.7% (Lu et al., 2023).  However, it is ultimately more beneficial to choose the number of components that preserve 95% of the variance in the dataset as image data tends to be noisy and complex; by reducing unnecessary noise in the data, we reduce the risk of overfitting our model. Our model aims to identify AI versus real images with an accuracy of at least 87%. Therefore, an overfitted model is less likely to meet this goal as it will not be as effective as a well-fit model in classifying the testing images. ")
image = Image.open('pcaresults.png')
pca.image(image)
pca.write("The benefits of using 298 components to capture 95% of the cumulative expected variance versus 712 components to capture 99% of the cumulative expected variance is justified by the compression ratio. ")
image = Image.open('pcacoderesults.png')
pca.image(image)
pca.write("The compression ratio shows that each retained principal component represents information from approximately 12 original features. By choosing 298 components over 712 components, we ensure that our model will be computationally efficient with an acceptable level of information loss. ")

cnn.write("For both CNN models (gray and RGB images), we found a decent amount of fluctuation in our accuracy for both training data and testing data over time. However, when running CNN with a greater number of epochs, we saw that our accuracy values leveled out with our training accuracy settling higher than our test data. This result was expected, and due to the fact that our plot lines for train accuracy and test accuracy are relatively close to each other, we can safely assume we have achieved minimal overfitting. We found very similar accuracy levels between our two different models, with an accuracy value of roughly 0.9148 for our grayscale CNN model and 0.9186 for our RGB CNN model. This metric tells us that our models were good at predicting labels of unknown data in comparison to the true labels. The high performance of this CNN is likely due to both our convoluted layering structure (i.e., built-in dimensionality reduction) and the large size of our dataset, which helps us reduce overfitting. ")
image = Image.open('cnnresultgraph.png')
cnn.image(image)
cnn.write("While our CNN does have a high level of accuracy, this metric alone is not always reliable. In the case of an unbalanced data set, accuracy becomes a poor evaluation metric as the model can achieve high accuracy by simply predicting the majority label for every instance without us knowing. While we do already know that our Kaggle dataset is balanced, we still decided to generate precision, recall, and f1 scores (harmonic mean) to verify the conclusions we drew from our accuracy score. As displayed in the classification reports shown below, we found that we also had high results for each additional metric for both the gray and RGB models, giving us a more comprehensive evaluation of their overall performance. Additionally, we found very little differences in f1 scores between our gray and RGB models, leading us to further conclude that there is no significant difference in classifying images with our CNN when they are grayscale as opposed to left in color. ")
image = Image.open('classificationreportgray.png')
cnn.image(image)
image = Image.open('classificationreportrgb.png')
cnn.image(image)
cnn.write("Lastly, we then decided to break down our accuracy metric into a confusion matrix to understand our results better. As we know, accuracy is simply true positive added with true negative divided by the total number of predictions. For our grayscale model, we found a slightly higher true positive value than true negative, meaning that this particular model was best at correctly predicting when inputted images were fake (AI-generated). On the contrary, our RGB model had a slightly higher true negative value than true positive, meaning that this model was best at correctly predicting when inputted images were real (not AI-generated). ")
with cnn:
      col1, col2 = st.columns(2)
      image = Image.open('confusionmatrixgray.png')
      col1.image(image)
      image = Image.open('confusionmatrixrgb.png')
      col2.image(image)

cnnwpca.write("Performing PCA on the gray image dataset and the RGB image dataset showed vastly different results in both the training and testing accuracy. ")
image = Image.open('cnnwpcaresults.png')
cnnwpca.image(image)
cnnwpca.write("The RGB training accuracy is significantly higher than the gray training accuracy, with both training accuracies higher than the validation accuracy. The training accuracy around epoch 4 for both gray and RGB increased rapidly, while the validation accuracy began to level out at epoch 4.  The higher training accuracy indicates that the model may be overfitted; this may be due to the fact that PCA simplified the data by reducing dimensionality, making the CNN model too complex for the data. The number of convolutional and pooling layers in the CNN model for the PCA processed data set was decreased to reduce the degree of overfitting. ")
cnnwpca.write("To evaluate the effectiveness of the models, we generated precision, recall, and f1 scores to compare the models and draw more insightful conclusions. There was a 3% difference in the f1 score between the gray and RGB datasets, indicating that performing PCA on the datasets does impact how well the model classified images when they are grayscale as opposed to left in color. ")
image = Image.open('cnnwpcaclassreport.png')
cnnwpca.image(image)
cnnwpca.write("The confusion matrix below further evaluates our accuracy metric. The RGB model had an accuracy of 89%, while the grayscale model had an accuracy of 86%. This means that CNN performed better when the dimensionality of the dataset was higher. PCA with 298 components reduced the dimensionality of both datasets by 83.87%, and making the images gray resulted in a lower-dimensional 2D array rather than the 3D array needed for RGB images. With a higher dimensionality, our CNN model was able to exploit the spatial relationships more effectively, resulting in a higher accuracy for our RGB model. Additionally, the RGB model had a higher precision, recall, and f1 score than the gray model, further validating the claim that our CNN model performs better when the dimensionality of the dataset is higher. It is important to note that PCA with 298 components was chosen as our unsupervised learning analysis showed that 298 components resulted in 95% cumulative expected variance. ")
cnnwpca.write("Regarding identifying AI-generated versus real images, the RGB model had a higher true negative value, meaning that it is the best at correctly identifying when inputted images are real in cases when PCA is and is not performed on the dataset. Additionally, the RGB model is also better at identifying when inputted images are fake (AI-generated) as it has a higher true positive value as well. ")
with cnnwpca:
      col1, col2 = st.columns(2)
      image = Image.open('confusionmatrixgraycnnwpca.png')
      col1.image(image)
      image = Image.open('confusionmatrixrgbcnnwpca.png')
      col2.image(image)

nb.write( "In order to evaluate the performance of our Naive Bayes model, we plotted a confusion matrix showing the number of true positives, true negatives, false positives, and false negatives of each model. From these values, we were able to computationally derive the accuracy, precision, recall and f1 scores. The model had an overall accuracy of 60.30% for grayscale images, and 60.33% for RGB images. As it can be seen, both models performed relatively poorly at classifying images as real or AI-generated." )
with nb:
      col1, col2 = st.columns(2)
      image = Image.open('nbconfusionmatrixgray.png')
      col1.image(image)
      image = Image.open('nbconfusionmatrixrgb.png')
      col2.image(image)
nb.write( "Although this model did not perform as well as our other methods, it’s not surprising given that Naive Bayes is inherently built on the naive assumption that features of the input data are independent of each other. It’s likely that the features of our input data are correlated, thus causing this assumption to be inaccurate. Additionally since we used a Gaussian distribution, it might be the case that our data doesn’t closely follow a continuous normal distribution. For future purposes, it might be valuable to compare the Naive Bayes model on Bernoulli or Multinomial distributions. In general while the independence of Naive Bayes allows for a computationally fast and simple approach, there is a tradeoff in accuracy when the assumption does not hold.")

lr.write("For our logistic regression model, we yielded very different results between our grayscale and RGB datasets. Our accuracy metric was quite high for our grayscale model, landing at around ~93.97%. Accuracy for the RGB data landed at around ~67.15%. Similarly, our loss (calculated via Scikit’s log loss function) differed substantially between our two color variations. For our grayscale model, loss landed at around ~2.099; in comparison, loss for our RGB data was around ~11.346. Given this difference, we suspect that this particular supervised model was better suited for our data when it was preprocessed to be in grayscale. While there are many reasons why this may be the case, we mostly suspect feature redundancy/noise within our RGB dataset due to the increased number of channels. For this particular classification task, logistic regression might not be able to effectively distinguish between useful color features and irrelevant ones, further leading to reduced performance. Grayscaling images reduces the complexity of the data by converting it to a single channel-- this can simplify the learning process, thus making it easier to identify relevant patterns and features for classification. In addition, ​​grayscale images might offer better class separability compared to color images. This could be due to the nature of the images and/or how color information is distributed across the two classes, causing RGB data to not fit well to the linearly-separable assumption that logistic regression holds to.")
lr.write("Similar to our CNN model, we decided to further investigate our results by leveraging more informative evaluation metrics such as precision, recall, and f1. Additionally, we also generated confusion matrices for both color-variant datasets.")
lr.write("As shown below, our classification reports yielded very consistent values for precision, recall, and f1 scores for both our grayscale data and our RGB data. In addition, these values were also very similar to our accuracy score. While this observation originally made us skeptical of our model implementation, upon further analysis, we concluded that we simply encountered a rare case given that our data is evenly balanced between the two labels.  ")
image = Image.open('grayclassreportnb.png')
lr.image(image)
image = Image.open('rgbclassreportnb.png')
lr.image(image)
lr.write("Assuming that precision, recall, and f1 are calculated as shown below, we can actually solve a system of equations to find that all three values can be equal when false positive (FP) = false negative (FN). While rare, this result is not necessarily impractical given we are using a perfectly balanced dataset. ")
image = Image.open('equation.png')
lr.image(image)
lr.write("However, to verify further, we conducted a few manual calculations to find our total amounts of false positives and false negatives. As shown below, we can see that in both cases, we had nearly identical false positive and false negative values, leading us to verify the conclusion drawn above. ")
with lr:
      col1, col2 = st.columns(2)
      image = Image.open('grayclasstotals.png')
      col1.image(image)
      image = Image.open('rgbclasstotals.png')
      col2.image(image)
lr.write("Our confusion matrices showed a similar symmetric property, further reinforcing our outcome and helping us visualize this special case. These matrices are shown below. Once again, while unlikely, symmetric confusion matrices typically indicate that the model is performing equally well (or equally poorly) on both classes in this particular dataset. While the balanced dataset is a good candidate to explain why this occurred, this could also be due to the features of each image not providing enough discriminatory information between the two classes.")
with lr:
      col1, col2 = st.columns(2)
      image = Image.open('lrconfusiongray.png')
      col1.image(image)
      image = Image.open('rgbconfusionlr.png')
      col2.image(image)

st.divider()
st.title("Comparison of Methods: ")
st.write("After analyzing each of the four methods we utilized to classify our dataset, namely Convolutional Neural Network, Convolution Neural Network with Principal Component Analysis Processed Data, Naive Bayes Classifier, and Logistic regression, we looked more closely at their associated metrics to identify the best method to utilize for our purpose as well as whether or not the images should be grayscale or RGB. We decided to utilize these methods to address our goal as they have been previously used for other image classification purposes as described in literature. The tables below summarize the results with the first table comparing the methods for the grayscale images and the second table comparing the methods for the RGB images.")
image = Image.open('grayscalemethods.png')
st.image(image)
image = Image.open('rgbmethods.png')
st.image(image)
st.write("At the start, we were interested in exploring whether or not dimensionality reduction was required for successful classification of real vs AI-generated images. To do so, we ran a convolutional neural network on the original, non-modified images and then ran a convolutional neural network on the PCA-processed images. Regarding performing PCA on the dataset prior to inputting it into a CNN model, it was found that CNN performed better when the data preprocessing did not include PCA. One reason for this result is that PCA results in the loss of information as it projects data onto a lower-dimensional subspace. If the information lost includes essential patterns and relationships, then CNN can struggle to learn meaningful relationships in the dataset and accurately identify if the images are real or AI-generated. Another reason that performing PCA on the dataset reduced the accuracy of CNN is that PCA is a linear transformation. CNN models are designed to identify non-linear relationships in data. Thus, performing a linear transformation on the dataset may remove the non-linear relationships and reduce the complexity of the dataset making it difficult for a CNN to automatically learn hierarchical and non-linear relationships. ")
st.write("Furthermore, it is important to analyze how performing PCA on the dataset inputted into CNN can make the CNN model overfit. We chose to perform PCA using 298 components to preserve 95% of the variance in the dataset, rather than performing PCA using 712 components to preserve 99% of the expected variance in the dataset. This was done to reduce the risk of overfitting, as retaining more variance in a smaller dataset may result in a model performing well on training data and not generalizing well on validation data. However, even with 298 components, the CNN model showed signs of overfitting the data. This means that the reduced dataset from PCA was either too small or not representative of the original data, resulting in the CNN model learning the noise in the training data. Thus, a more generalizable CNN model was created when PCA was not performed on the dataset. ")
st.write("Lastly, the performance of the CNN model on the PCA reduced gray and RGB dataset highlighted the fact that CNN performed better when there was more complexity in the data. For all metrics (precision, accuracy, recall, and f1-score), CNN performed better on the RGB dataset. This means that the spatial information encoded in the different color channels was essential for identifying whether an image was real or AI-generated when PCA-induced information loss occurred. Without PCA, the CNN model on the gray image dataset and RGB image dataset performed almost identically. However, performing PCA and converting the images to gray-scale resulted in the loss of essential spatial information that the CNN model needed to produce accurate results. ")
st.write("Another aspect we were interested in exploring is whether or not the images should be grayscale or colored when determining if an image is real or AI-generated/fake. Looking at the metrics listed in the tables above, each respective method had similar accuracy, precision, recall, and f1-score values for both versions of an image except logistic regression. For logistic regression, this supervised method performed significantly better for classifying grayscale images compared to RGB images as its accuracy was 26.82% higher and its precision, recall, and f1-score was 27% higher for grayscale than colored version. Thus, if one has a dataset of grayscale images, they should ideally use logistic regression as its accuracy, precision, recall, and f1-score metrics were the highest overall. As previously described in the Supervised Learning: Logistic Regression section above, the reason for this drastic difference between grayscale and RGB images may be due to the fact that feature redundancy/noise in the RGB version of the data is more prevalent due to an increase in the number of channels from 1 for grayscale images to 3 for RGB images. ")
st.write("Yet, in general, if one wants to be able to utilize a single method for any type of image, whether or not that image is grayscale or RGB, a CNN would be the best option. Although logistic regression has higher metrics for grayscale images compared to CNN, it does significantly worse in classifying RGB images. Therefore, CNN would be the optimal choice for classifying if images are real or AI-generated/fake. CNN is ideal as it contains multiple convolutional layers that reduce the high dimensionality of images without losing any vital information. This aspect allows a CNN to be a generalizable model that can classify unseen images at a rather high accuracy. On the other hand, Naive Bayes performed the worst for image classification and therefore, should not be used. This could be due to the fact that although Naive Bayes can train fast, it has difficulty in handling unknown features. Therefore, based on the metrics displayed above, CNN is the best choice for classifying images as real or AI-generated as it has the capabilities to easily extract features from images and recognize the patterns that exist within them compared to naive bayes classifier and logistic regression. ")

st.divider()
st.title("Conclusion")
st.write("Our goal was to identify the optimal machine learning technique that can easily and successfully classify if an image is real or AI-generated at a level of accuracy higher than human classification (61.3%) and ideally higher than current AI model accuracy (~87%). In a world where AI is becoming more prevalent, this is more important than ever as false images can alter one’s perception of reality and historical events, amplify stereotypes, and impede on an individual's creativity. Identifying a technique that can classify the integrity of images is essential as humans misclassify real images from AI-generated ones 38.7% of the time (Lu et al., 2023). Thus, throughout the project, we explored different data pre-processing techniques as well as various unsupervised and supervised learning methods to achieve the best results.")
st.write("For this project, we desired to explore various techniques to determine if one yielded better results than the other. The four main methods we tested were Convolutional Neural Networks, Convolutional Neural Networks with PCA processed data, Naive Bayes, and Logistic Regression. We first started by determining if there was a need to pre-process the data via PCA prior to running it into CNN. However, based on our findings described above, this additional data pre-processing step is not required as CNN’s accuracy with the non-PCA processed images is about 3% higher than CNN’s accuracy with PCA-processed images. Simultaneously, we analyzed the differences in model accuracy, precision, recall, and f1-score if the images were grayscale or RGB. With the exception of logistic regression, all methods had similar metrics for both versions of the image. We identified that the reason for the differences in the logistic regression metrics was due to higher feature redundancy/noise in the RGB images compared to the grayscale images. Although logistic regression had the highest metrics for grayscale images out of all the methods, CNN is the best option as it is able to classify any type of image as real or AI-generated at ~91% accuracy rate. ")
st.write("In conclusion, we were able to achieve our goal of identifying a machine learning technique that has the capability to identify images as real or fake at a level of accuracy higher than human classification (61.3%) and existing AI model accuracy (~87%). In our case, CNN was the ideal method that satisfied our purpose and had the capability to classify all types of images at a high accuracy rate of ~91%. Utilizing this technique to facilitate in identifying the validity of images circulating in the world and in the media is promising and exploration on how to further refine and perfect the CNN could have significant societal benefits.")
st.write( "Contribution table: ")
image = Image.open('contribution.png')
st.image(image)


st.divider()
st.title("Gantt Chart: ")
st.write("Below is our Gantt Chart depicting the steps we have taken to complete the project thus far and our plans to "
         "inalize it in time for the Final Report due December 5th. Here is a link to our gantt chart: https://drive.google.com/file/d/1RDsISOcSNLwdXwvQJjfrbmRLNtUDY9CT/view?usp=sharing")

with open("ganntchart.pdf","rb") as f:
      base64_pdf = base64.b64encode(f.read()).decode('utf-8')
pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

st.markdown(pdf_display, unsafe_allow_html=True)
st.divider()

st.title("References")
st.markdown("[1] Bird, J. J., & Lotfi, A. (2023, March 24). CIFAKE: Image classification and explainable identification of AI-generated synthetic images. arXiv.org. https://arxiv.org/abs/2303.14126")
st.markdown("[2] Krizhevsky, A. (2009). Learning multiple layers of features from tiny images - semantic scholar. https://www.semanticscholar.org/paper/Learning-Multiple-Layers-of-Features-from-Tiny-Krizhevsky/5d90f06bb70a0a3dced62413346235c02b1aa086")
st.markdown("[3] Lu, Z., Huang, D., Bai, L., Qu, J., Wu, C., Liu, X., & Ouyang, W. (2023, September 22). Seeing is not always believing: Benchmarking human and model perception of AI-generated images. arXiv.org. https://arxiv.org/abs/2304.13023v3")
st.markdown("[4] Partadiredja, R.A., Serrano, C.E., & Ljubenkov, D. (2020). AI or Human: The Socio-ethical Implications of AI-Generated Media Content. 2020 13th CMI Conference on Cybersecurity and Privacy (CMI) - Digital Transformation - Potentials and Challenges. https://doi.org/10.1109/CMI51275.2020.9322673")
st.markdown("[5] Ragot, M., Martin, N., & Cojean, S. (2020) AI-generated vs. Human Artworks. A Perception Bias Towards Artificial Intelligence? CHI EA ‘20. https://doi.org/10.1145/3334480.3382892 ")
st.markdown("[6] Tiku, N., Schaul, K., & Yu Chen, S. (2023, November 1). These fakes images reveal how AI amplifies our worst stereotypes. The Washington Post. https://www.washingtonpost.com/technology/interactive/2023/ai-generated-images-bias-racism-sexism-stereotypes/")
st.markdown("[7] Verma, R. AI-generated images and videos: A game-changer or a threat to authenticity? Business Insider India. https://www.businessinsider.in/tech/news/ai-generated-images-and-videos-a-game-changer-or-a-threat-to-authenticity/articleshow/99560443.cms")

st.divider()
