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
st.markdown("AI-generated images allow for the manipulation of reality. "
            " The motivation of this project is to identify images as real (non-AI-generated) or fake (AI-generated) to a"
            " level of accuracy higher than human classification (61.3%) and ideally higher than existing AI model accuracy (~87%)"
            " (Lu et al., 2023).")
st.divider()

st.title("Data Preprocessing")
st.write("Prior to working with our unsupervised and supervised learning models, it was necessary to clean and pre-process "
         "the data to run the models successfully. As described above, our dataset was pre-sorted into real and fake images "
         "which was further divided into training and testing datasets. Additionally, the dataset indicated that it was pre-cleaned "
         "and contained no duplicates. Therefore, we performed the following tasks to pre-process our dataset for CNN:")

tab1, tab2, tab3 = st.tabs(["Resize the Images", "Normalize the Images", "Generate Grayscale and Color (RGB) Versions of the Images"])
tab1.write("To ensure that all the images have the same dimensions, we resized the images to be (32x32x3)."
           "This will help ensure that our dataset can properly run in our models. To do so, when importing our"
           "images using the image_dataset_from_directory method from the tensorflow keras package, we changed"
           "the size of the image by specifying the image_size metric to be (32, 32). Below is a snippet of our code:")
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
tab1, tab2 = st.tabs(["Unsupervised Learning: Principal Component Analysis (PCA)", "Supervised Learning: Convolutional Neural Network (CNN)"])
tab1.write("For our unsupervised learning model, we decided to use Principal Component Analysis (PCA). "
           "PCA is used for dimensionality reduction and feature extraction, lending itself well to our input of high-dimensional images. "
           "It can also help filter noisy data by emphasizing principal components where data exhibits the most variance. "
           "Thus, in addition to analyzing PCA on its own, we can also treat it as a type of data pre-processing to feed into "
           "our Convolutional Neural Network. ")
tab1.write("We loaded, flattened, and standardized the images and applied PCA to our dataset using the SciKit-Learn library. "
           "As part of our PCA implementation, we tested all possible principal components against the cumulative expected variance, "
            "iterating from 1 component to 3072 components (the number of original features in the matrix). "
            "Lastly, we plotted a graph comparing the number of principal components to the cumulative explained variance "
            "for further analysis.")
tab1.write("Below is a snippet of the PCA code. ")
tab1.code("n_components = data_matrix.shape[1]\npca = PCA(n_components=n_components)\npca.fit(data_matrix)", language="python")
tab2.write("For our supervised learning model, we decided to use a Convolutional Neural Network (CNN). "
           "CNNs are designed for image recognition and recognition-based tasks, making it ideal for our goal "
           "of classifying images as either real or AI-generated. This model sequentially uses inputs from the previous "
           "layer to learn patterns in data adaptively.")
tab2.write("In order to implement CNN, we leveraged Keras (a library built on top of Tensorflow). "
           "Our neural network consists of three convolutional layers with max-pooling layers in between. "
           "Max pooling improves computational complexity by reducing spatial dimensions, thus enabling transition invariance between layers. "
           "The first convolutional layer uses 32 filters that correspond with the input shape of (32, 32, 1) for grayscale "
           "images and (32, 32, 3) for RGB images. Additionally, we use the ReLu activation function to introduce non-linearity in the data. "
           "In the next convolutional layer, we increase the number of filters to 64 in order to learn even more complex patterns. "
           "The last convolutional layer applies 64 filters and ReLu activation another time. The sequence of each of our layers is shown below:")
image = Image.open('flowchart_updated.jpeg')
tab2.image(image)
tab2.write("At first, we ran the model using 5 epochs. However, we noticed that the accuracy of testing data for both "
           "grayscale and RGB images was higher than the accuracy of training data. This result goes against our intuition that, "
           "initially, training should be more accurate than testing. Additionally, the accuracy of these models fluctuates "
           "greatly for all datasets. Thus, we decided to fine-tune our number of epochs to 10. With this adjustment, "
           "the accuracy levels for both grayscale and RGB images leveled off acceptably.")


st.divider()
st.title("Results and Discussions:")
tab1, tab2 = st.tabs(["Unsupervised Learning: Principal Component Analysis (PCA)", "Supervised Learning: Convolutional Neural Network (CNN)"])
tab1.write("Principal Component Analysis (PCA) is an unsupervised learning method that reduces the number of features (dimensions)"
           " in a data set while preserving dataset trends and patterns. The benefits of performing PCA include reducing the complexity "
           "of the model and thus reducing the risk of overfitting the data. However, PCA can result in information loss if the "
           "cumulative expected variance is too low. A low cumulative expected variance means that the selected components do not "
           "relate to the original data’s variability, making the loss of information a negative consequence. ")
tab1.write("For our dataset specifically, the trade-off between 95% and 99% cumulative expected variance must be considered. "
           "On one hand, choosing the number of components that retain 99% of the cumulative expected variance ensures that the "
           "finer details are preserved in the image. This is valuable when distinguishing between AI and real images, as these "
           "details are imperceptible to the human eye. Specifically, research shows that humans misclassify real images from "
           "AI-generated ones at a rate of 38.7% (Lu et al., 2023).  However, it is ultimately more beneficial to choose the number "
           "of components that preserve 95% of the variance in the dataset as image data tends to be noisy and complex; by reducing "
           "the noise in the data, we reduce the risk of overfitting our model. Our model aims to identify AI versus real images "
           "with an accuracy of at least 87%. Therefore, an overfitted model is less likely to meet this goal as it will not be as "
           "effective as a well-fit model in classifying the testing images. ")
image = Image.open('pcaresults.png')
tab1.image(image)
tab1.write("The benefits of using 254 components to capture 95% of the cumulative expected variance versus 664 components to capture 99% of "
           "the cumulative expected variance is justified by the compression ratio. ")
image = Image.open('pcacoderesults.png')
tab1.image(image)
tab1.write("The compression ratio shows that each retained principal component represents information from approximately 12 original features. "
           "By choosing 254 components over 664 components, we ensure that our model will be computationally efficient with an acceptable level of information loss. ")

tab2.write("For both CNN models (gray and RGB images), we found a decent amount of fluctuation in our accuracy for both training "
           "data and testing data over time. However, when running CNN with a greater number of epochs, we saw that our accuracy "
           "values leveled out with our training accuracy which settled higher than our test data. This result was expected, "
           "and due to the fact that our plot lines for train accuracy and test accuracy are relatively close to each other, "
           "we can safely assume we have achieved minimal overfitting. We found very similar accuracy levels between our two "
           "different models, with an accuracy value of roughly 0.9148 for our grayscale CNN model and 0.9186 for our RGB CNN model. "
           "This metric tells us that our models were good at predicting labels of unknown data in comparison to the true labels. "
           "The high performance of this CNN is likely due to both our convoluted layering structure "
           "(i.e., built-in dimensionality reduction) and the large size of our dataset, which helps us reduce overfitting. ")
image = Image.open('cnnresultgraph.png')
tab2.image(image)
tab2.write("While our CNN does have a high level of accuracy, this metric alone is not always reliable. "
           "In the case of an unbalanced data set, accuracy becomes a poor evaluation metric as the model can achieve "
           "high accuracy by simply predicting the majority label for every instance without us knowing. "
           "While we do already know that our Kaggle dataset is balanced, we still decided to generate precision, recall, "
           "and f1 scores (harmonic mean) to verify the conclusions we drew from our accuracy score. As displayed in the "
           "classification reports shown below, we found that we also had high results for each additional metric for both "
           "the gray and RGB models, giving us a more comprehensive evaluation of their overall performance. "
           "Additionally, we found very little differences in f1 scores between our gray and RGB models, leading us to further "
           "conclude that there is no significant difference in classifying images with our CNN when they are grayscale as "
           "opposed to left in color.")
image = Image.open('classificationreportgray.png')
tab2.image(image)
image = Image.open('classificationreportrgb.png')
tab2.image(image)
tab2.write("Lastly, we then decided to break down our accuracy metric into a confusion matrix to understand our results better. "
           "As we know, accuracy is simply true positive added with true negative divided by the total number of predictions. "
           "For our grayscale model, we found a slightly higher true positive value than true negative, "
           "meaning that this particular model was best at correctly predicting when inputted images were fake (AI-generated). "
           "On the contrary, our RGB model had a slightly higher true negative value than true positive, "
           "meaning that this model was best at correctly predicting when inputted images were real (not AI-generated). ")

with tab2:
      col1, col2 = st.columns(2)
      image = Image.open('confusionmatrixgray.png')
      col1.image(image)
      image = Image.open('confusionmatrixrgb.png')
      col2.image(image)

st.divider()
st.title("Discussion of  PCA and CNN and Future Works:")
st.write("Before our final report, we will also advance our data preprocessing for PCA by normalizing and centering the images. "
         "Currently, the images are flattened for PCA preprocessing. Regarding the relationship between PCA and CNN, PCA is a "
         "dimensionality reduction technique while CNN is a deep-learning technique that is extremely effective in image "
         "classification tasks. To expand on our implementation of CNN, we will perform CNN using the PCA-reduced dataset against "
         "the non-PCA-reduced dataset. This will assess the impact of PCA on the performance of CNN. Additionally, we will explore "
         "the possibility of using a GAN combined with a CNN to improve our model’s accuracy. ")

st.divider()
st.title("Contribution Table:")
image = Image.open('contributionchart.png')
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
