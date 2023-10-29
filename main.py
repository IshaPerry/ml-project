import streamlit as st
import pandas as pd
import base64
from PIL import Image

st.title("Classifying Images: Real Vs AI Generated")
st.caption("Members: Sydney Defelice, Claire Matheny, Isha Perry, Ariane Richard, Emily Wu")
st.divider()

st.title("Introduction/Background")
st.markdown("Artificial Intelligence (AI) generated images lead to a mistrust in the reliability of photographs, which serves as proof of an event. In fact, research shows that humans misclassify real images from AI-generated photos at a rate of 38.7% (Lu et al., 2023). AI-generated images can have a widespread negative impact from creating false alibis, winning art competitions, and generating false historical and current events images. The dataset utilized for this project is obtained from Kaggle.com and contains 120,000 images. Its features consist of 60,000 synthetically-generated images made with Stable Diffusion and 60,000 real images obtained from CIFAR-10. 10,000 images from each feature makes up the testing data while the remaining are the training data.")
st.divider()

st.title("Problem Definition")
st.markdown("AI-generated images allow for the manipulation of reality. The motivation of this project is to identify images as real (non AI-generated) or fake (AI-generated) to a level of accuracy higher than human classification (61.3%) and ideally higher than existing AI model accuracy (~87%).")
st.divider()

st.title("Methods")
st.markdown("To classify images into our two categories (real or fake), we will be utilizing supervised and unsupervised learning techniques to cross analyze detection accuracy. For supervised learning, we will develop a convoluted neural network (CNN). Using CNN, we can leverage image matrix inputs to extract features that get progressively more accurate with each layer. For unsupervised learning, we will use K-Means Clustering (via scikit-learn) to create clusters of images that share characteristics with those within their cluster and are dissimilar to those outside of their cluster. In both cases, several image processing techniques are required to assign weights to different aspects of the image that are then propagated throughout the code. For image processing, we will be using various Python libraries such as OpenCV, Scikit-image, Python Image Library (PIL), NumPy, and Mahotas.")
st.divider()


st.title("Potential Results/Discussion")
st.markdown("By comparing multiple models of classification, we will be able to determine the best model. Because of the previously stated 61.3% accuracy of humans when classifying real vs AI images, we can define success as any model classifying images with an accuracy above that value. Knowing that this is a low number, we have also found evidence that most modern classification AI algorithms can reach accuracy levels of ~87% (Lu et al., 2023). Thus, we will be aiming for this level of accuracy as well. We also want to assess precision (rate of false positives), recall (rate of false negatives), and f1 (a metric to help combine precision and recall). To help extract values for our success criteria, we will use the sklearn.metrics library and the testing set from our data set to aid in calculating these quantitative metrics and assessing the overall accuracy of our models.")


st.title("Proposed Timeline")

with open("ganntchart.pdf","rb") as f:
      base64_pdf = base64.b64encode(f.read()).decode('utf-8')
pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

st.markdown(pdf_display, unsafe_allow_html=True)
st.divider()

st.title("Contribution Chart")
image = Image.open('contributionchart.png')
st.image(image, caption='Contribution Chart')
st.divider()

st.title("Checkpoints")
st.markdown("1. Clean data by 10/20/23")
st.markdown("2. Complete K-Means Model by midterm report")
st.markdown("3. Meet with mentor week of 11/15 and complete CNN model")
st.divider()



st.title("Datasets")
url = "https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/data"
st.markdown("[Dataset Link](%s)" % url)
st.markdown("Our dataset includes 60,000 synthetically-generated images and 60,000 real images where 100,000 of those are for training and 20,000 are for testing.")
st.divider()

st.title("References")
st.markdown("[1] Bird, J. J., & Lotfi, A. (2023, March 24). CIFAKE: Image classification and explainable identification of AI-generated synthetic images. arXiv.org. https://arxiv.org/abs/2303.14126")
st.markdown("[2] Krizhevsky, A. (2009). Learning multiple layers of features from tiny images - semantic scholar. https://www.semanticscholar.org/paper/Learning-Multiple-Layers-of-Features-from-Tiny-Krizhevsky/5d90f06bb70a0a3dced62413346235c02b1aa086 ")
st.markdown("[3] Lu, Z., Huang, D., Bai, L., Qu, J., Wu, C., Liu, X., & Ouyang, W. (2023, September 22). Seeing is not always believing: Benchmarking human and model perception of AI-generated images. arXiv.org. https://arxiv.org/abs/2304.13023v3 ")
st.markdown("[4] Maher Salman , F., & S. Abu-Nase, S. (2022). Classification of Real and Fake Human Faces Using Deep Learning. https://philpapers.org/archive/SALCOR-3.pdf ")

st.divider()



# # ------ PART 1 ------





# df = pd.DataFrame(
#     [
#        {"command": "st.selectbox", "rating": 4, "is_widget": True},
#        {"command": "st.balloons", "rating": 5, "is_widget": False},
#        {"command": "st.time_input", "rating": 3, "is_widget": True},
#    ]
# )

# # Display text
# st.text('Fixed width text')
# st.markdown('_**Markdown**_') # see #*
# st.caption('Balloons. Hundreds of them...')
# st.latex(r''' e^{i\pi} + 1 = 0 ''')
# st.write('Most objects') # df, err, func, keras!
# st.write(['st', 'is <', 3]) # see *
# st.title('My title')
# st.header('My header')
# st.subheader('My sub')
# st.code('for i in range(8): foo()')

# # * optional kwarg unsafe_allow_html = True



# # Interactive widgets
# st.button('Hit me')
# st.data_editor(df)
# st.checkbox('Check me out')
# st.radio('Pick one:', ['nose','ear'])
# st.selectbox('Select', [1,2,3])
# st.multiselect('Multiselect', [1,2,3])
# st.slider('Slide me', min_value=0, max_value=10)
# st.select_slider('Slide to select', options=[1,'2'])
# st.text_input('Enter some text')
# st.number_input('Enter a number')
# st.text_area('Area for textual entry')
# st.date_input('Date input')
# st.time_input('Time entry')
# st.file_uploader('File uploader')

# # -- add download button (start) --
# @st.cache_data
# def convert_df(df):
#     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#     return df.to_csv().encode('utf-8')

# csv = convert_df(df)

# st.download_button(
#     label="Download data as CSV",
#     data=csv,
#     file_name='large_df.csv',
#     mime='text/csv',
# )
# # -- add download button (end) --

# st.camera_input("一二三,茄子!")
# st.color_picker('Pick a color')

# # ------ PART 2 ------

# data = pd.read_csv("employees.csv")

# # Display Data
# st.dataframe(data)
# st.table(data.iloc[0:10])
# st.json({'foo':'bar','fu':'ba'})
# st.metric('My metric', 42, 2)

# # Media
# st.image('./smile.png')

# # Display Charts
# st.area_chart(data[:10])
# st.bar_chart(data[:10])
# st.line_chart(data[:10])
# # st.map(data[:10])
# st.scatter_chart(data[:10])

# # Add sidebar
# a = st.sidebar.radio('Select one:', [1, 2])
# st.sidebar.caption("This is a cool caption")
# st.sidebar.image('./smile.png')

# # Add columns
# col1, col2 = st.columns(2)
# col1.write("This is column 1")
# col2.write("This is column 2")
