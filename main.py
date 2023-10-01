import streamlit as st
import pandas as pd


st.title("Classifying Images: Real Vs AI Generated")
st.caption("Members: Isha Perry, Ariane Richard, Emily Wu, Sydney Defelice, Claire Matheny")
st.divider()

st.title("Introduction/Background")
st.markdown("Artificial Intelligence (AI) generated images lead to a mistrust in the reliability of photographs, which once served as proof of an event. In fact, research shows that humans misclassify real images from AI generated photos at a rate of 38.7% (Lu et al., 2023). AI-generated images can have a widespread negative impact from creating false alibis, winning art-competitions, and generating false historical and current events images. The dataset utilized for this project is obtained from Kaggle.com and contains 120,000 images. Its features consist of 60,000 synthetically-generated (fake) images and 60,000 real images that were obtained from CIFAR-10. 10,000 images of those two categories compose the testing data.")
st.divider()

st.title("Problem Definition")
st.markdown("AI-generated images allow for the manipulation of reality. The motivation of this project is to identify images as real (non AI-generated) or fake (AI-generated) to a high degree of accuracy.")
st.divider()

st.title("Methods")
st.markdown("To classify images into our two prediction categories (real or fake), we will be utilizing both supervised and unsupervised learning techniques to cross analyze detection accuracy. For supervised learning, we will be utilizing a convoluted neural network (CNN). Using a CNN, we can leverage image matrix inputs to extract features that get progressively more accurate with each layer. For unsupervised learning, we will be utilizing K-Means Clustering (via scikit-learn) to create clusters of images that share characteristics with those within their cluster, and dissimilar to those outside of their cluster. In both cases, several image processing techniques are required to assign weights that are propagated throughout the network. For image processing, we will be using various Python libraries such as OpenCV, Scikit-image, PIL (Python Image Library), NumPy, and Mahotas.")
st.divider()


st.title("Potential Results/Discussion")
st.markdown("By comparing multiple models of classification, we will be able to determine the best model for identifying real and AI generated photos. Because of the humans' previously stated ~40% accuracy of this classification, we can define success as any model classifying images with accuracy above a human’s 38.7%. Knowing that this is a lower number, we have also found evidence that most modern classification AI algorithms can reach accuracy levels of ~87% (Lu et al., 2023), so we will be aiming for this level of accuracy as well. We also want to assess precision (to identify the rate of false positives), recall (to identify the rate of false negatives), and f1 (a metric to help combine precision and recall). To help extract values for our success criteria, we will use sklearn.metrics library and the testing set from our data set, allowing us to easily extract quantitative metrics. ")
st.divider()

st.title("Proposed Timeline")
st.divider()

st.title("Datasets")
url = "https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/data"
st.markdown("[Dataset Link](%s)" % url)
st.markdown("Our dataset includes 60,000 synthetically-generated images (made with Stable Diffusion), 60,000 real images, 100,000 images for training, and 20,000 images for testing.")
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
