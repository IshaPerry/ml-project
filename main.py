import streamlit as st
import pandas as pd



st.title("Introduction/Background")



st.title("Problem Definition")

st.markdown('Our Project')
st.text('Fixed width text')

st.title("Methods")


st.title("Potential Results/Discussion")
st.markdown("By comparing multiple models of classification, we will be "
            "able to determine the best classification model for Because "
            "of the previously stated ~40% accuracy of humans when classifying "
            "real vs AI, we can define success as any model classifying images "
            "with accuracy above a human’s 38.7%. Knowing that this is a lower "
            "number, we have also found evidence that most modern classification "
            "AI algorithms can reach accuracy levels of ~87% (Lu et al., 2023), "
            "so we will be aiming for this level of accuracy as well. We also want "
            "to assess precision (to identify the rate of false positives), recall "
            "(to identify the rate of false negatives), and f1 (a metric to help "
            "combine precision and recall). To help extract values for our success "
            "criteria, we will use sklearn.metrics library and the testing set from "
            "our data set, allowing us to easily extract quantitative metrics. ")

st.title("Proposed Timeline")

st.title("Datasets")

st.title("References")




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
