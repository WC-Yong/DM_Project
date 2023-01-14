import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import datetime
import csv

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics.pairwise import haversine_distances
import sklearn.metrics
from apyori import apriori
import pickle

from fpdf import FPDF
import base64
from PIL import Image

radioOptions = np.array(["EDA", "Association Rule Mining", "Predictive Model 1 (Regression): Age Range", "Predictive Model 2 (Classification): Basket Size", "Predictive Model 3 (Classification): Gender"]) 
sideBar = st.sidebar.radio("Menus", radioOptions, index=0)

df = pd.read_csv("cleaned_dataset.csv")
dfp = pd.read_csv("property.csv")

dist = sklearn.metrics.DistanceMetric.get_metric('haversine')
dist_matrix = (dist.pairwise
    (df[['latitude','longitude']],
     dfp[['Latitude','Longitude']])*3959
)
# Note that 3959 is the radius of the earth in miles
df_dist_matrix = (
    pd.DataFrame(dist_matrix,index=df.index, 
                 columns=dfp['Title'])
)

min_dist = df_dist_matrix.idxmin(axis=1)

df_kek = dfp[(dfp['Title'] == min_dist[0])]
for i in range(1,len(min_dist)):
    df_kek = pd.concat([df_kek,dfp[(dfp['Title'] == min_dist[i])]])
    
df_kek = df_kek.reset_index(drop=True)

df2 = pd.concat([df, df_kek], axis=1)

temp = pd.DataFrame(df2['Keywords'].str[3:]).apply(lambda x: x.str.strip())
temp = temp['Keywords'].str.replace(',','').astype('float')
df2['Keywords'] = temp
df2 = df2.rename(columns = {'Keywords':'Total_Price'})

temp = pd.DataFrame(df2['listingfloorarea6'].str[3:9]).apply(lambda x: x.str.strip())
temp = temp['listingfloorarea6'].str.replace(',','').astype('float')
df2['listingfloorarea6'] = temp
df2 = df2.rename(columns = {'listingfloorarea6':'Listing_Price_psf'})

df2 = df2.drop(columns=['Unnamed: 0', 'Latitude', 'Longitude', 'Type3', 'Name'])

df = df2

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

if (sideBar == radioOptions[0]):

    title = '<p style="font-family:arial; color:Cyan; font-size:50px; font-weight: bold; text-align: left">Exploratory Data Analysis</p>'
    st.markdown(title, unsafe_allow_html=True)
    st.write('---')
    
    df_text = '<p style="font-family:arial; color:White; font-size:30px; font-weight: bold; text-align: center">Dataset</p>'
    st.markdown(df_text, unsafe_allow_html=True)
    st.write(df)
    
    desc_text = '<p style="font-family:arial; color:White; font-size:30px; font-weight: bold; text-align: center">Statistical Summary</p>'
    st.markdown(desc_text, unsafe_allow_html=True)
    st.write(df.describe())
    
    corr_text = '<p style="font-family:arial; color:White; font-size:30px; font-weight: bold; text-align: center">Correlation Heatmap</p>'
    st.markdown(corr_text, unsafe_allow_html=True)
    #fig = Figure(figsize=(6, 4), dpi=300)
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax)
    st.write(fig)
    
    cluster = '<p style="font-family:arial; color:White; font-size:30px; font-weight: bold; text-align: center">Clustering Analysis</p>'
    st.markdown(cluster, unsafe_allow_html=True)
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    newdf = df.select_dtypes(include=numerics)
    
    st.write("Please do not select the same columns!")
    option1 = st.selectbox('Select X-axis for scatter plot',
    newdf.columns)
    option2 = st.selectbox('Select Y-axis for scatter plot',
    newdf.columns)
    
    st.write("Elbow method graph to determine optimal clusters")
    
    data = newdf[[option1 ,option2]]
    inertias = []

    for i in range(1,11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    fig1, ax = plt.subplots()
    ax.plot(range(1,11), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    st.pyplot(fig1)
    
    num_clusters = st.number_input(label="Please input the optimal number of clusters", min_value=1, value=2)

    km = KMeans(n_clusters = num_clusters, random_state=1)
    km.fit(data)
    data['y']=km.labels_
    fig2 = sns.relplot(x=newdf[option1], y=newdf[option2], hue="y", data=data)
    st.pyplot(fig2)
    
    export_as_pdf = st.button("Export Report")


    if export_as_pdf:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 30)
        pdf.cell(100,100, "Exploratory Data Analysis")
        
        pdf.add_page()
        pdf.set_font('Times', 'B', 20)
        pdf.cell(40, 10, "Statistical Summary")
        
    
        df_desc = df.describe()
        df_desc = df_desc.round(decimals = 4)
        df_desc = df_desc.rename(columns = {"TimeSpent_minutes":"TimeSpent", "TotalSpent_RM": "TSpent", "Num_of_Baskets":"BasketsNo", "busyness_(%)": "busyness", "Listing_Price_psf": "ListPrice"})
        df_desc = df_desc.transpose()
        df_desc.to_csv('summary.csv')
        with open('summary.csv',newline='') as f:
            reader = csv.reader(f)
            page_width = pdf.w - 2 * pdf.l_margin
                
            pdf.ln(10)

            pdf.set_font('Courier', '', 11)
            
            col_width = page_width/6
            
            pdf.ln(1)
            
            th = pdf.font_size
            
            for row in reader:
                #print(row)
                pdf.cell(col_width, th, str(row[0]), border=1)
                pdf.cell(col_width, th, str(row[1]), border=1)
                pdf.cell(col_width, th, str(row[2]), border=1)
                pdf.cell(col_width, th, str(row[3]), border=1)
                pdf.cell(col_width, th, str(row[4]), border=1)

                
                pdf.ln(th)
                
            pdf.ln(10)
        with open('summary.csv',newline='') as f:
            reader = csv.reader(f)
            page_width = pdf.w - 2 * pdf.l_margin
                
            pdf.ln(10)

            pdf.set_font('Courier', '', 11)
            
            col_width = page_width/7
            
            pdf.ln(1)
            
            th = pdf.font_size
            
            for row in reader:
                pdf.cell(col_width, th, str(row[0]), border=1)
                pdf.cell(col_width, th, str(row[5]), border=1)
                pdf.cell(col_width, th, str(row[6]), border=1)
                pdf.cell(col_width, th, str(row[7]), border=1)
                pdf.cell(col_width, th, str(row[8]), border=1)
                
                pdf.ln(th)
                
            pdf.ln(10)
        
        pdf.add_page()
        pdf.set_font('Times', 'B', 20)
        pdf.cell(40, 10, "Correlaton Heatmap")
        fig.savefig("figure.png", format="png", bbox_inches="tight")
        filepath = "figure.png"
        img = Image.open(filepath)
        
        pdf.image(filepath,x = 30, y = 30, w=img.width/4,h=img.height/4) 
        
        pdf.add_page()
        pdf.set_font('Times', 'B', 20)
        pdf.cell(40, 10, "Clustering Analysis")
        fig2.savefig("figure1.png", format="png", bbox_inches="tight")
        filepath = "figure1.png"
        img = Image.open(filepath)
        
        pdf.image(filepath,x = 30, y = 30, w=img.width/4,h=img.height/4) 
        
        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "EDA")

        st.markdown(html, unsafe_allow_html=True)

elif (sideBar == radioOptions[1]):
    df_new = df[['Attire', 'Shirt_Colour', 'shirt_type', 'Pants_Colour', 'pants_type', 'Spectacles', 'buyDrinks']]
    list1 = df_new.columns.tolist()
    dict1 = {}
    for i in list1:
        dict1[i] = []
        
    for i in range(df_new.shape[0]):
        dict1[list(dict1.keys())[0]].append(df_new.iloc[i][0])
        dict1[list(dict1.keys())[1]].append("Shirt_" + df_new.iloc[i][1])
        dict1[list(dict1.keys())[2]].append(df_new.iloc[i][2])
        dict1[list(dict1.keys())[3]].append("Pants_" + df_new.iloc[i][3])
        dict1[list(dict1.keys())[4]].append(df_new.iloc[i][4])
        if df_new.iloc[i][5] == 'no':
            dict1[list(dict1.keys())[5]].append("")
        else:
            dict1[list(dict1.keys())[5]].append("Specs")
        if df_new.iloc[i][6] <= 0:
            dict1[list(dict1.keys())[6]].append("")
        else:
            dict1[list(dict1.keys())[6]].append("Drinks") 

    dict_df = pd.DataFrame.from_dict(dict1)
    dict_df.to_csv('assoc.csv', index=False,header=False)
    df_assoc = pd.read_csv('assoc.csv', header=None)
    
    records = []
    for i in range(0, df_assoc.shape[0]):
        records.append([str(df_assoc.values[i,j]) for j in range(0, df_assoc.shape[1])]) 
    
    association_rules = apriori(records, min_support=0.005, min_confidence=0.2, min_lift=3, min_length=2)
    association_results = list(association_rules)
    
    cnt = 0
    item1 = []
    item2 = []
    supp = []
    conf = []
    lift = []
    for item in association_results:
        
        # first index of the inner list
        # Contains base item and add item
        pair = item[0] 
        items = [x for x in pair]
        if ('Drinks' in items):
            for i in items:
                if (i not in item1) and i != "Drinks" and i != 'nan':
                    item1.append(i)
                    break
                else:
                    continue
            
            item2.append(items[items.index('Drinks')])
            supp.append(str(round(item[1],3)))
            conf.append(str(round(item[2][0][2],4)))
            lift.append(str(round(item[2][0][3],4)))

    for i in range(len(item1)):
        cnt += 1
        st.write("(Rule " + str(cnt) + ") " + item1[i] + " -> " + item2[i])
        #second index of the inner list
        st.write("Support: " + supp[i])

        #third index of the list located at 0th
        #of the third index of the inner list

        st.write("Confidence: " + conf[i])
        st.write("Lift: " + lift[i])
        st.write("=====================================")
        
elif (sideBar == radioOptions[2]):
    
    rf_reg = pickle.load(open('rf_regressor_model.sav', 'rb'))
    #features = df[]
    
    title = '<p style="font-family:arial; color:Cyan; font-size:50px; font-weight: bold; text-align: left">Predictive Model 1 (Regression): Age Range</p>'
    st.markdown(title, unsafe_allow_html=True)
    st.write('---')
    
    lat = st.number_input(label="Latitude", value=df['latitude'].mean())
    lon = st.number_input(label="Longitude", value=df['longitude'].mean())
    rain = st.number_input(label="Rainfall (mm)", min_value=0, value=0)
    timespent = st.number_input(label="Time Spent (min)", min_value=1, value=1)
    propPrice = st.number_input(label="Property Price (RM)", min_value=1, value=1)
    totalSpent = st.number_input(label="Total Money Spent (RM)", min_value=1, value=1)
    
    cols = ['latitude', 'longitude', 'Rain_(mm)', 'TimeSpent_minutes', 'Property_Price', 'TotalSpent_RM']
    rows = np.array([lat,lon,rain,timespent,propPrice,totalSpent])
    
    def predict(rows, cols, rf_reg):
        X = pd.DataFrame([rows], columns = cols)
        return rf_reg.predict(X)[0]  
     
    click = st.button('Predict')
    predicted = '<p style="font-family:arial; color:White; font-size:30px; font-weight: bold; text-align: center">Predicted Age</p>'
    st.markdown(predicted, unsafe_allow_html=True)
    st.write('---')
    
    if click:
        result = predict(rows, cols, rf_reg)
        st.write(str(result))
        
        
    export_as_pdf = st.button("Export Report")

    if export_as_pdf:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 30)
        txt1 = "Predictive Model 1 (Regression):" + '\n'
        txt1 += "Age Range"
        pdf.multi_cell(200,50, txt1)
        
        pdf.add_page()
        pdf.set_font('Times', '', 20.0)
        txt = "Latitude: " + str(lat) + '\n'
        txt += "Longitude: " + str(lon) + '\n'
        txt += "Rain Fall (mm): " + str(rain) + '\n'
        txt += "Time Spent (min): " + str(timespent) + '\n'
        txt += "Property Price (RM): " + str(propPrice) + '\n'
        txt += "Total Spent (RM): " + str(totalSpent) + '\n'
        txt += "Predicted Age Range: " + str(predict(rows, cols, rf_reg))
        
        pdf.multi_cell(0, 10, txt)
        
        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "Model_1_Age_Range")

        st.markdown(html, unsafe_allow_html=True)
    
elif (sideBar == radioOptions[3]):
    dt = pickle.load(open('model_DT.sav', 'rb'))
    
    title = '<p style="font-family:arial; color:Cyan; font-size:50px; font-weight: bold; text-align: left">Predictive Model 2 (Classification): Basket Size</p>'
    st.markdown(title, unsafe_allow_html=True)
    st.write('---')
    
    lat = st.number_input(label="Latitude", value=df['latitude'].mean())
    lon = st.number_input(label="Longitude", value=df['longitude'].mean())
    rain = st.number_input(label="Rainfall (mm)", min_value=0, value=0)
    date = st.date_input("Date", datetime.date(2015, 10, 1))
    year1 = date.year
    age = st.number_input(label="Age", min_value=1, value=1)
    
    cols = ['latitude', 'longitude', 'Rain_(mm)', 'DateTime', 'Age_Range']
    rows = np.array([lat,lon,rain,year1,age])
    
    def predict(rows, cols, dt):
        X = pd.DataFrame([rows], columns = cols)
        return dt.predict(X)[0]  
     
    click = st.button('Predict')
    predicted = '<p style="font-family:arial; color:White; font-size:30px; font-weight: bold; text-align: center">Predicted Basket Size</p>'
    st.markdown(predicted, unsafe_allow_html=True)
    st.write('---')
    
    if click:
        result = predict(rows, cols, dt)
        st.write(str(result))
        
    export_as_pdf = st.button("Export Report")

    if export_as_pdf:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 30)
        txt1 = "Predictive Model 2 (Classification):" + '\n'
        txt1 += "Basket Size"
        pdf.multi_cell(200,50, txt1)
        
        pdf.add_page()
        pdf.set_font('Times', '', 20.0)
        txt = "Latitude: " + str(lat) + '\n'
        txt += "Longitude: " + str(lon) + '\n'
        txt += "Rain Fall (mm): " + str(rain) + '\n'
        txt += "Date: " + str(date) + '\n'
        txt += "Age: " + str(age) + '\n'
        txt += "Predicted Basket Size: " + str(predict(rows, cols, dt))
        
        pdf.multi_cell(0, 10, txt)
        
        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "Model_2_Basket_Size")

        st.markdown(html, unsafe_allow_html=True)
        
elif (sideBar == radioOptions[4]):
    nb = pickle.load(open('model_nb.sav', 'rb'))
    
    title = '<p style="font-family:arial; color:Cyan; font-size:50px; font-weight: bold; text-align: left">Predictive Model 3 (Classification): Gender</p>'
    st.markdown(title, unsafe_allow_html=True)
    st.write('---')
    
    lat = st.number_input(label="Latitude", value=df['latitude'].mean())
    lon = st.number_input(label="Longitude", value=df['longitude'].mean())
    busyness = st.number_input(label="Busyness", min_value=1,value=1)
    age = st.number_input(label="Age", min_value=1, value=1)
    rain = st.number_input(label="Rainfall (mm)", min_value=0, value=0)
    
    cols = ['latitude', 'longitude','busyness_(%)','Age_range', 'Rain_(mm)']
    rows = np.array([lat,lon,busyness,age,rain])
    
    def predict(rows, cols, nb):
        X = pd.DataFrame([rows], columns = cols)
        return nb.predict(X)[0]  
     
    click = st.button('Predict')
    predicted = '<p style="font-family:arial; color:White; font-size:30px; font-weight: bold; text-align: center">Predicted Gender</p>'
    st.markdown(predicted, unsafe_allow_html=True)
    st.write('---')
    
    if click:
        result = predict(rows, cols, nb)
        st.write(str(result))
    
    export_as_pdf = st.button("Export Report")

    if export_as_pdf:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 30)
        txt1 = "Predictive Model 3 (Classification):" + '\n'
        txt1 += "Gender"
        pdf.multi_cell(200,50, txt1)
        

        pdf.add_page()
        pdf.set_font('Times', '', 20.0)
        txt = "Latitude: " + str(lat) + '\n'
        txt += "Longitude: " + str(lon) + '\n'
        txt += "Busyness: " + str(busyness) + '\n'
        txt += "Age: " + str(age) + '\n'
        txt += "Rain Fall (mm): " + str(rain) + '\n'
        
        txt += "Predicted Gender: " + str(predict(rows, cols, nb))
        
        pdf.multi_cell(0, 10, txt)
        
        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "Model_3_Gender")

        st.markdown(html, unsafe_allow_html=True)