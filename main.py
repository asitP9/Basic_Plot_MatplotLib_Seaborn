# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def my_histogram():
    # Use a breakpoint in the code line below to debug your script.
    plt.style.use('seaborn-white')
    data = np.random.randn(1000)
    plt.hist(data)
    plt.hist(data, bins=30, alpha=0.5, histtype='stepfilled', color="steelblue", edgecolor="none")
    plt.title("Score distribution")
    plt.xlabel("Score")
    plt.ylabel("#Employee")
    plt.savefig("SavedPlot1.png")
    plt.show()  # Press Ctrl+F8 to toggle the breakpoint.



def boxPlotGraph():
    np.random.seed(10)
    collection_1=np.random.normal(100, 10, 200)
    collection_2=np.random.normal(80, 30, 200)
    collection_3=np.random.normal(90, 20, 200)
    collection_4=np.random.normal(70, 25, 200)
    fig=plt.figure()
    ax=fig.add_axes([0, 0, 1, 1])
    plt_data=[collection_1, collection_2, collection_3, collection_4]
    bp=ax.boxplot(plt_data)
    plt.savefig("SavedPlot1.png")
    plt.show()



def barChart():
    Objects=("Python", "Java", "JavaScript", "C#", "PHP", "C/C++")
    y_pos=np.arange(len(Objects))
    performance=[31.6, 17.7, 8, 7, 6.2, 5.9]
    plt.bar(y_pos, performance, align="center", alpha=0.5)
    plt.xticks(y_pos, Objects)
    plt.ylabel("Share")
    plt.title("Programming Language Popularity")
    plt.show()



def violinChart():
    np.random.seed(10)
    data_1=np.random.normal(100, 10, 200)
    data_2=np.random.normal(80, 30, 200)
    data_3=np.random.normal(90, 20, 200)
    data_4=np.random.normal(70, 25, 200)

    data_to_plot=[data_1, data_2, data_3, data_4]

    # Create a figure Instance
    fig=plt.figure()

    # Create an axes instance
    ax=fig.add_axes([0, 0, 1, 1])
    # Create a boxPlot
    bp=ax.violinplot(data_to_plot)
    plt.show()



def stackedColumnChart():
    from matplotlib import rc
    rc('font', weight="bold")

    # Values of each group
    bars1=[11, 14, 10, 18]
    bars2=[17, 21, 19, 8]
    bars3= [36, 22, 3, 16]

    # Heights of Bar1 + Bar2
    bars=np.add(bars1, bars2).tolist()
    # The position of the bar on the X Axis
    r=[0, 1, 2, 3]

    # Names of group and Bar Width
    names=["Hyderabad", "Delhi", "Mumbai", "Kolkata", "Chennai"]
    barWidth=1

    # Create Brown Bars
    plt.bar(r, bars1, color="#7f6d5f", edgecolor="white", width=barWidth)

    # Create Green Bars(Middle), on top of first ones
    plt.bar(r, bars2, bottom=bars1, color="#557f2d", edgecolor="white", width=barWidth)

    # Create Green Bars on top
    plt.bar(r, bars3, bottom=bars, color="#2d7f5e", edgecolor="white", width=barWidth)

    # Custom X Axis
    plt.xticks(r, names, fontweight='bold')
    plt.xlabel("Cities")
    plt.ylabel("in thousand crores")
    plt.title("Quarterly Average GST Collection")
    plt.suptitle("Top 5 Cities in India"),

    # Show Graphic
    plt.show()



def scatterPlot():
    x_hours_study=[44, 25, 35, 40, 18, 40, 45, 21, 30, 12]
    y_students_grade=[89, 69, 50, 81, 98, 80, 90, 70, 80, 34]

    plt.scatter(x_hours_study, y_students_grade, color="r")
    plt.title("Scatter Plot Example")
    plt.xlabel("Hours/Week")
    plt.ylabel("Grades Scored")
    plt.show()



def pieChart():
    objects=("Python", "Java", "JavaScript", "C#", "PHP", "C/C++")
    y_pos=np.arange(len(objects))
    performance=[31.6, 17.7, 8, 7, 6.2, 5.9]
    plt.pie(performance, labels=objects)
    plt.title("Programming Language Popularity")
    plt.show()



def heatMap():
    # Generate a random no, you can refer the data numbers also
    data=np.random.rand(4, 2)
    rows=list('1234')
    columns=list("MF")
    fig, ax=plt.subplots()
    # Advance color controls
    ax.pcolor(data, cmap=plt.cm.Reds, edgecolors='k')
    ax.set_xticks(np.arange(0, 2)+0.5)
    ax.set_yticks(np.arange(0, 4)+0.5)
    # Here we position the tick labels for x and y axis
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    # Values against each labels
    ax.set_xticklabels(columns, minor=False, fontsize=20)
    ax.set_yticklabels(rows, minor=False, fontsize=20)
    plt.show()



def ThreeDWireframePlot():
    def f(x, y):
        return np.sin(np.sqrt(x**2 + y**2))
    x=np.linspace(-5, 5, 25)
    y=np.linspace(-5, 5, 25)
    X, Y=np.meshgrid(x, y)
    Z=f(X, Y)
    fig=plt.figure()
    ax=plt.axes(projection="3d")
    ax.plot_wireframe(X, Y, Z, color="black")
    ax.set_title("3D Wireframe")
    plt.savefig("SavedPlot1.png")
    plt.show()



def areaGraph():
    my_count=["France", "Australia", "Japan", "USA", "Germany", "SriLanka", "China", "England", "Spain", "Greece", "Morocco", "South Africa",
         "India", "Argentina", "Chili", "Brazil"]
    df=pd.DataFrame({
        "Country":np.repeat(my_count, 10),
        "years":[2009, 2010, 2011, 2012, 2013]*32,
        "value":np.random.rand(160)
    })
    # Create a grid: Initialize it
    g=sns.FacetGrid(df, col="Country", hue="Country", col_wrap=4)
    # Add the line over the area with the plot function
    g=g.map(plt.plot, "years", "value")

    # Fill the area with Fill Between
    g=g.map(plt.fill_between, 'years', 'value', alpha=0.2).set_titles("{col_name} country")

    # Control the title of each facet
    g=g.set_titles("{col_name}")

    # Add a title for the whole
    plt.subplots_adjust(top=0.92)
    g = g.fig.suptitle("Evolution of the data in 16 countries")

    plt.show()



def bulletGraph():
    import plotly.graph_objects as go
    fig=go.Figure(go.Indicator(
        mode="number+gauge+delta", value=220,
        domain={'x':[0, 1], 'y':[0, 1]},
        delta={'reference': 280, 'position':'top'},
        title={'text':"<b>Profit</b><br><span style='color: gray; font-size:0.8em'>U.S $</span>", 'font':{'size': 14}},
        gauge={
            'shape':"bullet",
            "axis":{'range':[None, 300]},
            "threshold":{
                "line":{"color":'red', 'width': 2},
                "thickness":0.75,
                'value':270
            },
            "bgcolor":"white",
            "steps":[
                {
                    'range':[0, 150],
                    'color':'cyan'
                 },
                {
                    'range':[150, 250],
                    'color':'royalblue'
                }],
            "bar":{'color':"darkblue"}
        }
    ))
    fig.update_layout(height=250)
    fig.show()



def multivalueBulletGraph():
    import plotly.graph_objects as go
    fig=go.Figure()
    fig.add_trace(go.Indicator(
        mode="number+gauge+delta", value=180,
        domain={'x': [0.5, 1], 'y': [0.08, 0.25]},
        delta={'reference': 200},
        title={'text': "Revenue"},
        gauge={
            'shape': "bullet",
            "axis": {'range': [None, 300]},
            "threshold": {
                "line": {"color": 'black', 'width': 2},
                "thickness": 0.75,
                'value': 170
            },
            "steps": [
                {
                    'range': [0, 150],
                    'color': 'gray'
                },
                {
                    'range': [150, 250],
                    'color': 'lightgray'
                }],
            "bar": {'color': "black"}}))
    fig.add_trace(go.Indicator(
        mode="number+gauge+delta", value=35,
        domain={'x': [0.5, 1], 'y': [0.4, 0.6]},
        delta={'reference': 200},
        title={'text': "Profit"},
        gauge={
            'shape': "bullet",
            "axis": {'range': [None, 100]},
            "threshold": {
                "line": {"color": 'black', 'width': 2},
                "thickness": 0.75,
                'value': 50
            },
            "steps": [
                {
                    'range': [0, 25],
                    'color': 'gray'
                },
                {
                    'range': [25, 75],
                    'color': 'lightgray'
                }],
            "bar": {'color': "black"}}))
    fig.add_trace(go.Indicator(
        mode="number+gauge+delta", value=220,
        domain={'x': [0.25, 1], 'y': [0.7, 0.9]},
        delta={'reference': 200},
        title={'text': "Satisfaction"},
        gauge={
            'shape': "bullet",
            "axis": {'range': [None, 300]},
            "threshold": {
                "line": {"color": 'black', 'width': 2},
                "thickness": 0.75,
                'value': 210
            },
            "steps": [
                {
                    'range': [0, 150],
                    'color': 'gray'
                },
                {
                    'range': [150, 250],
                    'color': 'lightgray'
                }],
            "bar": {'color': "black"}}))
    fig.update_layout(height=400, margin= {'t': 0, 'b': 0, 'l':0})
    fig.show()



def calendarPlot():
    import calmap
    df=pd.read_csv("https://raw.githubusercontent.com/swapnilsaurav/Dataset/master/stock_price.csv")
    # Prepare the data for plotting
    # print("df= \n", df)
    df["date"]=pd.to_datetime(df["date"])
    # print("df[date]\n", df["date"])

    # The data must be a series with a date time index
    df.set_index("date", inplace=True)

    x=df[df["year"]==2016]["Close"]
    print("x=\n", x)
    # Plot the data using calmap
    calmap.calendarplot(x, fig_kws={'figsize': (16, 10)},
                        yearlabel_kws={"color": "black", "fontsize" : 14},
                        subplot_kws={"title": "Stock Prices"})
    plt.show()



def twinXExample():
    t=np.arange(1, 50)
    data1=t**2
    data2=t**3

    fig, ax1=plt.subplots()
    color="tab:blue"
    ax1.set_xlabel("Value (s)")
    ax1.set_ylabel('Square Value ', color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2=ax1.twinx()
    color="tab:green"
    ax2.set_ylabel('Cube', color=color)
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('twinx() example\n\n', fontweight="bold")
    plt.show()



def wordCloud():
    import nltk
    from urllib.request import urlopen
    from bs4 import BeautifulSoup
    nltk.download('punkt')
    # Step 1: Read in the data
    url="https://storymirror.com/read/story/english/0hlplnpj/ojass-helps-little-people"
    html=urlopen(url).read()
    print(html)

    # Step 2: Extract just the Text from the WebPage
    soup=BeautifulSoup(html)
    print("Text from the web page\n ", soup)
    #
    # # Kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()  # Rip it out
    print("******************************Extracting Soup:***********************************\n\n\n\n", soup)
    #
    # Step 3: Extract plain Text and remove WhiteSpacing
    text=soup.get_text()
    print("My Text:\n\n",text)

    # Break into lines and remove leading and trailing space on each
    lines=(line.strip() for line in text.splitlines())

    # Break multi-headlines into a line each
    chunks=(phrase.strip() for line in lines for phrase in line.split(" "))
    print("my chunks...\n", chunks)

    # Drop blank lines
    text='n'.join(chunk for chunk in chunks if chunk)
    print("Updated Text\n\n",text)

    # Step 4: Remove Stop Words, tokenise and convert to lower case
    # download and print the stop words for the english language
    from nltk.corpus import stopwords
    # nltk.download('stopwords')

    stop_words=set(stopwords.words("english"))
    print("stop_words = \n",stop_words)

    # tokenise the data set
    from nltk.tokenize import sent_tokenize, word_tokenize
    words=word_tokenize(text)
    print("updated words after tokenizing\n\n", words)

    # Removes punctuation and numbers
    wordsFiltered=[word.lower() for word in words if word.isalpha()]
    print("Filtered Words \n\n\n", wordsFiltered)

    # Remove Stop Words from tokenised data set
    filtered_words=[word for word in wordsFiltered if word not in stopwords.words('english')]
    print(filtered_words)
    #
    # Step 5: Create the word cloud
    from wordcloud import WordCloud
    wc=WordCloud(max_words=1000, margin=10, background_color="white", scale=3, relative_scaling=0.5, width=500, height=400, random_state=1)\
        .generate(' '.join(filtered_words))
    plt.figure(figsize=(20, 10))
    plt.imshow(wc)
    plt.axis("off")
    plt.show()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    wordCloud()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
