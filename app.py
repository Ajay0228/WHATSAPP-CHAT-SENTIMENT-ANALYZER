# Importing modules
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import nltk
import helper
import preprocessor
import mplcyberpunk

# App title
st.sidebar.title("Whatsapp Chat Analyzer")

# VADER : is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments
nltk.download('vader_lexicon')

# File upload button
uploaded_file = st.sidebar.file_uploader("Choose a file")
checkbox_button = st.sidebar.checkbox
# Main heading
st. markdown("<h1 style='text-align: center; color: grey;'>Whatsapp Chat Analyzer</h1>", unsafe_allow_html=True)

if uploaded_file is not None:

    # Getting byte form & then decoding
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")

    # Perform preprocessing
    df = preprocessor.preprocess(data)

    # Importing SentimentIntensityAnalyzer class from "nltk.sentiment.vader"
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # Object
    sentiments = SentimentIntensityAnalyzer()

    # Creating different columns for (Positive/Negative/Neutral)
    df["po"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]]  # Positive
    df["ne"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]]  # Negative
    df["nu"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]]  # Neutral


    # To indentify true sentiment per row in message column
    def sentiment(data):
        if data["po"] >= data["ne"] and data["po"] >= data["nu"]:
            return 1
        if data["ne"] >= data["po"] and data["ne"] >= data["nu"]:
            return -1
        if data["nu"] >= data["po"] and data["nu"] >= data["ne"]:
            return 0

    # Creating new column & Applying function
    df['value'] = df.apply(lambda row: sentiment(row), axis=1)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user,df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # monthly timeline
        col1, col2, col3 = st.columns(3)
        with col1:
            st.title("Monthly Timeline(Positive)")
            timeline = helper.monthly_timeline(selected_user,df,1)
            fig,ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'],color='green')
            plt.xticks(rotation='vertical')
            plt.style.use('cyberpunk')
            st.pyplot(fig)
        with col2:
            st.title("Monthly Timeline(Neutral)")
            timeline = helper.monthly_timeline(selected_user,df,0)
            fig,ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'],color='fuchsia',marker='s')
            plt.xticks(rotation='vertical')
            plt.style.use('cyberpunk')
            st.pyplot(fig)
        with col3:
            st.title("Monthly Timeline(Negative)")
            timeline = helper.monthly_timeline(selected_user,df,-1)
            fig,ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'],color='green')
            plt.xticks(rotation='vertical')
            plt.style.use('cyberpunk')
            st.pyplot(fig)

        # daily timeline
        col1, col2, col3 = st.columns(3)
        with col1:
            st.title("Daily Timeline(Positive)")
            daily_timeline = helper.daily_timeline(selected_user, df, 1)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='orange',marker='.')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.title("Daily Timeline(Neutral)")
            daily_timeline = helper.daily_timeline(selected_user, df, 0)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='lime',marker='.')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.title("Daily Timeline(Negative)")
            daily_timeline = helper.daily_timeline(selected_user, df, -1)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='orange',marker='.')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.day_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        # Weekly Activity Map
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                st.title("Weekly Activity Map(Positive)")
                user_heatmap = helper.activity_heatmap(selected_user, df,1)
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except:
                st.image('error.webp')
        with col2:
            try:
                st.title("Weekly Activity Map(Neutral)")
                user_heatmap = helper.activity_heatmap(selected_user, df,0)
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except:
                st.image('error.webp')
        with col3:
            try:
                st.title("Weekly Activity Map(Negative)")
                user_heatmap = helper.activity_heatmap(selected_user, df,-1)
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except:
                st.image('error.webp')

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                st.header("By graph")
                ax.bar(x.index, x.values,color='greenyellow')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.header("Percentage contribution in chat")
                st.dataframe(new_df)

         # Percentage contributed
        if selected_user == 'Overall':
            col1,col2,col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: white;'>Most Positive Contribution</h3>",unsafe_allow_html=True)
                x = helper.percentage(df, 1)
                
                # Displaying
                st.dataframe(x)
            with col2:
                st.markdown("<h3 style='text-align: center; color: white;'>Most Neutral Contribution</h3>",unsafe_allow_html=True)
                y = helper.percentage(df, 0)
                
                # Displaying
                st.dataframe(y)
            with col3:
                st.markdown("<h3 style='text-align: center; color: white;'>Most Negative Contribution</h3>",unsafe_allow_html=True)
                z = helper.percentage(df, -1)
                
                # Displaying
                st.dataframe(z)

        # WordCloud
        col1,col2,col3 = st.columns(3)
        with col1:
            try:
                st.title("Positive-Word-cloud")
                df_wc = helper.create_wordcloud(selected_user, df,1)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                plt.grid(visible=False)
                st.pyplot(fig)
            except:
                # Display error message
                st.image('error.webp')
        with col2:
            try:
                st.title("Neutral-Word-cloud")
                df_wc = helper.create_wordcloud(selected_user, df,0)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                plt.grid(visible=False)
                st.pyplot(fig)
            except:
                # Display error message
                st.image('error.webp')
        with col3:
            try:
                st.title("Negative-Word-cloud")
                df_wc = helper.create_wordcloud(selected_user, df,-1)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                plt.grid(visible=False)
                st.pyplot(fig)
            except:
                # Display error message
                st.image('error.webp')


        # most common words
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                most_common_df = helper.most_common_words(selected_user, df,1)

                fig, ax = plt.subplots()

                ax.barh(most_common_df[0], most_common_df[1])
                plt.xticks(rotation='vertical')

                st.title('Positive Most common words')
                st.pyplot(fig)
            except:
                # Disply error image
                st.image('error.webp')   
        with col2:
            try:
                most_common_df = helper.most_common_words(selected_user, df,0)

                fig, ax = plt.subplots()

                ax.barh(most_common_df[0], most_common_df[1])
                plt.xticks(rotation='vertical')

                st.title('Neutral Most common words')
                st.pyplot(fig)
            except:
                # Disply error image
                st.image('error.webp')   
        with col3:
            try:
                most_common_df = helper.most_common_words(selected_user, df,-1)

                fig, ax = plt.subplots()

                ax.barh(most_common_df[0], most_common_df[1])
                plt.xticks(rotation='vertical')

                st.title('Negative Most common words')
                st.pyplot(fig)
            except:
                # Disply error image
                st.image('error.webp')   


        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            # fig, ax = plt.subplots()
            # ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            # st.pyplot(fig)
            fig, ax = plt.subplots()
            # Bar Chart
            ax.bar(emoji_df[0], emoji_df[1],color='yellow')
            # ax.set_xlabel('Emoji')
            ax.set_ylabel('Frequency')
            ax.set_title('Emoji Frequency Distribution')
            ax.set_xticklabels(emoji_df[0], fontname='Segoe UI Emoji')

            st.pyplot(fig)











