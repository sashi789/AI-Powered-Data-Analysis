import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import io
import os
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime

# Try to import openai, but handle the case where it's not installed
try:
    import openai
    openai_available = True
except ImportError:
    openai_available = False
    st.sidebar.warning("OpenAI is not installed. AI features will be limited. Run 'pip install openai' to install it.")

# Try to import langchain, but handle the case where it's not installed
try:
    from langchain.llms import OpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    langchain_available = True
except ImportError:
    langchain_available = False
    st.sidebar.warning("LangChain is not installed. Some AI features will be limited. Run 'pip install langchain' to install it.")

# Set page configuration
st.set_page_config(
    page_title="Data Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'api_data' not in st.session_state:
    st.session_state.api_data = None

# Set OpenAI API key
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    openai.api_key = openai_api_key

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Upload", "Data Exploration", "Data Cleaning", "Visualization", "AI Insights", "Download"])

# Function to load data from various sources
def load_data(file_or_api):
    if file_or_api == "File Upload":
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'json'])
        if uploaded_file is not None:
            st.session_state.file_name = uploaded_file.name
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.json_normalize(json.loads(uploaded_file.getvalue().decode('utf-8')))
                
                st.session_state.df = df
                st.session_state.original_df = df.copy()
                return df
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return None
    elif file_or_api == "API":
        api_url = st.text_input("Enter API URL")
        if api_url and st.button("Fetch Data"):
            try:
                response = requests.get(api_url)
                data = response.json()
                st.session_state.api_data = data
                df = pd.json_normalize(data)
                st.session_state.df = df
                st.session_state.original_df = df.copy()
                st.session_state.file_name = f"api_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                return df
            except Exception as e:
                st.error(f"Error fetching API data: {e}")
                return None
    return None

# Function for exploratory data analysis
def explore_data(df):
    st.subheader("Data Overview")
    
    # Display basic information
    col1, col2 = st.columns(2)
    with col1:
        st.write("Data Shape:", df.shape)
    with col2:
        st.write("Data Types:")
        st.write(df.dtypes)
    
    # Display data sample
    st.subheader("Data Sample")
    st.dataframe(df.head(10))
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    # Check for missing values
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percent
    })
    st.write(missing_df[missing_df['Missing Values'] > 0])
    
    # Display correlation matrix for numerical columns
    st.subheader("Correlation Matrix")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numerical columns to calculate correlation.")

# Function for data cleaning
def clean_data(df):
    st.subheader("Data Cleaning")
    
    # Rename columns
    st.write("### Rename Columns")
    col1, col2 = st.columns(2)
    with col1:
        column_to_rename = st.selectbox("Select column to rename", df.columns)
    with col2:
        new_name = st.text_input("Enter new column name")
    
    if st.button("Rename Column") and new_name:
        df = df.rename(columns={column_to_rename: new_name})
        st.session_state.df = df
        st.success(f"Column renamed from '{column_to_rename}' to '{new_name}'")
    
    # Change data types
    st.write("### Change Data Types")
    col1, col2 = st.columns(2)
    with col1:
        column_to_change = st.selectbox("Select column to change type", df.columns)
    with col2:
        new_type = st.selectbox("Select new data type", ["int", "float", "str", "datetime"])
    
    if st.button("Change Data Type"):
        try:
            if new_type == "int":
                df[column_to_change] = df[column_to_change].astype(int)
            elif new_type == "float":
                df[column_to_change] = df[column_to_change].astype(float)
            elif new_type == "str":
                df[column_to_change] = df[column_to_change].astype(str)
            elif new_type == "datetime":
                df[column_to_change] = pd.to_datetime(df[column_to_change])
            
            st.session_state.df = df
            st.success(f"Data type of '{column_to_change}' changed to {new_type}")
        except Exception as e:
            st.error(f"Error changing data type: {e}")
    
    # Handle missing values
    st.write("### Handle Missing Values")
    col1, col2, col3 = st.columns(3)
    with col1:
        column_with_missing = st.selectbox("Select column with missing values", df.columns)
    with col2:
        handling_method = st.selectbox("Select handling method", ["Drop", "Fill with mean", "Fill with median", "Fill with mode", "Fill with value"])
    with col3:
        if handling_method == "Fill with value":
            fill_value = st.text_input("Enter fill value")
    
    if st.button("Handle Missing Values"):
        try:
            if handling_method == "Drop":
                df = df.dropna(subset=[column_with_missing])
            elif handling_method == "Fill with mean":
                df[column_with_missing] = df[column_with_missing].fillna(df[column_with_missing].mean())
            elif handling_method == "Fill with median":
                df[column_with_missing] = df[column_with_missing].fillna(df[column_with_missing].median())
            elif handling_method == "Fill with mode":
                df[column_with_missing] = df[column_with_missing].fillna(df[column_with_missing].mode()[0])
            elif handling_method == "Fill with value" and fill_value:
                df[column_with_missing] = df[column_with_missing].fillna(fill_value)
            
            st.session_state.df = df
            st.success(f"Missing values in '{column_with_missing}' handled using {handling_method}")
        except Exception as e:
            st.error(f"Error handling missing values: {e}")
    
    # Remove duplicates
    st.write("### Remove Duplicates")
    if st.button("Remove Duplicate Rows"):
        original_count = len(df)
        df = df.drop_duplicates()
        st.session_state.df = df
        st.success(f"Removed {original_count - len(df)} duplicate rows")
    
    # Display current dataframe
    st.subheader("Current Data")
    st.dataframe(df.head(10))
    
    return df

# Function for data visualization
def visualize_data(df):
    st.subheader("Data Visualization")
    
    # Select visualization type
    viz_type = st.selectbox("Select Visualization Type", 
                           ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", 
                            "Box Plot", "Pie Chart", "Heatmap", "Pair Plot"])
    
    if viz_type == "Bar Chart":
        col1, col2 = st.columns(2)
        with col1:
            x_column = st.selectbox("Select X-axis column", df.columns)
        with col2:
            y_column = st.selectbox("Select Y-axis column", df.select_dtypes(include=[np.number]).columns)
        
        fig = px.bar(df, x=x_column, y=y_column, title=f"Bar Chart: {y_column} by {x_column}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Line Chart":
        col1, col2 = st.columns(2)
        with col1:
            x_column = st.selectbox("Select X-axis column", df.columns)
        with col2:
            y_column = st.selectbox("Select Y-axis column", df.select_dtypes(include=[np.number]).columns)
        
        fig = px.line(df, x=x_column, y=y_column, title=f"Line Chart: {y_column} by {x_column}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Scatter Plot":
        col1, col2, col3 = st.columns(3)
        with col1:
            x_column = st.selectbox("Select X-axis column", df.select_dtypes(include=[np.number]).columns)
        with col2:
            y_column = st.selectbox("Select Y-axis column", df.select_dtypes(include=[np.number]).columns)
        with col3:
            color_column = st.selectbox("Select Color column (optional)", ["None"] + list(df.columns))
        
        if color_column == "None":
            fig = px.scatter(df, x=x_column, y=y_column, title=f"Scatter Plot: {y_column} vs {x_column}")
        else:
            fig = px.scatter(df, x=x_column, y=y_column, color=color_column, title=f"Scatter Plot: {y_column} vs {x_column} by {color_column}")
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Histogram":
        column = st.selectbox("Select column", df.select_dtypes(include=[np.number]).columns)
        bins = st.slider("Number of bins", 5, 100, 20)
        
        fig = px.histogram(df, x=column, nbins=bins, title=f"Histogram of {column}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot":
        col1, col2 = st.columns(2)
        with col1:
            y_column = st.selectbox("Select Y-axis column", df.select_dtypes(include=[np.number]).columns)
        with col2:
            x_column = st.selectbox("Select X-axis column (optional)", ["None"] + list(df.columns))
        
        if x_column == "None":
            fig = px.box(df, y=y_column, title=f"Box Plot of {y_column}")
        else:
            fig = px.box(df, x=x_column, y=y_column, title=f"Box Plot of {y_column} by {x_column}")
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Pie Chart":
        column = st.selectbox("Select column", df.columns)
        
        value_counts = df[column].value_counts()
        fig = px.pie(values=value_counts.values, names=value_counts.index, title=f"Pie Chart of {column}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Heatmap":
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            correlation_matrix = numeric_df.corr()
            fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No numerical columns to create heatmap.")
    
    elif viz_type == "Pair Plot":
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 1:
            selected_columns = st.multiselect("Select columns (2-5 recommended)", numeric_columns, default=list(numeric_columns)[:3])
            if len(selected_columns) >= 2:
                color_column = st.selectbox("Select Color column (optional)", ["None"] + list(df.columns))
                if color_column == "None":
                    fig = px.scatter_matrix(df, dimensions=selected_columns, title="Pair Plot")
                else:
                    fig = px.scatter_matrix(df, dimensions=selected_columns, color=color_column, title=f"Pair Plot colored by {color_column}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Please select at least 2 columns for pair plot.")
        else:
            st.write("Not enough numerical columns for pair plot.")

# Function for AI insights using LangChain and OpenAI
def ai_insights(df):
    st.subheader("AI-Powered Insights")
    
    if not openai_available:
        st.error("The OpenAI package is not installed. Please install it using 'pip install openai' to use AI features.")
        return
    
    if not langchain_available:
        st.warning("The LangChain package is not installed. Some advanced AI features may not be available. Install it using 'pip install langchain'.")
    
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to use AI features.")
        return
    
    # Data summary with OpenAI
    if st.button("Generate Data Summary"):
        try:
            # Prepare data description
            data_info = f"Dataset shape: {df.shape}\n"
            data_info += f"Columns: {', '.join(df.columns)}\n"
            data_info += f"Data types:\n{df.dtypes}\n"
            data_info += f"Summary statistics:\n{df.describe().to_string()}\n"
            data_info += f"First few rows:\n{df.head(5).to_string()}\n"
            
            # Create prompt for OpenAI
            prompt = f"""
            You are a data analyst. Analyze the following dataset and provide key insights:
            
            {data_info}
            
            Please provide:
            1. A summary of what this dataset contains
            2. Key observations about the data
            3. Potential issues or anomalies in the data
            4. Suggestions for further analysis
            
            Keep your response concise and focused on the most important aspects.
            """
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful data analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            
            # Display the response
            st.write("### AI-Generated Data Summary")
            st.write(response.choices[0].message.content)
            
        except Exception as e:
            st.error(f"Error generating AI insights: {e}")
    
    # Column analysis with OpenAI
    st.write("### Analyze Specific Column")
    column_to_analyze = st.selectbox("Select column to analyze", df.columns)
    
    if st.button("Analyze Column"):
        try:
            # Prepare column data
            column_data = df[column_to_analyze]
            data_type = str(column_data.dtype)
            
            column_info = f"Column name: {column_to_analyze}\n"
            column_info += f"Data type: {data_type}\n"
            
            if pd.api.types.is_numeric_dtype(column_data):
                column_info += f"Min: {column_data.min()}\n"
                column_info += f"Max: {column_data.max()}\n"
                column_info += f"Mean: {column_data.mean()}\n"
                column_info += f"Median: {column_data.median()}\n"
                column_info += f"Standard deviation: {column_data.std()}\n"
            else:
                column_info += f"Unique values: {column_data.nunique()}\n"
                column_info += f"Most common values: {column_data.value_counts().head(5).to_dict()}\n"
            
            column_info += f"Missing values: {column_data.isnull().sum()} ({column_data.isnull().mean() * 100:.2f}%)\n"
            
            # Create prompt for OpenAI
            prompt = f"""
            You are a data analyst. Analyze the following column from a dataset and provide insights:
            
            {column_info}
            
            Please provide:
            1. A summary of what this column represents
            2. Key observations about the distribution or patterns
            3. Potential issues or anomalies
            4. Suggestions for handling this column in analysis
            
            Keep your response concise and focused on the most important aspects.
            """
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful data analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800
            )
            
            # Display the response
            st.write(f"### AI Analysis of '{column_to_analyze}'")
            st.write(response.choices[0].message.content)
            
        except Exception as e:
            st.error(f"Error analyzing column: {e}")
    
    # Generate visualization recommendations
    if st.button("Recommend Visualizations"):
        try:
            # Prepare data description
            data_info = f"Dataset shape: {df.shape}\n"
            data_info += f"Columns: {', '.join(df.columns)}\n"
            data_info += f"Data types:\n{df.dtypes}\n"
            
            # Create prompt for OpenAI
            prompt = f"""
            You are a data visualization expert. Based on the following dataset information, recommend 3-5 specific visualizations that would be most insightful:
            
            {data_info}
            
            For each recommendation, please provide:
            1. The type of visualization (e.g., bar chart, scatter plot)
            2. Which specific columns to use
            3. Why this visualization would be insightful
            4. Any specific settings or transformations to apply
            
            Focus on visualizations that would reveal important patterns or relationships in the data.
            """
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful data visualization expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            
            # Display the response
            st.write("### AI-Recommended Visualizations")
            st.write(response.choices[0].message.content)
            
        except Exception as e:
            st.error(f"Error generating visualization recommendations: {e}")

# Function to download the cleaned data
def download_data(df):
    st.subheader("Download Cleaned Data")
    
    # Select file format
    file_format = st.selectbox("Select file format", ["CSV", "Excel", "JSON"])
    
    if st.button("Download Data"):
        if file_format == "CSV":
            csv = df.to_csv(index=False)
            b64 = base64_encode_data(csv)
            href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        elif file_format == "Excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
            excel_data = output.getvalue()
            b64 = base64_encode_data(excel_data)
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="cleaned_data.xlsx">Download Excel File</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        elif file_format == "JSON":
            json_str = df.to_json(orient='records')
            b64 = base64_encode_data(json_str)
            href = f'<a href="data:file/json;base64,{b64}" download="cleaned_data.json">Download JSON File</a>'
            st.markdown(href, unsafe_allow_html=True)

# Helper function to encode data for download
def base64_encode_data(data):
    import base64
    if isinstance(data, str):
        data = data.encode()
    b64 = base64.b64encode(data).decode()
    return b64

# Main application
def main():
    st.title("📊 Advanced Data Analysis Tool")
    
    # Data Upload Page
    if page == "Data Upload":
        st.header("Data Upload")
        
        # Choose data source
        data_source = st.radio("Select Data Source", ["File Upload", "API"])
        
        df = load_data(data_source)
        
        if df is not None:
            st.success(f"Data loaded successfully! Shape: {df.shape}")
            st.dataframe(df.head())
    
    # Data Exploration Page
    elif page == "Data Exploration":
        st.header("Exploratory Data Analysis")
        
        if st.session_state.df is not None:
            explore_data(st.session_state.df)
        else:
            st.warning("Please upload data first.")
    
    # Data Cleaning Page
    elif page == "Data Cleaning":
        st.header("Data Cleaning")
        
        if st.session_state.df is not None:
            clean_data(st.session_state.df)
            
            # Option to reset to original data
            if st.button("Reset to Original Data"):
                st.session_state.df = st.session_state.original_df.copy()
                st.success("Data reset to original state")
        else:
            st.warning("Please upload data first.")
    
    # Visualization Page
    elif page == "Visualization":
        st.header("Data Visualization")
        
        if st.session_state.df is not None:
            visualize_data(st.session_state.df)
        else:
            st.warning("Please upload data first.")
    
    # AI Insights Page
    elif page == "AI Insights":
        st.header("AI-Powered Insights")
        
        if st.session_state.df is not None:
            ai_insights(st.session_state.df)
        else:
            st.warning("Please upload data first.")
    
    # Download Page
    elif page == "Download":
        st.header("Download Cleaned Data")
        
        if st.session_state.df is not None:
            download_data(st.session_state.df)
        else:
            st.warning("Please upload data first.")

if __name__ == "__main__":
    main()