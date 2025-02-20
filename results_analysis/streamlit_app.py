import streamlit as st
import pandas as pd
import ast
import matplotlib as mpl
import matplotlib.pyplot as plt
from log_parser.visualization import plot_aggregated_comparisons, plot_granularity_combined


mpl.rcParams['text.usetex'] = False
st.title("Interactive Dashboard with Tag Filtering")

# Allow multiple CSV files to be uploaded
uploaded_files = st.sidebar.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    # If more than one file is uploaded, let the user select one
    if len(uploaded_files) > 1:
        file_names = [file.name for file in uploaded_files]
        selected_file_name = st.sidebar.selectbox("Select a CSV file", file_names)
        selected_file = next(file for file in uploaded_files if file.name == selected_file_name)
    else:
        selected_file = uploaded_files[0]

    # Read the selected CSV file into a DataFrame
    df = pd.read_csv(selected_file)
    st.write(f"### Data from {selected_file.name}")
    st.dataframe(df.head())

    # Preprocess the tags column: convert the string representation of a list into an actual list
    if 'tags' in df.columns:
        df['tags_list'] = df['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    else:
        st.error("CSV does not contain a 'tags' column.")
        st.stop()

    # Build a sorted list of unique tags for filtering
    all_tags = sorted({tag for tags in df['tags_list'] for tag in tags})

    # Create a sidebar widget for tag selection
    selected_tags = st.sidebar.multiselect("Select Tags", all_tags)

    # Filter the DataFrame based on the selected tags
    if selected_tags:
        filtered_df = df[df['tags_list'].apply(lambda tags: any(tag in tags for tag in selected_tags))]
    else:
        filtered_df = df

    st.write("### Filtered Data")
    st.dataframe(filtered_df)

    # Automatically generate and display the Aggregated Comparison Plot
    temp_output_agg = "aggregated_plot.png"
    plot_aggregated_comparisons(filtered_df, temp_output_agg)
    st.image(temp_output_agg, caption="Aggregated Comparison Plot", use_column_width=True)

    # Check if granularity data columns are available before plotting
    required_granularity_columns = ['avg_block_size', 'variables', 'constraints']
    if all(col in filtered_df.columns for col in required_granularity_columns):
        temp_output_gran = "granularity_plot.png"
        plot_granularity_combined(filtered_df, temp_output_gran)
        st.image(temp_output_gran, caption="Granularity Plot", use_column_width=True)
    else:
        st.info("Granularity data is not available for this dataset.")
else:
    st.info("Please upload one or more CSV files to start.")
