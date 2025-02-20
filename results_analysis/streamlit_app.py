import streamlit as st
import pandas as pd
import ast
import matplotlib.pyplot as plt
from log_parser.visualization import plot_aggregated_comparisons, plot_granularity_combined

st.title("Interactive Dashboard with Tag Filtering")

# File uploader to allow CSV selection
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Preprocess the tags column: convert string representation of list to an actual list
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

    # Automatically generate and display the plots without needing to click buttons
    temp_output_agg = "aggregated_plot.png"
    plot_aggregated_comparisons(filtered_df, temp_output_agg)
    st.image(temp_output_agg, caption="Aggregated Comparison Plot", use_column_width=True)

    temp_output_gran = "granularity_plot.png"
    plot_granularity_combined(filtered_df, temp_output_gran)
    st.image(temp_output_gran, caption="Granularity Plot", use_column_width=True)
else:
    st.info("Please upload a CSV file to start.")
