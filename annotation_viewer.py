import streamlit as st
import pandas as pd
import json
import os

# Configuration
OUTPUT_JSON = "reviewed_output.json"

st.set_page_config(
    page_title="Annotation Results Viewer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_annotation_data():
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_JSON)
    if not os.path.exists(output_path):
        st.error(f"No annotation data found at {output_path}")
        return pd.DataFrame()
    
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def save_annotation_data(df):
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_JSON)
    data = df.to_dict('records')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Main app
st.title("📊 Annotation Results Viewer")

df = load_annotation_data()
if df.empty:
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Annotation status filter
annotation_status = st.sidebar.selectbox(
    "Annotation Status",
    ["All", "Annotated", "Not Annotated"],
    index=0
)

# Filter by annotation status
if annotation_status == "Annotated":
    filtered_df = df[df["human_label"].notna() & (df["human_label"] != "")]
elif annotation_status == "Not Annotated":
    filtered_df = df[df["human_label"].isna() | (df["human_label"] == "")]
else:
    filtered_df = df.copy()

# Review reason filter
if "review_reason" in df.columns:
    review_reasons = ["All"] + sorted(df["review_reason"].dropna().unique().tolist())
    selected_reason = st.sidebar.selectbox("Review Reason", review_reasons)
    if selected_reason != "All":
        filtered_df = filtered_df[filtered_df["review_reason"] == selected_reason]

# Confidence filter
if "confidence" in df.columns:
    min_conf, max_conf = st.sidebar.slider(
        "Confidence Range",
        min_value=float(df["confidence"].min()),
        max_value=float(df["confidence"].max()),
        value=(float(df["confidence"].min()), float(df["confidence"].max())),
        step=0.01
    )
    filtered_df = filtered_df[
        (filtered_df["confidence"] >= min_conf) & 
        (filtered_df["confidence"] <= max_conf)
    ]

# Label filters
if "label" in df.columns:
    model_labels = ["All"] + sorted(df["label"].dropna().unique().tolist())
    selected_model_label = st.sidebar.selectbox("Model Label", model_labels)
    if selected_model_label != "All":
        filtered_df = filtered_df[filtered_df["label"] == selected_model_label]

if "human_label" in df.columns:
    human_labels = ["All"] + sorted(df["human_label"].dropna().unique().tolist())
    selected_human_label = st.sidebar.selectbox("Human Label", human_labels)
    if selected_human_label != "All":
        filtered_df = filtered_df[filtered_df["human_label"] == selected_human_label]

# Display summary statistics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", len(df))
with col2:
    annotated_count = len(df[df["human_label"].notna() & (df["human_label"] != "")])
    st.metric("Annotated", annotated_count)
with col3:
    not_annotated_count = len(df) - annotated_count
    st.metric("Not Annotated", not_annotated_count)
with col4:
    if len(df) > 0:
        progress = annotated_count / len(df)
        st.metric("Progress", f"{progress:.1%}")

st.markdown("---")

# Display filtered results
st.subheader(f"📋 Results ({len(filtered_df)} records)")

if len(filtered_df) == 0:
    st.info("No records match the current filters.")
else:
    # Record selection for editing
    if len(filtered_df) > 0:
        record_options = [f"Record {i+1}: {row.get('title', 'No title')[:50]}..." 
                         for i, row in filtered_df.iterrows()]
        
        selected_idx = st.selectbox(
            "Select record to view/edit:",
            range(len(filtered_df)),
            format_func=lambda x: record_options[x]
        )
        
        if selected_idx is not None:
            selected_record = filtered_df.iloc[selected_idx]
            original_idx = filtered_df.index[selected_idx]
            
            # Display record details
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📝 Message Content")
                st.text_area(
                    "User Text",
                    value=selected_record.get("usertext", ""),
                    height=200,
                    disabled=True
                )
                
                # Edit annotation
                st.subheader("✏️ Edit Annotation")
                
                # Human label selection
                label_options = ["", "A0", "A1", "A2", "A3"]
                current_human_label = selected_record.get("human_label", "")
                if pd.isna(current_human_label):
                    current_human_label = ""
                
                new_human_label = st.selectbox(
                    "Human Label",
                    label_options,
                    index=label_options.index(current_human_label) if current_human_label in label_options else 0,
                    key=f"human_label_{original_idx}"
                )
                
                # Annotator notes
                current_notes = selected_record.get("annotator_notes", "")
                if pd.isna(current_notes):
                    current_notes = ""
                
                new_notes = st.text_area(
                    "Annotator Notes",
                    value=current_notes,
                    height=100,
                    key=f"notes_{original_idx}"
                )
                
                # Save changes
                if st.button("💾 Save Changes", key=f"save_{original_idx}"):
                    df.loc[original_idx, "human_label"] = new_human_label
                    df.loc[original_idx, "annotator_notes"] = new_notes
                    save_annotation_data(df)
                    st.success("✅ Changes saved successfully!")
                    st.rerun()
            
            with col2:
                st.subheader("📊 Record Info")
                st.write(f"**Title:** {selected_record.get('title', 'N/A')}")
                st.write(f"**Model Label:** {selected_record.get('label', 'N/A')}")
                st.write(f"**Confidence:** {selected_record.get('confidence', 'N/A'):.3f}")
                st.write(f"**Review Reason:** {selected_record.get('review_reason', 'N/A')}")
                
                if "rationale" in selected_record:
                    st.write("**Model Rationale:**")
                    st.write(selected_record.get("rationale", "N/A"))

# Display summary table
st.subheader("📈 Summary Table")

# Create summary columns
display_columns = ["title", "label", "human_label", "confidence", "review_reason"]
available_columns = [col for col in display_columns if col in filtered_df.columns]

if available_columns:
    summary_df = filtered_df[available_columns].copy()
    
    # Truncate title for display
    if "title" in summary_df.columns:
        summary_df["title"] = summary_df["title"].str[:50] + "..."
    
    # Format confidence
    if "confidence" in summary_df.columns:
        summary_df["confidence"] = summary_df["confidence"].round(3)
    
    st.dataframe(
        summary_df,
        use_container_width=True,
        height=400
    )

# Export functionality
st.subheader("📤 Export Data")
col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Export Filtered Data as CSV"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="filtered_annotations.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📋 Export All Data as JSON"):
        json_str = df.to_json(orient="records", indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="all_annotations.json",
            mime="application/json"
        )