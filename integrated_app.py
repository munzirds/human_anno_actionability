import streamlit as st
import pandas as pd
import json
import os

# Configuration
INPUT_JSON = "human_review_queue.json"
OUTPUT_JSON = "reviewed_output.json"
LABELS = ["A0", "A1", "A2", "A3"]

st.set_page_config(
    page_title="Crisis Actionability Review System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data functions
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), INPUT_JSON)
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        st.stop()
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

@st.cache_data
def load_annotation_data():
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_JSON)
    if not os.path.exists(output_path):
        return pd.DataFrame()
    
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def save_annotation_data(df):
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_JSON)
    data = df.to_dict('records')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Initialize data
df = load_data()

# Initialize output file if not exists
output_path = os.path.join(os.path.dirname(__file__), OUTPUT_JSON)
if not os.path.exists(output_path):
    df["human_label"] = ""
    df["annotator_notes"] = ""
    data = df.to_dict('records')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Load reviewed data
reviewed_df = load_annotation_data()
if reviewed_df.empty:
    reviewed_df = df.copy()
    reviewed_df["human_label"] = ""
    reviewed_df["annotator_notes"] = ""

# Navigation
st.sidebar.title("🏠 Navigation")
page = st.sidebar.radio(
    "Select Mode:",
    ["🔍 Review Interface", "📊 View Results", "📈 Analytics"]
)

if page == "🔍 Review Interface":
    # Original review interface
    st.title("Human Review: Crisis Actionability")
    
    # Progress tracking
    pending = reviewed_df[reviewed_df["human_label"] == ""]
    completed = reviewed_df[reviewed_df["human_label"] != ""]
    
    st.sidebar.title("📊 Review Progress")
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        st.metric("📋", len(reviewed_df))
    with col2:
        st.metric("✅", len(completed))
    with col3:
        st.metric("⏳", len(pending))
    
    progress = len(completed) / len(reviewed_df) if len(reviewed_df) > 0 else 0
    st.sidebar.progress(progress)
    st.sidebar.caption(f"{progress:.1%} Complete")
    
    if len(pending) == 0:
        st.success("All reviews completed!")
        st.stop()
    
    current_idx = pending.index[0]
    row = reviewed_df.loc[current_idx]
    
    # Display message
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📝 User Message")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                    border-left: 4px solid #007bff; border-radius: 10px; padding: 20px; margin: 10px 0;">
            <div style="font-family: 'Georgia', serif; font-size: 16px; line-height: 1.6; color: #2c3e50;">
                {row["usertext"].replace(chr(10), '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Model Prediction")
        st.write(f"**Label:** {row['label']}")
        st.write(f"**Confidence:** {row['confidence']:.2f}")
        st.write(f"**Reason:** {row['review_reason']}")
    
    # Annotation controls
    st.markdown("### Your Judgment")
    
    human_label = st.radio(
        "Select actionability level:",
        LABELS,
        index=LABELS.index(row["label"]) if row["label"] in LABELS else 0,
        horizontal=True
    )
    
    notes = st.text_area(
        "Annotator notes (optional):",
        value=row["annotator_notes"],
        height=100
    )
    
    # Save action
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save & Next", type="primary", use_container_width=True):
            reviewed_df.loc[current_idx, "human_label"] = human_label
            reviewed_df.loc[current_idx, "annotator_notes"] = notes
            save_annotation_data(reviewed_df)
            st.success(f"✅ Saved as {human_label}!")
            st.rerun()
    
    with col2:
        if st.button("⏭️ Skip", use_container_width=True):
            st.rerun()

elif page == "📊 View Results":
    st.title("📊 Annotation Results Viewer")
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Annotation status filter
    annotation_status = st.sidebar.selectbox(
        "Annotation Status",
        ["All", "Annotated", "Not Annotated"]
    )
    
    # Filter by annotation status
    if annotation_status == "Annotated":
        filtered_df = reviewed_df[reviewed_df["human_label"].notna() & (reviewed_df["human_label"] != "")]
    elif annotation_status == "Not Annotated":
        filtered_df = reviewed_df[reviewed_df["human_label"].isna() | (reviewed_df["human_label"] == "")]
    else:
        filtered_df = reviewed_df.copy()
    
    # Additional filters
    if "review_reason" in reviewed_df.columns:
        review_reasons = ["All"] + sorted(reviewed_df["review_reason"].dropna().unique().tolist())
        selected_reason = st.sidebar.selectbox("Review Reason", review_reasons)
        if selected_reason != "All":
            filtered_df = filtered_df[filtered_df["review_reason"] == selected_reason]
    
    if "confidence" in reviewed_df.columns:
        min_conf, max_conf = st.sidebar.slider(
            "Confidence Range",
            min_value=float(reviewed_df["confidence"].min()),
            max_value=float(reviewed_df["confidence"].max()),
            value=(float(reviewed_df["confidence"].min()), float(reviewed_df["confidence"].max())),
            step=0.01
        )
        filtered_df = filtered_df[
            (filtered_df["confidence"] >= min_conf) & 
            (filtered_df["confidence"] <= max_conf)
        ]
    
    # Display summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(reviewed_df))
    with col2:
        annotated_count = len(reviewed_df[reviewed_df["human_label"].notna() & (reviewed_df["human_label"] != "")])
        st.metric("Annotated", annotated_count)
    with col3:
        st.metric("Filtered Results", len(filtered_df))
    with col4:
        if len(reviewed_df) > 0:
            progress = annotated_count / len(reviewed_df)
            st.metric("Progress", f"{progress:.1%}")
    
    st.markdown("---")
    
    # Record editing
    if len(filtered_df) > 0:
        st.subheader("✏️ Edit Annotations")
        
        record_options = [f"Record {i+1}: {row.get('title', 'No title')[:50]}..." 
                         for i, row in filtered_df.iterrows()]
        
        selected_idx = st.selectbox(
            "Select record to edit:",
            range(len(filtered_df)),
            format_func=lambda x: record_options[x]
        )
        
        if selected_idx is not None:
            selected_record = filtered_df.iloc[selected_idx]
            original_idx = filtered_df.index[selected_idx]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.text_area(
                    "User Text",
                    value=selected_record.get("usertext", ""),
                    height=150,
                    disabled=True
                )
                
                # Edit controls
                current_human_label = selected_record.get("human_label", "")
                if pd.isna(current_human_label):
                    current_human_label = ""
                
                label_options = [""] + LABELS
                new_human_label = st.selectbox(
                    "Human Label",
                    label_options,
                    index=label_options.index(current_human_label) if current_human_label in label_options else 0
                )
                
                current_notes = selected_record.get("annotator_notes", "")
                if pd.isna(current_notes):
                    current_notes = ""
                
                new_notes = st.text_area(
                    "Annotator Notes",
                    value=current_notes,
                    height=80
                )
                
                if st.button("💾 Save Changes"):
                    reviewed_df.loc[original_idx, "human_label"] = new_human_label
                    reviewed_df.loc[original_idx, "annotator_notes"] = new_notes
                    save_annotation_data(reviewed_df)
                    st.success("✅ Changes saved!")
                    st.rerun()
            
            with col2:
                st.write(f"**Title:** {selected_record.get('title', 'N/A')}")
                st.write(f"**Model Label:** {selected_record.get('label', 'N/A')}")
                st.write(f"**Confidence:** {selected_record.get('confidence', 'N/A'):.3f}")
                st.write(f"**Review Reason:** {selected_record.get('review_reason', 'N/A')}")
    
    # Summary table
    st.subheader("📋 Summary Table")
    display_columns = ["title", "label", "human_label", "confidence", "review_reason"]
    available_columns = [col for col in display_columns if col in filtered_df.columns]
    
    if available_columns and len(filtered_df) > 0:
        summary_df = filtered_df[available_columns].copy()
        if "title" in summary_df.columns:
            summary_df["title"] = summary_df["title"].str[:40] + "..."
        if "confidence" in summary_df.columns:
            summary_df["confidence"] = summary_df["confidence"].round(3)
        
        st.dataframe(summary_df, use_container_width=True, height=300)

elif page == "📈 Analytics":
    st.title("📈 Annotation Analytics")
    
    annotated_df = reviewed_df[reviewed_df["human_label"].notna() & (reviewed_df["human_label"] != "")]
    
    if len(annotated_df) == 0:
        st.info("No annotations available for analysis.")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Label Distribution")
        if "human_label" in annotated_df.columns:
            label_counts = annotated_df["human_label"].value_counts()
            st.bar_chart(label_counts)
    
    with col2:
        st.subheader("Model vs Human Agreement")
        if "label" in annotated_df.columns and "human_label" in annotated_df.columns:
            agreement = (annotated_df["label"] == annotated_df["human_label"]).mean()
            st.metric("Agreement Rate", f"{agreement:.1%}")
            
            # Confusion matrix
            confusion_data = pd.crosstab(
                annotated_df["label"], 
                annotated_df["human_label"], 
                margins=True
            )
            st.write("**Confusion Matrix:**")
            st.dataframe(confusion_data)
    
    # Export functionality
    st.subheader("📤 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export as CSV"):
            csv = reviewed_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="annotations.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📋 Export as JSON"):
            json_str = reviewed_df.to_json(orient="records", indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="annotations.json",
                mime="application/json"
            )