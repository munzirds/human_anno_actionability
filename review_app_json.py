import streamlit as st
import pandas as pd
import json
import os

# -----------------------------
# Configuration
# -----------------------------
INPUT_JSON = "human_review_queue.json"
OUTPUT_JSON = "reviewed_output.json"

LABELS = ["A0", "A1", "A2", "A3"]

st.set_page_config(
    page_title="Crisis Actionability Review",
    layout="wide",
    initial_sidebar_state="auto"
)

# Navigation
page = st.sidebar.selectbox(
    "📍 Navigate",
    ["🔍 Review Interface", "📊 View Results"]
)

# Load reviewed data
output_path = os.path.join(os.path.dirname(__file__), OUTPUT_JSON)
with open(output_path, 'r', encoding='utf-8') as f:
    reviewed_data = json.load(f)
reviewed_df = pd.DataFrame(reviewed_data)

if page == "📊 View Results":
    st.title("📊 Annotation Results Viewer")
    
    # Filters
    st.sidebar.header("🔍 Filters")
    annotation_status = st.sidebar.selectbox("Status", ["All", "Annotated", "Not Annotated"])
    
    # Additional filters
    review_reasons = ["All"] + sorted(reviewed_df["review_reason"].unique().tolist())
    selected_reason = st.sidebar.selectbox("Review Reason", review_reasons)
    
    min_conf, max_conf = st.sidebar.slider(
        "Confidence", 0.0, 1.0, (0.0, 1.0), 0.01
    )
    
    model_labels = ["All"] + sorted(reviewed_df["label"].unique().tolist())
    selected_model_label = st.sidebar.selectbox("Model Label", model_labels)
    
    human_labels = ["All"] + [l for l in LABELS if l in reviewed_df["human_label"].values]
    selected_human_label = st.sidebar.selectbox("Human Label", human_labels)
    
    # Apply filters
    filtered_df = reviewed_df.copy()
    
    if annotation_status == "Annotated":
        filtered_df = filtered_df[filtered_df["human_label"] != ""]
    elif annotation_status == "Not Annotated":
        filtered_df = filtered_df[filtered_df["human_label"] == ""]
    
    if selected_reason != "All":
        filtered_df = filtered_df[filtered_df["review_reason"] == selected_reason]
    
    filtered_df = filtered_df[
        (filtered_df["confidence"] >= min_conf) & 
        (filtered_df["confidence"] <= max_conf)
    ]
    
    if selected_model_label != "All":
        filtered_df = filtered_df[filtered_df["label"] == selected_model_label]
    
    if selected_human_label != "All":
        filtered_df = filtered_df[filtered_df["human_label"] == selected_human_label]
    
    # Summary
    completed = reviewed_df[reviewed_df["human_label"] != ""]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total", len(reviewed_df))
    with col2:
        st.metric("Annotated", len(completed))
    with col3:
        st.metric("Filtered", len(filtered_df))
    
    # Record editor
    if len(filtered_df) > 0:
        record_idx = st.selectbox(
            "Select record:",
            range(len(filtered_df)),
            format_func=lambda x: f"Record {x+1}: {filtered_df.iloc[x].get('title', '')[:40]}..."
        )
        
        selected_record = filtered_df.iloc[record_idx]
        original_idx = filtered_df.index[record_idx]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.text_area("Message", selected_record["usertext"], height=150, disabled=True)
            
            # Edit annotation
            new_label = st.selectbox(
                "Human Label",
                [""] + LABELS,
                index=([""] + LABELS).index(selected_record.get("human_label", ""))
            )
            
            new_notes = st.text_area(
                "Notes",
                selected_record.get("annotator_notes", ""),
                height=80
            )
            
            if st.button("💾 Save Changes"):
                reviewed_df.loc[original_idx, "human_label"] = new_label
                reviewed_df.loc[original_idx, "annotator_notes"] = new_notes
                data = reviewed_df.to_dict('records')
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                st.success("✅ Saved!")
                st.rerun()
        
        with col2:
            st.write(f"**Model Label:** {selected_record['label']}")
            st.write(f"**Confidence:** {selected_record['confidence']:.3f}")
            st.write(f"**Reason:** {selected_record['review_reason']}")
    
    # Export
    if st.button("📊 Export CSV"):
        csv = filtered_df.to_csv(index=False)
        st.download_button("Download", csv, "annotations.csv", "text/csv")
    
    st.stop()

# -----------------------------
# Load data
# -----------------------------
def load_data():
    import os
    file_path = os.path.join(os.path.dirname(__file__), INPUT_JSON)
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        st.stop()
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

df = load_data()

# Initialize output file if not exists
output_path = os.path.join(os.path.dirname(__file__), OUTPUT_JSON)
if not os.path.exists(output_path):
    df["human_label"] = ""
    df["annotator_notes"] = ""
    data = df.to_dict('records')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Progress tracking
pending = reviewed_df[reviewed_df["human_label"] == ""]
completed = reviewed_df[reviewed_df["human_label"] != ""]

# Add CSS for sidebar animations
st.markdown("""
<style>
.sidebar-metric {
    animation: slideIn 0.6s ease-out;
}

@keyframes slideIn {
    from { transform: translateX(-20px); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

.stButton > button {
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

if page == "🔍 Review Interface":
    st.sidebar.title("📊 Review Progress")
# Responsive progress metrics
if st.sidebar.button("📱 Toggle Mobile View"):
    st.session_state['mobile_view'] = not st.session_state.get('mobile_view', False)
    st.rerun()

# Progress metrics - responsive layout
if len(str(len(reviewed_df))) > 3:  # Adjust layout for large numbers
    st.sidebar.metric("📋 Total", len(reviewed_df))
    st.sidebar.metric("✅ Done", len(completed))
    st.sidebar.metric("⏳ Left", len(pending))
else:
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        st.metric("📋", len(reviewed_df))
    with col2:
        st.metric("✅", len(completed))
    with col3:
        st.metric("⏳", len(pending))

# Progress bar
progress = len(completed) / len(reviewed_df) if len(reviewed_df) > 0 else 0
st.sidebar.progress(progress)
st.sidebar.caption(f"{progress:.1%} Complete")

# Reset option
if st.sidebar.button("Reset All Reviews"):
    reviewed_df["human_label"] = ""
    reviewed_df["annotator_notes"] = ""
    data = reviewed_df.to_dict('records')
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_JSON)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    st.rerun()

# -----------------------------
# Select next item
# -----------------------------
if len(pending) == 0:
    st.success("All reviews completed. Thank you!")
    st.info("Use 'Reset All Reviews' in the sidebar to start over.")
    st.stop()

current_idx = pending.index[0]
row = reviewed_df.loc[current_idx]

# -----------------------------
# Main UI
# -----------------------------
st.title("Human Review: Crisis Actionability")

# Add help section
with st.expander("📋 Annotation Guide - Click to expand"):
    st.markdown("""
    ### Purpose
    Classify **response urgency** for crisis-support messages. You are NOT diagnosing mental illness or predicting outcomes.
    
    ### General Principles
    - **Focus on urgency, not emotion** - Strong emotion ≠ immediate danger
    - **Assume no prior context** - Judge as if this is the first message received
    - **When uncertain, choose LOWER urgency** - Ambiguity should not inflate risk
    - **Protective factors matter** - Family, pets, future plans reduce urgency
    
    ### Quick Decision Checklist
    1. **Is there explicit intent or plan?** → A3
    2. **Is there desire to die but no plan?** → A2  
    3. **Is there distress with ambivalence or coping?** → A1
    4. **Is it emotional but safe?** → A0
    
    **If unsure:** Choose lower urgency and leave a note explaining why.
    """)

# Responsive layout - stack on mobile, side-by-side on desktop
if st.session_state.get('mobile_view', False) or 'mobile' in st.query_params.get('view', ''):
    # Mobile layout - stacked
    col1 = st.container()
    col2 = st.container()
else:
    # Desktop layout - side by side
    col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📝 User Message")
    
    # Mobile-responsive CSS
    st.markdown("""
    <style>
    /* Mobile-first responsive design */
    .message-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 4px solid #007bff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        animation: fadeIn 0.8s ease-in;
        width: 100%;
        box-sizing: border-box;
    }
    
    .message-text {
        font-family: 'Georgia', serif;
        font-size: 14px;
        line-height: 1.6;
        color: #2c3e50;
        text-align: left;
        position: relative;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    .quote-mark {
        font-size: 32px;
        color: #007bff;
        opacity: 0.3;
        font-family: 'Times New Roman', serif;
        position: absolute;
    }
    
    .quote-left {
        top: -5px;
        left: -5px;
    }
    
    .quote-right {
        bottom: -20px;
        right: 5px;
        transform: rotate(180deg);
    }
    
    /* Desktop styles */
    @media (min-width: 768px) {
        .message-text {
            font-size: 16px;
            text-align: justify;
        }
        .message-container {
            padding: 20px;
        }
        .quote-mark {
            font-size: 48px;
        }
        .quote-left {
            top: -10px;
            left: -10px;
        }
        .quote-right {
            bottom: -30px;
            right: 10px;
        }
    }
    
    /* Streamlit specific mobile adjustments */
    @media (max-width: 767px) {
        .stButton > button {
            width: 100% !important;
            margin: 5px 0 !important;
        }
        .stRadio > div {
            flex-direction: column !important;
        }
        .stTextArea textarea {
            min-height: 100px !important;
        }
        .element-container {
            margin-bottom: 1rem !important;
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .message-container:hover {
        animation: pulse 0.5s ease-in-out;
        box-shadow: 0 4px 20px rgba(0,123,255,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display message with custom styling
    message_html = f"""
    <div class="message-container">
        <div class="message-text">
            <span class="quote-mark quote-left">"</span>
            {row["usertext"].replace('\n', '<br>')}
            <span class="quote-mark quote-right">"</span>
        </div>
    </div>
    """
    
    st.markdown(message_html, unsafe_allow_html=True)
    
    # Annotation controls right after message
    st.markdown("### Your Judgment")
    
    label_help = {
        "A0": "Non-Actionable - General sadness/frustration, no death mentions, advice posts. No immediate response needed.",
        "A1": "Monitoring - Passive distress, fleeting death thoughts without intent, ambivalence, protective factors present. Monitor or delayed review.",
        "A2": "Prompt Action - Repeated desire to die, worthlessness, past attempts without current plan. Prompt human intervention needed.",
        "A3": "Critical/Imminent - Explicit plans, means/method/timing mentioned, ongoing attempts, language of finality. Immediate escalation required."
    }
    
    human_label = st.radio(
        "Select actionability level:",
        LABELS,
        index=LABELS.index(row["label"]) if row["label"] in LABELS else 0,
        format_func=lambda x: f"{x}: {label_help[x]}",
        help="Choose the urgency level based on the content above. When uncertain, choose the LOWER urgency level."
    )
    
    notes = st.text_area(
        "Annotator notes (optional):",
        value=row["annotator_notes"],
        height=80,
        help="Explain your reasoning, especially if you're uncertain or see borderline indicators."
    )

with col2:
    st.subheader("Model Prediction")
    st.write(f"**Label:** {row['label']}")
    st.write(f"**Confidence:** {row['confidence']:.2f}")
    st.write(f"**Reason:** {row['review_reason']}")
    
    with st.expander("⚠️ Borderline Cases"):
        st.markdown("""
        **Choose A1 instead of A2 if:**
        - Person expresses fear of dying
        - Mentions wanting help  
        - Emphasizes protective factors
        
        **Choose A2 instead of A3 if:**
        - Suicidal ideation but no plan, means, or timing
        - Message reflects rumination rather than intent
        
        **Choose A3 only when:**
        - Risk is explicit, concrete, or imminent
        - Clear plan, means, method, or timing mentioned
        """)

st.markdown("---")

# Save action
if st.session_state.get('mobile_view', False):
    # Mobile: stacked buttons
    if st.button("💾 Save & Next", type="primary", use_container_width=True):
        reviewed_df.loc[current_idx, "human_label"] = human_label
        reviewed_df.loc[current_idx, "annotator_notes"] = notes
        data = reviewed_df.to_dict('records')
        output_path = os.path.join(os.path.dirname(__file__), OUTPUT_JSON)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        st.balloons()
        st.success(f"✅ Saved as {human_label}! Moving to next item...")
        st.rerun()
    
    if st.button("⏭️ Skip", use_container_width=True):
        st.rerun()
else:
    # Desktop: side by side
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save & Next", type="primary", use_container_width=True):
            reviewed_df.loc[current_idx, "human_label"] = human_label
            reviewed_df.loc[current_idx, "annotator_notes"] = notes
            data = reviewed_df.to_dict('records')
            output_path = os.path.join(os.path.dirname(__file__), OUTPUT_JSON)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            st.balloons()
            st.success(f"✅ Saved as {human_label}! Moving to next item...")
            st.rerun()
    
    with col2:
        if st.button("⏭️ Skip", use_container_width=True):
            st.rerun()