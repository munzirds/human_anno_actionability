# Review App

A Python application for managing review queues and processing review data with annotation viewing and editing capabilities.

## Files
- `review_app_json.py` - Original annotation interface
- `annotation_viewer.py` - Standalone annotation results viewer
- `integrated_app.py` - **Recommended** - Combined interface with review, viewing, and analytics
- `human_review_queue.json` - Queue data
- `reviewed_output.json` - Processed review results
- `run_app.bat` - Run original annotation interface
- `run_viewer.bat` - Run standalone viewer
- `run_integrated_app.bat` - **Recommended** - Run integrated application

## Features

### 🔍 Review Interface
- Annotate messages with actionability levels (A0-A3)
- Progress tracking
- Skip and save functionality
- Mobile-responsive design

### 📊 Results Viewer
- Filter by annotation status (annotated/not annotated)
- Filter by review reason, confidence, labels
- Edit existing annotations
- Export data as CSV/JSON

### 📈 Analytics
- Label distribution charts
- Model vs human agreement metrics
- Confusion matrix analysis

## Usage

### Recommended: Integrated Application
```bash
run_integrated_app.bat
```

### Alternative: Separate Applications
```bash
# Run annotation interface
run_app.bat

# Run results viewer (in separate terminal)
run_viewer.bat
```

## Navigation
The integrated app includes three modes:
1. **🔍 Review Interface** - Annotate new records
2. **📊 View Results** - Browse and edit existing annotations
3. **📈 Analytics** - View annotation statistics and model performance