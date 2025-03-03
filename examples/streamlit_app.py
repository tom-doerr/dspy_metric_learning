"""
Streamlit app for managing and optimizing metric functions.

This app provides a user interface for:
1. Viewing and labeling instances
2. Optimizing metric functions
3. Testing optimized metrics on new examples
"""

import sys
import os
import streamlit as st
import pandas as pd
from datetime import datetime

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metric_learner import (
    MetricModule,
    MetricDataManager,
    optimize_metric_module
)

# Mock LM for demonstration
class MockLM:
    def __call__(self, prompt):
        st.session_state.last_prompt = prompt
        # In a real scenario, this would call the actual LM
        return "0.75"

# Initialize session state
if 'metric_modules' not in st.session_state:
    lm = MockLM()
    st.session_state.lm = lm
    st.session_state.metric_modules = {
        'accuracy': MetricModule(lm=lm, prompt_template="Rate the factual accuracy of '{prediction}' for '{input}' from 0 to 1."),
        'fluency': MetricModule(lm=lm, prompt_template="Rate the fluency of '{prediction}' from 0 to 1."),
        'relevance': MetricModule(lm=lm, prompt_template="Rate the relevance of '{prediction}' to '{input}' from 0 to 1.")
    }
    st.session_state.data_managers = {
        name: MetricDataManager(metric_name=name) 
        for name in st.session_state.metric_modules
    }
    st.session_state.optimized_modules = {}
    st.session_state.last_prompt = ""

# App title
st.title("DSPy Metric Learning")
st.subheader("Manage and optimize metric functions")

# Sidebar for navigation
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Label Instances", "Optimize Metrics", "Test Metrics"]
)

# Select metric
selected_metric = st.sidebar.selectbox(
    "Select Metric",
    list(st.session_state.metric_modules.keys())
)

# Home page
if page == "Home":
    st.markdown("""
    ## Welcome to DSPy Metric Learning
    
    This app demonstrates how to use the DSPy Metric Learning package to:
    
    1. **Label Instances**: View and score predictions
    2. **Optimize Metrics**: Train metric functions using labeled data
    3. **Test Metrics**: Try optimized metrics on new examples
    
    Select a page from the sidebar to get started.
    """)
    
    # Display metrics statistics
    st.subheader("Metrics Statistics")
    
    stats = []
    for name, dm in st.session_state.data_managers.items():
        instances = dm.load_instances()
        labeled = [i for i in instances if i.get("user_score") is not None]
        stats.append({
            "Metric": name,
            "Total Instances": len(instances),
            "Labeled Instances": len(labeled),
            "Optimized": name in st.session_state.optimized_modules
        })
    
    if stats:
        st.table(pd.DataFrame(stats))
    else:
        st.info("No metrics data available yet.")

# Label Instances page
elif page == "Label Instances":
    st.subheader(f"Label Instances for {selected_metric}")
    
    data_manager = st.session_state.data_managers[selected_metric]
    instances = data_manager.load_instances()
    unlabeled = [i for i in instances if i.get("user_score") is None]
    
    if not instances:
        st.info("No instances available. Add some examples first.")
        
        # Form to add new examples
        with st.form("add_example"):
            st.subheader("Add New Example")
            input_text = st.text_area("Input (e.g., question):")
            prediction = st.text_area("Prediction (e.g., answer):")
            gold = st.text_area("Gold Standard (optional):", "")
            
            submitted = st.form_submit_button("Score and Add")
            if submitted and input_text and prediction:
                # Score with the selected metric
                metric_module = st.session_state.metric_modules[selected_metric]
                score = metric_module(input_text, prediction, gold=gold if gold else None)
                
                # Save the instance
                data_manager.save_instance(
                    input_text,
                    prediction,
                    gold=gold if gold else None,
                    score=score
                )
                
                st.success("Example added successfully!")
                st.experimental_rerun()
    
    elif not unlabeled:
        st.success("All instances are labeled! Go to the Optimize page to train your metric.")
        
        # Show labeled instances
        st.subheader("Labeled Instances")
        for idx, instance in enumerate(instances):
            with st.expander(f"Instance {idx+1}: {instance['input'][:50]}..."):
                st.text(f"Input: {instance['input']}")
                st.text(f"Prediction: {instance['prediction']}")
                if instance.get("gold"):
                    st.text(f"Gold: {instance['gold']}")
                st.text(f"Model Score: {instance.get('score')}")
                st.text(f"User Score: {instance.get('user_score')}")
    
    else:
        # Show unlabeled instance for labeling
        instance = unlabeled[0]
        
        st.text(f"Input: {instance['input']}")
        st.text(f"Prediction: {instance['prediction']}")
        if instance.get("gold"):
            st.text(f"Gold: {instance['gold']}")
        st.text(f"Model Score: {instance.get('score')}")
        
        # Score input
        user_score = st.slider("Your Score:", 0.0, 1.0, 0.5, 0.01)
        if st.button("Submit Score"):
            data_manager.update_user_score(instance["datetime"], user_score)
            st.success("Score submitted!")
            st.experimental_rerun()
        
        # Skip button
        if st.button("Skip"):
            st.experimental_rerun()
        
        # Progress
        st.progress(len(instances) - len(unlabeled) / max(1, len(instances)))
        st.text(f"Labeled {len(instances) - len(unlabeled)}/{len(instances)} instances")

# Optimize Metrics page
elif page == "Optimize Metrics":
    st.subheader(f"Optimize {selected_metric} Metric")
    
    data_manager = st.session_state.data_managers[selected_metric]
    dataset = data_manager.get_labeled_dataset()
    
    if not dataset:
        st.warning("No labeled data available for optimization. Please label some instances first.")
    else:
        st.info(f"Found {len(dataset)} labeled examples for optimization.")
        
        if st.button("Optimize Metric"):
            with st.spinner("Optimizing metric..."):
                # Get the metric module
                metric_module = st.session_state.metric_modules[selected_metric]
                
                # Optimize
                optimized_module = optimize_metric_module(metric_module, dataset)
                
                # Store the optimized module
                st.session_state.optimized_modules[selected_metric] = optimized_module
                
                st.success("Optimization complete!")
        
        # Show optimization status
        if selected_metric in st.session_state.optimized_modules:
            st.success(f"The {selected_metric} metric has been optimized.")
        
        # Show labeled data
        st.subheader("Labeled Data")
        data = []
        for example in dataset:
            data.append({
                "Input": example.input[:50] + "..." if len(example.input) > 50 else example.input,
                "Prediction": example.prediction[:50] + "..." if len(example.prediction) > 50 else example.prediction,
                "User Score": example.user_score
            })
        
        if data:
            st.table(pd.DataFrame(data))

# Test Metrics page
elif page == "Test Metrics":
    st.subheader(f"Test {selected_metric} Metric")
    
    # Check if the metric is optimized
    is_optimized = selected_metric in st.session_state.optimized_modules
    
    # Input form
    with st.form("test_form"):
        input_text = st.text_area("Input (e.g., question):")
        prediction = st.text_area("Prediction (e.g., answer):")
        gold = st.text_area("Gold Standard (optional):", "")
        
        submitted = st.form_submit_button("Score")
    
    if submitted and input_text and prediction:
        # Get the metric modules
        original_module = st.session_state.metric_modules[selected_metric]
        optimized_module = st.session_state.optimized_modules.get(selected_metric)
        
        # Score with original metric
        original_score = original_module(
            input_text, 
            prediction, 
            gold=gold if gold else None
        )
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Metric")
            st.metric("Score", f"{original_score:.2f}")
        
        with col2:
            st.subheader("Optimized Metric")
            if is_optimized:
                # Score with optimized metric
                optimized_score = optimized_module(
                    input_text, 
                    prediction, 
                    gold=gold if gold else None
                )
                st.metric("Score", f"{optimized_score:.2f}", 
                          delta=f"{optimized_score - original_score:.2f}")
            else:
                st.warning("Metric not optimized yet")
        
        # Show the prompt used
        with st.expander("Show LM Prompt"):
            st.code(st.session_state.last_prompt)
        
        # Option to save this example
        if st.button("Save this example"):
            data_manager = st.session_state.data_managers[selected_metric]
            data_manager.save_instance(
                input_text,
                prediction,
                gold=gold if gold else None,
                score=original_score
            )
            st.success("Example saved!")

# Run the app with: streamlit run examples/streamlit_app.py --server.headless=true
