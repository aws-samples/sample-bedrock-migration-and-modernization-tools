import streamlit as st
import sys
import logging
import os

# Add the project root to path to allow importing dashboard modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Configure logging
os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
# Use in-memory logging instead of file-based logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Suppress annoying Streamlit thread warnings
class StreamlitThreadWarningFilter(logging.Filter):
    """Filter to suppress 'missing ScriptRunContext' warnings from background threads."""
    def filter(self, record):
        msg = record.getMessage()
        # Filter out the specific warning about missing ScriptRunContext
        if "missing ScriptRunContext" in msg:
            return False
        return True

# Apply filter to all streamlit-related loggers
warning_filter = StreamlitThreadWarningFilter()
for logger_name in ['streamlit', 'streamlit.runtime', 'streamlit.runtime.scriptrunner']:
    logger_obj = logging.getLogger(logger_name)
    logger_obj.addFilter(warning_filter)

logger = logging.getLogger('streamlit_dashboard')
logger.info("Starting Streamlit dashboard")

# Import dashboard components
from src.dashboard.components.evaluation_setup import EvaluationSetupComponent
from src.dashboard.components.model_configuration import ModelConfigurationComponent
from src.dashboard.components.evaluation_monitor import EvaluationMonitorComponent
from src.dashboard.components.results_viewer import ResultsViewerComponent
from src.dashboard.components.report_viewer import ReportViewerComponent
from src.dashboard.components.unprocessed_viewer import UnprocessedRecordsViewer
from src.dashboard.utils.state_management import initialize_session_state
from src.dashboard.utils.constants import APP_TITLE, SIDEBAR_INFO, PROJECT_ROOT
from src.dashboard.utils.process_manager import cleanup_stale_processes

# Initialize session state at module level to ensure it's available before component rendering
if "evaluations" not in st.session_state:
    initialize_session_state()

# Cleanup stale processes from previous runs (only once per session)
if "process_cleanup_done" not in st.session_state:
    logger.info("Cleaning up stale processes from previous runs...")
    stats = cleanup_stale_processes()
    logger.info(f"Process cleanup stats: {stats}")
    st.session_state.process_cleanup_done = True
    
# Debug session state
print("Session state initialized at module level:")
print(f"Evaluations: {len(st.session_state.evaluations)}")
print(f"Active evaluations: {len(st.session_state.active_evaluations)}")
print(f"Completed evaluations: {len(st.session_state.completed_evaluations)}")

def main():
    """Main Streamlit dashboard application."""
    try:
        # Set page title and layout with custom icon
        logger.info("Initializing Streamlit dashboard")
        
        icon_path = os.path.join(PROJECT_ROOT, "assets", "scale_icon.png")
        
        st.set_page_config(
            page_title=APP_TITLE,
            page_icon=icon_path,
            layout="wide"
        )
        
        # Initialize session state again to ensure all variables are set
        initialize_session_state()
        logger.info("Session state initialized")
        
        # Display log file path for debugging
        log_dir = os.path.join(PROJECT_ROOT, 'logs')
        
        # Add log information to sidebar
        with st.sidebar:
            with st.expander("📋 Debug Information"):
                st.info(f"Log Directory: {log_dir}")
                st.info(f"Project Root: {PROJECT_ROOT}")
                # Add button to show latest logs
                if st.button("Show Latest Logs"):
                    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
                    if log_files:
                        latest_log = max(log_files, key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
                        with open(os.path.join(log_dir, latest_log), 'r') as f:
                            log_content = f.read()
                        st.text_area("Latest Log Entries", log_content[-5000:], height=300)
                    else:
                        st.warning("No log files found")
        
        # Header with logo and title
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.image(icon_path, width=150)
        
        with col2:
            st.title(APP_TITLE)
            st.markdown("Create, manage, and visualize LLM benchmark evaluations using LLM-as-a-JURY methodology")
        
        # Sidebar with info
        with st.sidebar:
            st.markdown(SIDEBAR_INFO)
            st.divider()

            # Evaluation Mode Selector
            st.markdown("### 🎯 Evaluation Mode")

            # Get current evaluation type from config to preserve selection
            current_eval_type = st.session_state.current_evaluation_config.get("evaluation_type", "llm")
            default_index = 1 if current_eval_type == "rag" else 0

            eval_mode = st.radio(
                "Select Evaluation Type",
                ["LLM Evaluation", "RAG Evaluation"],
                index=default_index,
                key="evaluation_mode_selector",
                help="Choose between LLM response evaluation or RAG retrieval evaluation"
            )

            # Update session state
            st.session_state.evaluation_mode = eval_mode

            # Set evaluation type in config
            if eval_mode == "RAG Evaluation":
                st.session_state.current_evaluation_config["evaluation_type"] = "rag"
            else:
                st.session_state.current_evaluation_config["evaluation_type"] = "llm"

            st.divider()

            # Navigation tabs in sidebar - include Unprocessed tab
            tab_names = ["Setup", "Monitor", "Evaluations", "Reports", "Unprocessed"]

            # Check if we need to navigate to Setup tab
            if "navigate_to_setup" in st.session_state and st.session_state.navigate_to_setup:
                st.session_state.nav_radio = "Setup"
                del st.session_state.navigate_to_setup

            active_tab = st.radio("Navigation", tab_names, key="nav_radio")
            logger.info(f"Selected tab: {active_tab}")
        
        # Main area - show different components based on active tab
        if active_tab == "Setup":
            # Conditional rendering based on evaluation mode
            if st.session_state.evaluation_mode == "RAG Evaluation":
                # RAG Evaluation Setup
                st.info("🚀 RAG Evaluation Mode - Configure your retrieval evaluation settings")

                try:
                    from src.dashboard.components.rag_evaluation_setup import RAGEvaluationSetupComponent
                    RAGEvaluationSetupComponent().render()
                except ImportError as e:
                    st.error(f"RAG Evaluation component not found: {str(e)}")
                    st.info("RAG evaluation UI is under development. Use CLI for now.")
                    st.code("""
# Example CLI usage for RAG evaluation:
python src/rag_benchmarks_run.py \\
    sample_rag_queries.csv \\
    sample_rag_datasource.txt \\
    --data_format txt \\
    --chunking_strategy recursive \\
    --chunk_size 512 \\
    --top_k 5
                    """, language="bash")

            else:
                # LLM Evaluation Setup (existing functionality)
                setup_tab1, setup_tab2, setup_tab3 = st.tabs(["Evaluation Setup", "Model Configuration", "Advanced Configuration"])

                with setup_tab1:
                    logger.info("Rendering Evaluation Setup component")
                    EvaluationSetupComponent().render()

                with setup_tab2:
                    logger.info("Rendering Model Configuration component")
                    ModelConfigurationComponent().render()

                with setup_tab3:
                    logger.info("Rendering Advanced Configuration component")
                    EvaluationSetupComponent().render_advanced_config()
                
        elif active_tab == "Monitor":
            logger.info("Rendering Evaluation Monitor component")
            EvaluationMonitorComponent().render()
            
        elif active_tab == "Evaluations":
            logger.info("Rendering Results Viewer component")
            ResultsViewerComponent().render()

        elif active_tab == "Unprocessed":
            logger.info("Rendering Unprocessed Records Viewer component")
            UnprocessedRecordsViewer().render()

        elif active_tab == "Reports":
            logger.info("Rendering Report Viewer component")
            ReportViewerComponent().render()
            
    except Exception as e:
        logger.exception(f"Unhandled exception in main dashboard: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        st.info(f"Check logs for details at: {log_dir}")

if __name__ == "__main__":
    main()