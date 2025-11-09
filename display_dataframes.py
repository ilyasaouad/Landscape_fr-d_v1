import streamlit as st
import pandas as pd
from pathlib import Path

# ---------------------------------
# Page configuration
# ---------------------------------
st.set_page_config(page_title="Patent Data Viewer", page_icon="📊", layout="wide")

# ---------------------------------
# Title
# ---------------------------------
st.title("📊 Patent Applicants & Inventors Data Viewer")

# ---------------------------------
# Specific file path
# ---------------------------------
# analyse_3 = r"C:\Users\iao\Desktop\Landscape_Fråd_v1\Analysis_NO_2020_2020\01_raw_data\applicants_inventors_data.csv"

#analyse_3 = (
#    r"C:\Users\iao\Desktop\Landscape_Fråd_v1\output\G06N10\main_G06N10_table.csv"
#)

#analyse_3 = r"C:\Users\iao\Desktop\Landscape_Fråd_v1\DataTables_NO_2020_2020\main_table.csv"
analyse_3= r"C:\Users\iao\Desktop\Landscape_Fråd_v1\DataTables_NO_2020_2020\main_table_agg.csv"


# ---------------------------------
# Load and display data
# ---------------------------------
def load_csv_from_path(path_str: str):
    """Load CSV from string path"""
    path = Path(path_str)
    if path.exists():
        return pd.read_csv(path)
    else:
        return None


# ---------------------------------
# Main content
# ---------------------------------
st.markdown("### 🔍 Additional Analysis Data (Raw Applicants/Inventors)")
st.markdown(f"**File:** `{analyse_3}`")

df_analyse_3 = load_csv_from_path(analyse_3)

if df_analyse_3 is not None:
    st.dataframe(df_analyse_3, use_container_width=True, height=600)

    # Basic info
    st.info(
        f"**Records:** {len(df_analyse_3):,} | **Columns:** {len(df_analyse_3.columns)}"
    )

    # Show column names
    st.markdown("#### 📝 Column Names")
    for i, col in enumerate(df_analyse_3.columns, 1):
        st.write(f"{i}. {col}")

else:
    st.error("❌ File not found. Please check the file path.")

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
st.caption("Patent Data Viewer | Enhanced Analysis Data Display")
