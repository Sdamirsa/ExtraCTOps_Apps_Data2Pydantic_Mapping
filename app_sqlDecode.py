#!/usr/bin/env python3
"""
SQL Database Decoder and Merger
A Streamlit application for merging multiple SQL databases (CSV format) 
and decoding observation/measurement codes using dictionary files.
"""

import streamlit as st
import pandas as pd
import io
from typing import Dict, List, Tuple, Optional

# Configure page
st.set_page_config(
    page_title="SQL Database Decoder & Merger",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """Initialize session state variables"""
    if "uploaded_sql_files" not in st.session_state:
        st.session_state.uploaded_sql_files = []
    if "uploaded_dict_files" not in st.session_state:
        st.session_state.uploaded_dict_files = []
    if "sql_dict_mappings" not in st.session_state:
        st.session_state.sql_dict_mappings = {}
    if "column_mappings" not in st.session_state:
        st.session_state.column_mappings = {}
    if "merged_data" not in st.session_state:
        st.session_state.merged_data = None
    if "processing_log" not in st.session_state:
        st.session_state.processing_log = []

def log_message(message: str, level: str = "INFO"):
    """Add message to processing log"""
    st.session_state.processing_log.append(f"[{level}] {message}")

def detect_file_type(df: pd.DataFrame) -> str:
    """Detect if file is observation, measurement, or dictionary type"""
    columns = [col.lower() for col in df.columns]
    
    # Check for observation patterns
    if any(col in columns for col in ['observationid', 'observation_id']):
        if 'value' in columns and 'studyid' in columns:
            return "observation_sql"
        elif any(col in columns for col in ['name', 'displayname', 'valuecount']):
            return "observation_dict"
    
    # Check for measurement patterns  
    if any(col in columns for col in ['measurementtypeidx', 'measurement_type_idx', 'measurementid']):
        if any(col in columns for col in ['floatvalue', 'float_value', 'value']):
            return "measurement_sql"
        elif any(col in columns for col in ['name', 'displayname', 'description']):
            return "measurement_dict"
    
    # Generic dictionary detection
    if len(df.columns) >= 2 and any(col in columns for col in ['name', 'displayname', 'description']):
        return "generic_dict"
    
    # Generic SQL data detection
    if 'studyid' in columns or 'study_id' in columns:
        return "generic_sql"
    
    return "unknown"

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names based on file type"""
    df = df.copy()
    columns = {col: col.lower() for col in df.columns}
    df.columns = [columns[col] for col in df.columns]
    
    # Standardize common variations
    column_mappings = {
        'observationid': 'observation_id',
        'measurementtypeidx': 'measurement_id', 
        'measurement_type_idx': 'measurement_id',
        'measurementid': 'measurement_id',
        'studyid': 'study_id',
        'floatvalue': 'value',
        'float_value': 'value',
        'displayname': 'display_name',
        'valuecount': 'value_count',
        'nativeunits': 'units'
    }
    
    df.columns = [column_mappings.get(col, col) for col in df.columns]
    return df

def load_sql_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
    """Load and analyze SQL CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        file_type = detect_file_type(df)
        df = standardize_columns(df)
        
        log_message(f"Loaded {uploaded_file.name}: {len(df)} rows, type: {file_type}")
        return df, file_type
    except Exception as e:
        log_message(f"Error loading {uploaded_file.name}: {str(e)}", "ERROR")
        return None, "error"

def load_dictionary_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
    """Load and analyze dictionary CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        file_type = detect_file_type(df)
        df = standardize_columns(df)
        
        log_message(f"Loaded dictionary {uploaded_file.name}: {len(df)} rows, type: {file_type}")
        return df, file_type
    except Exception as e:
        log_message(f"Error loading dictionary {uploaded_file.name}: {str(e)}", "ERROR")
        return None, "error"

def create_lookup_dict(dict_df: pd.DataFrame, file_type: str) -> Dict:
    """Create lookup dictionary from dictionary dataframe"""
    lookup = {}
    
    if file_type == "observation_dict":
        # Group by observation_id and create lookup
        for _, row in dict_df.iterrows():
            obs_id = row.get('observation_id') or row.get('observationid')
            name = row.get('name', row.get('display_name', ''))
            display_name = row.get('display_name', row.get('displayname', ''))
            
            if obs_id not in lookup:
                lookup[obs_id] = {
                    'name': name,
                    'display_name': display_name,
                    'values': []
                }
            
            if 'value' in row and pd.notna(row['value']):
                lookup[obs_id]['values'].append(row['value'])
                
    elif file_type == "measurement_dict":
        # Create measurement lookup
        for _, row in dict_df.iterrows():
            meas_id = row.get('measurement_id') or row.get('measurementid')
            name = row.get('name', row.get('display_name', ''))
            display_name = row.get('display_name', row.get('displayname', ''))
            
            if meas_id:
                lookup[meas_id] = {
                    'name': name,
                    'display_name': display_name
                }
    
    return lookup

def decode_observations(df: pd.DataFrame, obs_lookup: Dict) -> pd.DataFrame:
    """Decode observation IDs using lookup dictionary"""
    decoded_df = df.copy()
    
    # Add decoded columns
    decoded_df['observation_name'] = df['observation_id'].map(
        lambda x: obs_lookup.get(x, {}).get('name', f'Unknown_{x}')
    )
    decoded_df['observation_display_name'] = df['observation_id'].map(
        lambda x: obs_lookup.get(x, {}).get('display_name', f'Unknown_{x}')
    )
    
    return decoded_df

def decode_measurements(df: pd.DataFrame, meas_lookup: Dict) -> pd.DataFrame:
    """Decode measurement IDs using lookup dictionary"""
    decoded_df = df.copy()
    
    # Add decoded columns
    decoded_df['measurement_name'] = df['measurement_id'].map(
        lambda x: meas_lookup.get(x, {}).get('name', f'Unknown_{x}')
    )
    decoded_df['measurement_display_name'] = df['measurement_id'].map(
        lambda x: meas_lookup.get(x, {}).get('display_name', f'Unknown_{x}')
    )
    
    return decoded_df

def create_lookup_from_columns(dict_df: pd.DataFrame, id_column: str, name_column: str) -> Dict:
    """Create lookup dictionary using specified columns"""
    lookup = {}
    
    for _, row in dict_df.iterrows():
        code_id = row.get(id_column)
        name = row.get(name_column, '')
        
        if pd.notna(code_id):
            lookup[code_id] = str(name)
    
    return lookup

def transform_to_exct_format(sql_files_data: Dict, dict_files_data: Dict, 
                            sql_dict_mappings: Dict, column_mappings: Dict) -> pd.DataFrame:
    """Transform data to standardized EXCTsourceData format"""
    all_records = []
    
    # Process each SQL file
    for sql_filename, (sql_df, sql_type) in sql_files_data.items():
        log_message(f"Processing {sql_filename}...")
        
        # Get column mappings for this file
        file_mappings = column_mappings.get(sql_filename, {})
        
        if not file_mappings.get('id_column') or not file_mappings.get('value_column'):
            log_message(f"Skipping {sql_filename}: Missing required column mappings", "WARNING")
            continue
        
        id_column = file_mappings['id_column']
        value_column = file_mappings['value_column']
        unit_column = file_mappings.get('unit_column')
        
        # Create lookup dictionary if available
        lookup = {}
        dict_filename = sql_dict_mappings.get(sql_filename)
        if dict_filename and dict_filename in dict_files_data:
            dict_df, dict_type = dict_files_data[dict_filename]
            dict_mappings = column_mappings.get(dict_filename, {})
            
            if dict_mappings.get('dict_id_column') and dict_mappings.get('dict_name_column'):
                lookup = create_lookup_from_columns(
                    dict_df, 
                    dict_mappings['dict_id_column'], 
                    dict_mappings['dict_name_column']
                )
                log_message(f"Created lookup from {dict_filename} with {len(lookup)} entries")
        
        # Process each row in SQL file
        for _, row in sql_df.iterrows():
            variable_id = row.get(id_column)
            variable_value = row.get(value_column)
            variable_unit = row.get(unit_column, '') if unit_column else ''
            
            # Get variable name from lookup or use ID as fallback
            if pd.notna(variable_id) and variable_id in lookup:
                variable_name = lookup[variable_id]
            else:
                variable_name = f"Unknown_{variable_id}" if pd.notna(variable_id) else "Unknown"
            
            # Create standardized record
            record = {
                'EXCTsourceData_variableID': variable_id,
                'EXCTsourceData_variableName': variable_name,
                'EXCTsourceData_variableValue': variable_value,
                'EXCTsourceData_variableUnit': variable_unit,
                'source_file': sql_filename,
                'source_type': sql_type
            }
            
            # Add other columns from original data (like study_id, etc.)
            for col in sql_df.columns:
                if col not in [id_column, value_column, unit_column]:
                    record[f'original_{col}'] = row.get(col)
            
            all_records.append(record)
    
    if not all_records:
        return pd.DataFrame()
    
    # Create final dataframe
    result_df = pd.DataFrame(all_records)
    
    # Sort by source_file and variable_id for consistency
    result_df = result_df.sort_values(['source_file', 'EXCTsourceData_variableID']).reset_index(drop=True)
    
    log_message(f"Created EXCTsourceData format: {len(result_df)} rows, {len(result_df.columns)} columns")
    return result_df

def main():
    """Main Streamlit application"""
    init_session_state()
    
    st.title("🔄 SQL Database Decoder & Merger")
    st.markdown("""
    Upload multiple SQL database files (CSV format) and map them to their corresponding dictionary files 
    to merge and decode observation/measurement codes into readable names.
    """)
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("📁 File Upload")
        
        st.subheader("SQL Database Files")
        sql_files = st.file_uploader(
            "Upload SQL CSV files",
            type=['csv'],
            accept_multiple_files=True,
            key="sql_files",
            help="Upload observation_sql.csv, measurement_sql.csv, or similar database exports"
        )
        
        st.subheader("Dictionary Files (Optional)")
        dict_files = st.file_uploader(
            "Upload Dictionary CSV files",
            type=['csv'],
            accept_multiple_files=True,
            key="dict_files",
            help="Upload observation_dictionary.csv, measurement_dictionary.csv for code decoding"
        )
    
    # Main content area
    if sql_files:
        # Load and display SQL files
        st.header("📊 SQL Database Files")
        sql_files_data = {}
        
        for file in sql_files:
            with st.expander(f"📄 {file.name}"):
                df, file_type = load_sql_file(file)
                if df is not None:
                    sql_files_data[file.name] = (df, file_type)
                    st.write(f"**Type:** {file_type}")
                    st.write(f"**Shape:** {df.shape}")
                    st.write("**Columns:**", list(df.columns))
                    st.dataframe(df.head(), use_container_width=True)
        
        # Load dictionary files if available
        dict_files_data = {}
        if dict_files:
            st.header("📚 Dictionary Files")
            for file in dict_files:
                with st.expander(f"📖 {file.name}"):
                    df, file_type = load_dictionary_file(file)
                    if df is not None:
                        dict_files_data[file.name] = (df, file_type)
                        st.write(f"**Type:** {file_type}")
                        st.write(f"**Shape:** {df.shape}")
                        st.write("**Columns:**", list(df.columns))
                        st.dataframe(df.head(), use_container_width=True)
        
        # Mapping interface
        if sql_files_data:
            st.header("🔗 File & Column Mappings")
            
            # Dictionary mapping section
            st.subheader("📚 SQL-to-Dictionary Mapping")
            st.markdown("**Map each SQL file to its corresponding dictionary file (optional):**")
            
            for sql_filename in sql_files_data.keys():
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write(f"**{sql_filename}**")
                    st.write(f"Type: {sql_files_data[sql_filename][1]}")
                
                with col2:
                    dict_options = ["None"] + list(dict_files_data.keys())
                    current_mapping = st.session_state.sql_dict_mappings.get(sql_filename, "None")
                    
                    selected_dict = st.selectbox(
                        f"Dictionary for {sql_filename}",
                        options=dict_options,
                        index=dict_options.index(current_mapping) if current_mapping in dict_options else 0,
                        key=f"mapping_{sql_filename}"
                    )
                    
                    # Update mapping
                    if selected_dict != "None":
                        st.session_state.sql_dict_mappings[sql_filename] = selected_dict
                    else:
                        if sql_filename in st.session_state.sql_dict_mappings:
                            del st.session_state.sql_dict_mappings[sql_filename]
                
                st.divider()
            
            # Column mapping section
            st.subheader("📊 Column Mappings")
            st.markdown("**Define column mappings for EXCTsourceData format:**")
            
            # SQL files column mapping
            for sql_filename, (sql_df, sql_type) in sql_files_data.items():
                with st.expander(f"🔧 Column Mapping for {sql_filename}"):
                    st.markdown("**Required mappings for SQL file:**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        id_column = st.selectbox(
                            "Variable ID Column",
                            options=list(sql_df.columns),
                            key=f"sql_id_{sql_filename}",
                            help="Column containing the coded IDs (e.g., ObservationID, MeasurementTypeIdx)"
                        )
                    
                    with col2:
                        value_column = st.selectbox(
                            "Variable Value Column", 
                            options=list(sql_df.columns),
                            key=f"sql_value_{sql_filename}",
                            help="Column containing the actual values"
                        )
                    
                    with col3:
                        unit_options = ["None"] + list(sql_df.columns)
                        unit_column = st.selectbox(
                            "Variable Unit Column (Optional)",
                            options=unit_options,
                            key=f"sql_unit_{sql_filename}",
                            help="Column containing units (for measurements)"
                        )
                    
                    # Store column mappings
                    if sql_filename not in st.session_state.column_mappings:
                        st.session_state.column_mappings[sql_filename] = {}
                    
                    st.session_state.column_mappings[sql_filename].update({
                        'id_column': id_column,
                        'value_column': value_column,
                        'unit_column': unit_column if unit_column != "None" else None
                    })
            
            # Dictionary files column mapping
            if dict_files_data:
                st.markdown("**Dictionary column mappings:**")
                for dict_filename, (dict_df, dict_type) in dict_files_data.items():
                    with st.expander(f"📖 Column Mapping for {dict_filename}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            dict_id_column = st.selectbox(
                                "Dictionary ID Column",
                                options=list(dict_df.columns),
                                key=f"dict_id_{dict_filename}",
                                help="Column containing the coded IDs to match with SQL"
                            )
                        
                        with col2:
                            dict_name_column = st.selectbox(
                                "Dictionary Name Column",
                                options=list(dict_df.columns),
                                key=f"dict_name_{dict_filename}",
                                help="Column containing the descriptions/names"
                            )
                        
                        # Store dictionary column mappings
                        if dict_filename not in st.session_state.column_mappings:
                            st.session_state.column_mappings[dict_filename] = {}
                        
                        st.session_state.column_mappings[dict_filename].update({
                            'dict_id_column': dict_id_column,
                            'dict_name_column': dict_name_column
                        })
            
            # Show current mappings summary
            if st.session_state.sql_dict_mappings or st.session_state.column_mappings:
                st.subheader("📋 Current Configuration")
                
                if st.session_state.sql_dict_mappings:
                    st.markdown("**File Mappings:**")
                    for sql_file, dict_file in st.session_state.sql_dict_mappings.items():
                        st.write(f"• **{sql_file}** → **{dict_file}**")
                
                if st.session_state.column_mappings:
                    st.markdown("**Column Mappings:**")
                    for filename, mappings in st.session_state.column_mappings.items():
                        if filename.endswith('.csv'):
                            if 'id_column' in mappings:  # SQL file
                                st.write(f"• **{filename}**: ID={mappings.get('id_column')}, Value={mappings.get('value_column')}, Unit={mappings.get('unit_column', 'None')}")
                            elif 'dict_id_column' in mappings:  # Dictionary file
                                st.write(f"• **{filename}**: ID={mappings.get('dict_id_column')}, Name={mappings.get('dict_name_column')}")
            
            # Process button
            process_button = st.button("🔄 Process Files", type="primary", use_container_width=True)
            
            if process_button:
                st.session_state.processing_log = []  # Clear log
                log_message("Starting processing...")
                
                # Validate required column mappings
                missing_mappings = []
                for sql_filename in sql_files_data.keys():
                    if sql_filename not in st.session_state.column_mappings:
                        missing_mappings.append(f"{sql_filename}: No column mappings")
                    else:
                        mappings = st.session_state.column_mappings[sql_filename]
                        if not mappings.get('id_column'):
                            missing_mappings.append(f"{sql_filename}: Missing ID column")
                        if not mappings.get('value_column'):
                            missing_mappings.append(f"{sql_filename}: Missing Value column")
                
                if missing_mappings:
                    st.error("❌ Missing required column mappings:")
                    for msg in missing_mappings:
                        st.write(f"• {msg}")
                    return
                
                with st.spinner("Processing files..."):
                    # Transform to EXCTsourceData format
                    merged_data = transform_to_exct_format(
                        sql_files_data, 
                        dict_files_data, 
                        st.session_state.sql_dict_mappings,
                        st.session_state.column_mappings
                    )
                    st.session_state.merged_data = merged_data
                    log_message("Processing completed successfully!")
                    st.success("✅ Files processed successfully!")
                    st.rerun()
    
    # Processing log
    if st.session_state.processing_log:
        with st.expander("📋 Processing Log"):
            log_text = "\n".join(st.session_state.processing_log)
            st.text_area("Messages", log_text, height=200, disabled=True)
    
    # Display results
    if st.session_state.merged_data is not None and not st.session_state.merged_data.empty:
        st.header("🎯 Merged & Decoded Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(st.session_state.merged_data))
        with col2:
            st.metric("Total Columns", len(st.session_state.merged_data.columns))
        with col3:
            st.metric("Memory Usage", f"{st.session_state.merged_data.memory_usage().sum() / 1024:.1f} KB")
        
        # Show data preview
        st.dataframe(st.session_state.merged_data.head(100), use_container_width=True)
        
        # Download section
        st.subheader("💾 Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_buffer = io.StringIO()
            st.session_state.merged_data.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_buffer.getvalue(),
                file_name="EXCTsourceData_clean.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                st.session_state.merged_data.to_excel(writer, sheet_name='merged_data', index=False)
            
            st.download_button(
                label="📊 Download Excel",
                data=excel_buffer.getvalue(),
                file_name="EXCTsourceData_clean.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

if __name__ == "__main__":
    main()