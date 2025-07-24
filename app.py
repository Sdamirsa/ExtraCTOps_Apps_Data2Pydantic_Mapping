"""
Data2Pydantic_Map App

# Description
    [A Streamlit application for mapping and transforming EHR/database data to user-defined Pydantic models. Supports mapping template generation, data transformation, and LLM-assisted auto-mapping.]

    - Arguments:
        - Data (Excel, CSV): The file containing the source data to be mapped (supports both wide and long format).
        - Pydantic Model (.py) Or Code (string): The file or code string containing the Pydantic model definition.
        - Mapping File (Excel, CSV): The file defining the mapping between source data and Pydantic fields.

    - Environment Arguments:
        - None required for basic use. (LLM auto-mapping requires OpenAI/Azure API key.)

    - Returns
        - Mapping Template (Excel): A template for mapping source data to Pydantic fields.
        - Transformed Data (JSON): Data transformed to match the Pydantic model structure.
        - Transformed Data (Excel): Flattened data with nested fields separated by '::'.
        - Mapping Monitor (Excel): Mapping file with validation feedback.

# Engine:
    - Serve (utils/data/main-function/sub-function): main-function
    - Served by (API/Direct/Subprocess): Direct
    - Path to venv, if require separate venv: the_venvs/venv_streamlit
    - libraries to import: [pydantic,streamlit,pandas,openpyxl,xlsxwriter,openai]

# Identity
    - Last Status (future/in-progress/complete/published): published
    - Publish Date: 2025-05-21
    - Version: 0.1
    - License: MIT
    - Author: Seyed Amir Ahmad Safavi-Naini Safavi-Naini, sdamirsa@gmail.com (the nominee for the longest name ever)
    - Source: https://github.com/Sdamirsa/awesome_extraction_with_LLM_ExtraCTOps

# Changelog
    - 2025-05-21: version 0.1 (initial public release)
    - 2025-07-24: version 0.2
        - Better code organization with reusable functions
        - Enhanced error handling and validation
        - Improved UI/UX with better guidance
        - Data preview capabilities
        - Progress indicators
        - Better session state management
        - More comprehensive mapping validation
        - Performance optimizations
    

# To-do:
    - [] Improve error handling and user feedback for all sections
    - [] Add support for additional data formats (e.g., JSON, TXT)
    - [] Enhance LLM prompt engineering for more robust auto-mapping
    - [] Add more sample models and mapping templates
    - [] Add advanced mapping logic (e.g., custom evaluation methods)

"""


import streamlit as st
import pandas as pd
import json
import importlib.util
import sys
import tempfile
from types import ModuleType
from typing import Any, Dict, Tuple, List, Optional, get_origin, get_args, Union
from pydantic import BaseModel
import io
from enum import Enum, EnumMeta
import traceback
from dataclasses import dataclass
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Data2Pydantic Mapping Tool - Enhanced", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# Configuration & Constants
# ===========================

@dataclass
class AppConfig:
    """Application configuration"""
    MAX_PREVIEW_ROWS = 100
    MAX_UNIQUE_VALUES_DISPLAY = 20
    SUPPORTED_DATA_FORMATS = ["csv", "xlsx"]
    SUPPORTED_MODEL_FORMATS = ["py"]
    DEFAULT_EVALUATION_METHOD = "smart_exact_match"
    DEFAULT_MULTIVALUE_METHOD = "haveBoth"

# ===========================
# Session State Management
# ===========================

def init_session_state():
    """Initialize session state variables"""
    if "active_section" not in st.session_state:
        st.session_state["active_section"] = None
    if "loaded_models" not in st.session_state:
        st.session_state["loaded_models"] = {}
    if "current_mapping" not in st.session_state:
        st.session_state["current_mapping"] = None
    if "transformation_results" not in st.session_state:
        st.session_state["transformation_results"] = None

# ===========================
# Utility Functions
# ===========================

def load_pydantic_model_from_code(code_str: str) -> Tuple[Dict[str, Any], ModuleType]:
    """Load Pydantic models from code string (no caching due to module serialization)"""
    try:
        import types
        import typing
        import enum
        from pydantic import Field as PydanticField
        module = types.ModuleType('user_model')
        # Add common imports to module namespace
        module.__dict__.update({
            'BaseModel': BaseModel,
            'Field': PydanticField,  # Use actual Field instead of lambda
            'Optional': Optional,
            'List': List,
            'Dict': Dict,
            'Union': Union,
            'Enum': Enum,
            'Any': Any,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'typing': typing,  # Add typing module for complex models
            'Tuple': Tuple,
            'enum': enum  # Add enum module
        })
        exec(code_str, module.__dict__)
        
        # Find all BaseModel subclasses
        models = {
            k: v for k, v in module.__dict__.items() 
            if isinstance(v, type) and issubclass(v, BaseModel) and v is not BaseModel
        }
        return models, module
    except Exception as e:
        st.error(f"Error loading model code: {str(e)}")
        st.code(traceback.format_exc())
        return {}, None

def load_pydantic_model_from_file(py_file) -> Tuple[Dict[str, Any], ModuleType]:
    """Load Pydantic models from uploaded file"""
    try:
        code_str = py_file.read().decode('utf-8')
        return load_pydantic_model_from_code(code_str)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return {}, None

@st.cache_data
def read_data_file(file_content: bytes, file_name: str) -> pd.DataFrame:
    """Read data file with caching and error handling"""
    try:
        if file_name.endswith(".csv"):
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    return pd.read_csv(io.BytesIO(file_content), encoding=encoding, on_bad_lines='skip')
                except:
                    continue
            raise ValueError("Could not read CSV with any common encoding")
        else:  # Excel
            return pd.read_excel(io.BytesIO(file_content))
    except Exception as e:
        st.error(f"Error reading data file: {str(e)}")
        return pd.DataFrame()

# ===========================
# Model Analysis Functions
# ===========================

def flatten_model(model: type, prefix: str = "") -> List[Tuple[str, str]]:
    """Flatten a Pydantic model into field paths"""
    results = []
    for name, field in model.model_fields.items():
        typ = field.annotation
        path = f"{prefix}::{name}" if prefix else name
        origin = get_origin(typ)
        
        if origin in (list, List):
            inner = get_args(typ)[0]
            results.extend(_flatten_type(inner, path))
        else:
            results.extend(_flatten_type(typ, path))
    return results

def _flatten_type(typ, path) -> List[Tuple[str, str]]:
    """Helper to flatten a type"""
    origin = get_origin(typ)
    
    if origin is Union:
        # Handle Optional[...] or Union[...]
        args = [a for a in get_args(typ) if a is not type(None)]
        if args:
            return _flatten_type(args[0], path)
        else:
            return [(path, "Any")]
    elif isinstance(typ, type) and issubclass(typ, BaseModel):
        return flatten_model(typ, path)
    elif isinstance(typ, type) and issubclass(typ, Enum):
        return [(path, typ.__name__)]
    else:
        type_name = getattr(typ, "__name__", str(typ))
        return [(path, type_name)]

def get_enum_options_list(enum_cls) -> List[str]:
    """Get list of enum values"""
    if enum_cls is None:
        return []
    try:
        return [str(e.value) for e in enum_cls]
    except Exception:
        return []

def analyze_model_structure(model: type) -> Dict[str, Any]:
    """Analyze Pydantic model structure for UI display"""
    structure = {
        "fields": {},
        "nested_models": [],
        "enums": [],
        "total_fields": 0
    }
    
    flattened = flatten_model(model)
    structure["total_fields"] = len(flattened)
    
    for field_path, field_type in flattened:
        structure["fields"][field_path] = field_type
        if "::" in field_path:
            structure["nested_models"].append(field_path.split("::")[0])
        if field_type not in ["str", "int", "float", "bool", "Any"]:
            structure["enums"].append(field_type)
    
    structure["nested_models"] = list(set(structure["nested_models"]))
    structure["enums"] = list(set(structure["enums"]))
    
    return structure

# ===========================
# Mapping Generation
# ===========================

def generate_mapping_template(model: BaseModel.__class__, schema_module: ModuleType = None) -> pd.DataFrame:
    """Generate mapping template with improved structure"""
    rows = []
    
    for fld, typ_name in flatten_model(model):
        # Get enum options if applicable
        enum_cls = getattr(schema_module, typ_name, None) if schema_module else None
        options = get_enum_options_list(enum_cls)
        
        # Special handling for bool
        if typ_name.lower() == "bool" or (enum_cls and typ_name.lower().startswith("bool")):
            options = ["True", "False"]
        
        # Determine if pydantic_value should be blank
        is_numeric = typ_name.lower() in ("float", "int")
        
        if options:
            # Create row for each option (enum/bool values)
            for opt in options:
                rows.append({
                    "pydantic_field": fld,
                    "pydantic_type": typ_name,
                    "evaluation_method": AppConfig.DEFAULT_EVALUATION_METHOD,
                    "multiValue_handling_method": AppConfig.DEFAULT_MULTIVALUE_METHOD,
                    "pydantic_value": opt if not is_numeric else "",
                    "Observation_ColName": "",
                    "Observation_Value": "",
                    "Observation_Value2": "",
                    "Observation_Value3": "",
                    "Observation_Value4": "",
                    "notes": ""  # Added for user notes
                })
        else:
            # Single row for other types
            rows.append({
                "pydantic_field": fld,
                "pydantic_type": typ_name,
                "evaluation_method": AppConfig.DEFAULT_EVALUATION_METHOD,
                "multiValue_handling_method": AppConfig.DEFAULT_MULTIVALUE_METHOD,
                "pydantic_value": "",
                "Observation_ColName": "",
                "Observation_Value": "",
                "Observation_Value2": "",
                "Observation_Value3": "",
                "Observation_Value4": "",
                "notes": ""
            })
    
    # Ensure column order
    col_order = [
        "pydantic_field", "pydantic_type", "evaluation_method", "multiValue_handling_method",
        "pydantic_value", "Observation_ColName", "Observation_Value", "Observation_Value2", 
        "Observation_Value3", "Observation_Value4", "notes"
    ]
    
    return pd.DataFrame(rows)[col_order]

def export_mapping_with_guide(mapping_df: pd.DataFrame, out_path: Union[str, io.BytesIO]):
    """Export mapping with enhanced guide and formatting"""
    import xlsxwriter
    
    guide_text = """Mapping Template Instructions:

1. BASIC MAPPING:
   - pydantic_field: The target field in your Pydantic model (do not modify)
   - pydantic_type: The data type of the field (do not modify)
   - pydantic_value: For enums/bools, this is pre-filled. For other types, leave blank.
   - Observation_ColName: Enter the column name from your source data

2. VALUE MAPPING:
   - For direct value mapping: Use Observation_ColName for the source column
   - For conditional mapping: Use Observation_Value fields to specify matching values
   - You can specify up to 4 different values that map to the same pydantic_value

3. EVALUATION METHODS:
   - smart_exact_match: Case-insensitive exact matching (default)
   - exact_match: Case-sensitive exact matching
   - contains: Source value contains the specified value
   - regex: Use regular expressions for matching

4. MULTIVALUE HANDLING:
   - haveBoth: Keep all values (default for lists)
   - first: Take only the first value
   - join_semicolon: Join multiple values with semicolon

5. EXAMPLES:
   - Boolean field: Map "Yes"/"Y"/"1" to True, "No"/"N"/"0" to False
   - Enum field: Map various spellings to standardized enum values
   - Numeric field: Direct mapping from source column
   - List field: Collect multiple values into a list

6. NOTES:
   - Use the notes column to document your mapping decisions
   - Rows with errors will be highlighted in the mapping monitor
"""
    
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        # Write mapping sheet
        mapping_df.to_excel(writer, index=False, sheet_name="mapping")
        
        # Write guide sheet
        guide_df = pd.DataFrame({"Instructions": [guide_text]})
        guide_df.to_excel(writer, index=False, sheet_name="guide")
        
        # Write examples sheet
        examples = pd.DataFrame([
            {
                "pydantic_field": "patient_gender",
                "pydantic_type": "GenderEnum",
                "evaluation_method": "smart_exact_match",
                "multiValue_handling_method": "first",
                "pydantic_value": "male",
                "Observation_ColName": "gender",
                "Observation_Value": "M",
                "Observation_Value2": "Male",
                "Observation_Value3": "MALE",
                "Observation_Value4": "",
                "notes": "Maps various gender representations to standardized enum"
            },
            {
                "pydantic_field": "is_active",
                "pydantic_type": "bool",
                "evaluation_method": "smart_exact_match",
                "multiValue_handling_method": "first",
                "pydantic_value": "True",
                "Observation_ColName": "status",
                "Observation_Value": "Active",
                "Observation_Value2": "ACTIVE",
                "Observation_Value3": "1",
                "Observation_Value4": "",
                "notes": "Active status maps to True"
            }
        ])
        examples.to_excel(writer, index=False, sheet_name="examples")
        
        # Format the workbook
        workbook = writer.book
        worksheet = writer.sheets["mapping"]
        
        # Add formatting
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D3D3D3',
            'border': 1
        })
        
        # Apply header formatting
        for col_num, value in enumerate(mapping_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Adjust column widths
        worksheet.set_column('A:A', 30)  # pydantic_field
        worksheet.set_column('B:B', 15)  # pydantic_type
        worksheet.set_column('C:D', 20)  # methods
        worksheet.set_column('E:E', 15)  # pydantic_value
        worksheet.set_column('F:F', 25)  # Observation_ColName
        worksheet.set_column('G:J', 20)  # Observation_Values
        worksheet.set_column('K:K', 40)  # notes
        
        # Highlight rows with errors if present
        if "mapping_error" in mapping_df.columns:
            error_format = workbook.add_format({'bg_color': '#FFC7CE'})
            for row_num, error in enumerate(mapping_df["mapping_error"], start=1):
                if error:
                    worksheet.set_row(row_num, None, error_format)

# ===========================
# Mapping Validation
# ===========================

def validate_mapping(mapping_df: pd.DataFrame, schema_module: ModuleType) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Enhanced mapping validation with detailed error reporting"""
    from enum import EnumMeta
    
    mapping_df = mapping_df.copy()
    
    # Build enum registry
    enum_registry = {
        cls.__name__: cls
        for cls in vars(schema_module).values()
        if isinstance(cls, EnumMeta)
    }
    
    # Initialize error and warning columns
    mapping_df["mapping_error"] = ""
    mapping_df["mapping_warning"] = ""
    
    validation_summary = {
        "total_rows": len(mapping_df),
        "error_count": 0,
        "warning_count": 0,
        "unmapped_fields": [],
        "duplicate_mappings": []
    }
    
    # Check each row
    for i, row in mapping_df.iterrows():
        errors = []
        warnings = []
        
        ptype = row["pydantic_type"]
        pvalue = row.get("pydantic_value", "")
        obs_col = row.get("Observation_ColName", "")
        
        # Check enum values
        if ptype in enum_registry and pvalue:
            allowed = {str(e.value) for e in enum_registry[ptype]}
            if str(pvalue) not in allowed:
                errors.append(f"Invalid enum value '{pvalue}'. Allowed: {', '.join(allowed)}")
        
        # Check observation column
        if not str(obs_col).strip():
            errors.append("No Observation_ColName specified")
        
        # Check boolean values
        if ptype.lower() == "bool" and pvalue:
            if str(pvalue) not in ["True", "False"]:
                errors.append(f"Invalid boolean value '{pvalue}'. Use 'True' or 'False'")
        
        # Check evaluation method
        valid_methods = ["smart_exact_match", "exact_match", "contains", "regex"]
        if row.get("evaluation_method") not in valid_methods:
            warnings.append(f"Unknown evaluation method. Using default.")
        
        # Store errors and warnings
        mapping_df.at[i, "mapping_error"] = "; ".join(errors)
        mapping_df.at[i, "mapping_warning"] = "; ".join(warnings)
        
        if errors:
            validation_summary["error_count"] += 1
        if warnings:
            validation_summary["warning_count"] += 1
    
    # Check for unmapped fields
    all_fields = set(mapping_df["pydantic_field"].unique())
    mapped_fields = set(mapping_df[mapping_df["Observation_ColName"].notna()]["pydantic_field"].unique())
    validation_summary["unmapped_fields"] = list(all_fields - mapped_fields)
    
    # Check for duplicate mappings
    duplicates = mapping_df[mapping_df.duplicated(subset=["pydantic_field", "pydantic_value", "Observation_ColName"], keep=False)]
    if not duplicates.empty:
        validation_summary["duplicate_mappings"] = duplicates[["pydantic_field", "pydantic_value", "Observation_ColName"]].values.tolist()
    
    return mapping_df, validation_summary

# ===========================
# Data Transformation
# ===========================

def transform_data(
    mapping_df: pd.DataFrame,
    data_df: pd.DataFrame,
    model: type,
    data_format: str,
    patient_id_col: str = "",
    obs_name_col: str = "",
    obs_value_col: str = ""
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Enhanced data transformation with better error handling"""
    
    transformed_data_for_pydantic = []
    transformed_data_for_excel = []
    errors = []
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        if data_format == "Wide":
            total_rows = len(data_df)
            
            for idx, source_row in data_df.iterrows():
                # Update progress
                progress = (idx + 1) / total_rows
                progress_bar.progress(progress)
                status_text.text(f"Processing row {idx + 1} of {total_rows}")
                
                item_pydantic = {}
                item_excel = {}
                
                # Process each pydantic field
                for pydantic_field_name, mapping_group in mapping_df.groupby("pydantic_field"):
                    values_for_field = []
                    
                    for _, map_rule in mapping_group.iterrows():
                        obs_col_name = map_rule.get("Observation_ColName")
                        pydantic_target_value = map_rule.get("pydantic_value")
                        evaluation_method = map_rule.get("evaluation_method", "smart_exact_match")
                        
                        if obs_col_name and obs_col_name in source_row and pd.notna(source_row[obs_col_name]):
                            source_val = source_row[obs_col_name]
                            
                            # Apply evaluation method
                            if pydantic_target_value:
                                if evaluate_value_match(source_val, pydantic_target_value, evaluation_method):
                                    values_for_field.append(pydantic_target_value)
                                
                                # Check observation values
                                for i in range(1, 5):
                                    obs_val = map_rule.get(f"Observation_Value{i if i > 1 else ''}")
                                    if obs_val and evaluate_value_match(source_val, obs_val, evaluation_method):
                                        values_for_field.append(pydantic_target_value)
                                        break
                            else:
                                # Direct value mapping
                                values_for_field.append(source_val)
                    
                    # Handle multiple values
                    final_value = handle_multiple_values(
                        values_for_field,
                        mapping_group.iloc[0].get("multiValue_handling_method", "haveBoth")
                    )
                    
                    # Assign values
                    item_excel[pydantic_field_name] = final_value
                    assign_nested_value(item_pydantic, pydantic_field_name, final_value)
                
                transformed_data_for_excel.append(item_excel)
                
                # Try to create Pydantic instance
                try:
                    model_instance = model(**item_pydantic)
                    transformed_data_for_pydantic.append(model_instance.model_dump(exclude_none=True))
                except Exception as e:
                    errors.append({
                        "row_identifier": f"Row {idx}",
                        "error": str(e),
                        "data": item_pydantic
                    })
        
        elif data_format == "Long":
            if not all([patient_id_col, obs_name_col, obs_value_col]):
                raise ValueError("Long format requires all three column specifications")
            
            patient_groups = list(data_df.groupby(patient_id_col))
            total_patients = len(patient_groups)
            
            for patient_idx, (patient_id, group_df) in enumerate(patient_groups):
                # Update progress
                progress = (patient_idx + 1) / total_patients
                progress_bar.progress(progress)
                status_text.text(f"Processing patient {patient_idx + 1} of {total_patients}")
                
                item_pydantic = {}
                item_excel = {patient_id_col: patient_id}
                
                # Process each pydantic field
                for pydantic_field_name, mapping_rules_for_field in mapping_df.groupby("pydantic_field"):
                    values_for_field = []
                    
                    for _, map_rule in mapping_rules_for_field.iterrows():
                        target_obs_name = map_rule.get("Observation_ColName")
                        pydantic_target_value = map_rule.get("pydantic_value")
                        evaluation_method = map_rule.get("evaluation_method", "smart_exact_match")
                        
                        # Find matching observations
                        relevant_rows = group_df[group_df[obs_name_col] == target_obs_name]
                        
                        for _, source_row in relevant_rows.iterrows():
                            source_obs_value = source_row[obs_value_col]
                            
                            if pydantic_target_value:
                                # Check if any observation values match
                                matched = False
                                for i in range(1, 5):
                                    obs_val = map_rule.get(f"Observation_Value{i if i > 1 else ''}")
                                    if obs_val and evaluate_value_match(source_obs_value, obs_val, evaluation_method):
                                        values_for_field.append(pydantic_target_value)
                                        matched = True
                                        break
                                
                                # If no specific observation values, any non-null value triggers
                                if not matched and pd.notna(source_obs_value) and not any(
                                    map_rule.get(f"Observation_Value{i if i > 1 else ''}") 
                                    for i in range(1, 5)
                                ):
                                    values_for_field.append(pydantic_target_value)
                            else:
                                # Direct value mapping
                                if pd.notna(source_obs_value):
                                    values_for_field.append(source_obs_value)
                    
                    # Handle multiple values
                    final_value = handle_multiple_values(
                        values_for_field,
                        mapping_rules_for_field.iloc[0].get("multiValue_handling_method", "haveBoth")
                    )
                    
                    # Assign values
                    item_excel[pydantic_field_name] = final_value
                    assign_nested_value(item_pydantic, pydantic_field_name, final_value)
                
                # Add patient ID if it's a field in the model
                if hasattr(model, 'model_fields'):
                    if patient_id_col in model.model_fields and patient_id_col not in item_pydantic:
                        item_pydantic[patient_id_col] = patient_id
                    elif "patient_id" in model.model_fields and "patient_id" not in item_pydantic:
                        item_pydantic["patient_id"] = patient_id
                
                transformed_data_for_excel.append(item_excel)
                
                # Try to create Pydantic instance
                try:
                    model_instance = model(**item_pydantic)
                    transformed_data_for_pydantic.append(model_instance.model_dump(exclude_none=True))
                except Exception as e:
                    errors.append({
                        "row_identifier": f"Patient {patient_id}",
                        "error": str(e),
                        "data": item_pydantic
                    })
    
    finally:
        progress_bar.empty()
        status_text.empty()
    
    return transformed_data_for_pydantic, transformed_data_for_excel, errors

def evaluate_value_match(source_value: Any, target_value: Any, method: str) -> bool:
    """Evaluate if source value matches target value based on method"""
    source_str = str(source_value).strip()
    target_str = str(target_value).strip()
    
    if method == "exact_match":
        return source_str == target_str
    elif method == "smart_exact_match":
        return source_str.lower() == target_str.lower()
    elif method == "contains":
        return target_str.lower() in source_str.lower()
    elif method == "regex":
        import re
        try:
            return bool(re.match(target_str, source_str))
        except:
            return False
    else:
        # Default to smart exact match
        return source_str.lower() == target_str.lower()

def handle_multiple_values(values: List[Any], method: str) -> Any:
    """Handle multiple values based on specified method"""
    if not values:
        return None
    
    # Remove duplicates while preserving order
    unique_values = []
    seen = set()
    for v in values:
        if pd.notna(v) and v not in seen:
            unique_values.append(v)
            seen.add(v)
    
    if not unique_values:
        return None
    
    if len(unique_values) == 1:
        return unique_values[0]
    
    if method == "first":
        return unique_values[0]
    elif method == "join_semicolon":
        return ";".join(str(v) for v in unique_values)
    else:  # haveBoth or collect_all
        return unique_values

def assign_nested_value(data: Dict, field_path: str, value: Any):
    """Assign value to nested dictionary based on field path"""
    keys = field_path.split("::")
    d = data
    
    for i, key in enumerate(keys):
        if i == len(keys) - 1:
            d[key] = value
        else:
            d = d.setdefault(key, {})

# ===========================
# UI Components
# ===========================

def show_data_preview(df: pd.DataFrame, title: str = "Data Preview"):
    """Show data preview with statistics"""
    with st.expander(f"{title} (First {AppConfig.MAX_PREVIEW_ROWS} rows)", expanded=False):
        st.dataframe(df.head(AppConfig.MAX_PREVIEW_ROWS))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024:.1f} KB")

def show_model_info(model: type, structure: Dict[str, Any]):
    """Display model information"""
    with st.expander("Model Structure", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Fields", structure["total_fields"])
            st.metric("Nested Models", len(structure["nested_models"]))
        with col2:
            st.metric("Enum Types", len(structure["enums"]))
            st.metric("Field Types", len(set(structure["fields"].values())))
        
        if structure["nested_models"]:
            st.write("**Nested Models:**", ", ".join(structure["nested_models"]))
        if structure["enums"]:
            st.write("**Enum Types:**", ", ".join(structure["enums"]))

def show_validation_summary(validation_summary: Dict[str, Any]):
    """Display validation summary"""
    if validation_summary["error_count"] > 0:
        st.error(f"❌ Found {validation_summary['error_count']} errors in mapping")
    else:
        st.success("✅ No errors found in mapping")
    
    if validation_summary["warning_count"] > 0:
        st.warning(f"⚠️ Found {validation_summary['warning_count']} warnings in mapping")
    
    if validation_summary["unmapped_fields"]:
        st.info(f"ℹ️ Unmapped fields: {', '.join(validation_summary['unmapped_fields'][:5])}" + 
                ("..." if len(validation_summary["unmapped_fields"]) > 5 else ""))
    
    if validation_summary["duplicate_mappings"]:
        st.warning(f"⚠️ Found {len(validation_summary['duplicate_mappings'])} duplicate mappings")

# ===========================
# Main Sections
# ===========================

def section_generate_template():
    """Section 1: Generate Mapping Template"""
    st.header("📝 Step 1: Generate Mapping Template")
    
    # Model input
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Pydantic Model Input")
        py_file = st.file_uploader(
            "Upload .py file with Pydantic model", 
            type=["py"], 
            key="template_py_file",
            help="Upload a Python file containing your Pydantic model definitions"
        )
        
    with col2:
        st.subheader("Or Paste Code")
        code_text = st.text_area(
            "Paste your Pydantic model code here", 
            height=300, 
            key="template_code_text",
            placeholder="from pydantic import BaseModel\n\nclass MyModel(BaseModel):\n    field1: str\n    field2: int"
        )
    
    # Load models
    models = None
    schema_module = None
    
    if py_file is not None:
        models, schema_module = load_pydantic_model_from_file(py_file)
    elif code_text.strip():
        models, schema_module = load_pydantic_model_from_code(code_text)
    
    if models:
        st.success(f"✅ Found {len(models)} model(s)")
        
        # Model selection
        model_names = list(models.keys())
        if len(model_names) > 1:
            model_names = model_names[-1:] + model_names[:-1]  # Move last to first
        
        model_name = st.selectbox(
            "Select model to use", 
            model_names, 
            index=0,
            help="Choose which Pydantic model to generate a template for"
        )
        
        # Show model info
        if model_name:
            structure = analyze_model_structure(models[model_name])
            show_model_info(models[model_name], structure)
        
        # Generate template button
        if st.button("🚀 Generate Template", key="generate_template_btn", type="primary"):
            with st.spinner("Generating template..."):
                model = models[model_name]
                df_template = generate_mapping_template(model, schema_module)
                
                # Show preview
                show_data_preview(df_template, "Template Preview")
                
                # Export button
                towrite = io.BytesIO()
                export_mapping_with_guide(df_template, towrite)
                towrite.seek(0)
                
                st.download_button(
                    label="📥 Download Mapping Template (.xlsx)",
                    data=towrite,
                    file_name=f"mapping_template_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.info("💡 The template includes instructions and examples in separate sheets")

def section_transform_data():
    """Section 2: Transform Data Using Mapping"""
    st.header("🔄 Step 2: Transform Data Using Mapping")
    
    # File uploads
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Mapping File")
        map_file = st.file_uploader(
            "Upload mapping file", 
            type=["xlsx", "csv"], 
            key="transform_map_file",
            help="Upload the completed mapping template"
        )
    
    with col2:
        st.subheader("2. Pydantic Model")
        py_file = st.file_uploader(
            "Upload .py file", 
            type=["py"], 
            key="transform_py_file",
            help="Upload the same Pydantic model used for template generation"
        )
    
    with col3:
        st.subheader("3. Data File")
        data_file = st.file_uploader(
            "Upload data file", 
            type=["csv", "xlsx"], 
            key="transform_data_file",
            help="Upload the data file to transform"
        )
    
    # Model code alternative
    code_text = st.text_area(
        "Or paste your Pydantic model code here", 
        height=150, 
        key="transform_code_text",
        placeholder="Paste model code if not uploading .py file"
    )
    
    # Data format selection
    st.subheader("Data Format Configuration")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        data_format = st.radio(
            "Select data format:", 
            ("Wide", "Long"), 
            key="data_format_select",
            help="Wide: One row per entity\nLong: Multiple rows per entity"
        )
    
    with col2:
        patient_id_col_long = ""
        obs_name_col_long = ""
        obs_value_col_long = ""
        
        if data_format == "Long":
            st.info("📋 Long format requires specifying key columns")
            
            # Try to get column names from uploaded data
            column_names = []
            if data_file:
                try:
                    df_temp = read_data_file(data_file.read(), data_file.name)
                    data_file.seek(0)  # Reset file pointer
                    column_names = list(df_temp.columns)
                except:
                    pass
            
            if column_names:
                patient_id_col_long = st.selectbox(
                    "Patient/Entity ID Column", 
                    column_names, 
                    key="patient_id_col_long"
                )
                obs_name_col_long = st.selectbox(
                    "Observation Name Column", 
                    column_names, 
                    key="obs_name_col_long"
                )
                obs_value_col_long = st.selectbox(
                    "Observation Value Column", 
                    column_names, 
                    key="obs_value_col_long"
                )
            else:
                patient_id_col_long = st.text_input("Patient/Entity ID Column", key="patient_id_col_long")
                obs_name_col_long = st.text_input("Observation Name Column", key="obs_name_col_long")
                obs_value_col_long = st.text_input("Observation Value Column", key="obs_value_col_long")
    
    # Load models
    models = None
    schema_module = None
    
    if py_file is not None:
        models, schema_module = load_pydantic_model_from_file(py_file)
    elif code_text.strip():
        models, schema_module = load_pydantic_model_from_code(code_text)
    
    # Transform button
    if models and map_file is not None and data_file is not None:
        model_names = list(models.keys())
        if len(model_names) > 1:
            model_names = model_names[-1:] + model_names[:-1]
        
        model_name = st.selectbox(
            "Select model to use", 
            model_names, 
            index=0, 
            key="transform_model_select"
        )
        
        if st.button("🔄 Transform Data", key="transform_data_btn", type="primary"):
            with st.spinner("Transforming data..."):
                # Read files
                if map_file.name.endswith(".csv"):
                    mapping_df = pd.read_csv(map_file, dtype=str).fillna("")
                else:
                    mapping_df = pd.read_excel(map_file, sheet_name="mapping", dtype=str).fillna("")
                
                data_df = read_data_file(data_file.read(), data_file.name)
                
                # Validate mapping
                mapping_df_validated, validation_summary = validate_mapping(mapping_df, schema_module)
                
                # Show validation summary
                show_validation_summary(validation_summary)
                
                # Export mapping monitor
                col1, col2 = st.columns(2)
                
                with col1:
                    towrite2 = io.BytesIO()
                    export_mapping_with_guide(mapping_df_validated, towrite2)
                    towrite2.seek(0)
                    st.download_button(
                        label="📊 Download Mapping Monitor (Excel)",
                        data=towrite2,
                        file_name=f"mapping_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                # Transform data
                if validation_summary["error_count"] == 0:
                    transformed_pydantic, transformed_excel, errors = transform_data(
                        mapping_df_validated,
                        data_df,
                        models[model_name],
                        data_format,
                        patient_id_col_long,
                        obs_name_col_long,
                        obs_value_col_long
                    )
                    
                    # Store results in session state
                    st.session_state["transformation_results"] = {
                        "pydantic": transformed_pydantic,
                        "excel": transformed_excel,
                        "errors": errors
                    }
                    
                    # Show results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Successfully Transformed", len(transformed_pydantic))
                    with col2:
                        st.metric("Errors", len(errors))
                    with col3:
                        st.metric("Success Rate", f"{len(transformed_pydantic)/(len(transformed_pydantic)+len(errors))*100:.1f}%")
                    
                    # Show errors if any
                    if errors:
                        with st.expander(f"⚠️ Transformation Errors ({len(errors)})", expanded=False):
                            for error in errors[:10]:  # Show first 10
                                st.error(f"**{error['row_identifier']}**: {error['error']}")
                            if len(errors) > 10:
                                st.info(f"... and {len(errors) - 10} more errors")
                    
                    # Download buttons
                    st.subheader("📥 Download Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if transformed_pydantic:
                            json_str = json.dumps(transformed_pydantic, indent=2, default=str)
                            st.download_button(
                                label="Download Pydantic Data (JSON)",
                                data=json_str,
                                file_name=f"transformed_pydantic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                    
                    with col2:
                        if transformed_excel:
                            df_excel = pd.DataFrame(transformed_excel)
                            excel_io = io.BytesIO()
                            with pd.ExcelWriter(excel_io, engine='xlsxwriter') as writer:
                                df_excel.to_excel(writer, index=False, sheet_name='Transformed_Data')
                            excel_io.seek(0)
                            st.download_button(
                                label="Download Flattened Data (Excel)",
                                data=excel_io,
                                file_name=f"transformed_flat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    
                    # Show data preview
                    if transformed_excel:
                        df_preview = pd.DataFrame(transformed_excel)
                        show_data_preview(df_preview, "Transformed Data Preview")
                else:
                    st.error("❌ Please fix mapping errors before transforming data")

def section_smart_mapping():
    """Section 3: Smart (LLM) Mapping"""
    st.header("🤖 Step 3: Smart (LLM) Mapping")
    
    # Privacy warning
    with st.warning(""):
        st.markdown("""
        ⚠️ **Privacy Warning:** This feature will:
        1. Summarize your data structure (column names, types, sample values)
        2. Send this summary to an LLM to generate mapping suggestions
        3. **May expose data patterns to the LLM provider**
        
        ✅ **Safe Options:**
        - Local LLMs (Ollama)
        - HIPAA-compliant services (e.g., Azure OpenAI with BAA)
        - De-identified data only
        
        ❌ **Avoid if:**
        - Working with PHI/PII without proper agreements
        - Using public LLM services without data agreements
        """)
    
    # Data upload and format configuration
    st.subheader("1️⃣ Data Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_file = st.file_uploader(
            "Upload data file", 
            type=["csv", "xlsx"], 
            key="auto_data_file"
        )
        
        df = pd.DataFrame()
        if data_file:
            df = read_data_file(data_file.read(), data_file.name)
            data_file.seek(0)
            if not df.empty:
                show_data_preview(df, "Data Preview")
    
    with col2:
        data_format = st.radio(
            "Select data format:", 
            ("Wide", "Long"), 
            key="auto_data_format_select",
            help="Wide: One row per entity, Long: Multiple rows per entity"
        )
    
    # Column configuration based on format
    st.subheader("2️⃣ Column Configuration")
    
    patient_id_col_long = obs_name_col_long = obs_value_col_long = ""
    selected_columns_wide = []
    
    if data_file and not df.empty:
        if data_format == "Wide":
            st.info("📋 For Wide format, select which columns to include in the mapping")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Option to select all columns or specific ones
                mapping_mode = st.radio(
                    "Column selection mode:",
                    ("All columns", "Select specific columns"),
                    key="wide_column_mode"
                )
            
            with col2:
                if mapping_mode == "Select specific columns":
                    selected_columns_wide = st.multiselect(
                        "Select columns to map:",
                        options=list(df.columns),
                        default=list(df.columns)[:10] if len(df.columns) > 10 else list(df.columns),
                        key="wide_columns_select",
                        help="Choose which columns should be included in the mapping"
                    )
                else:
                    selected_columns_wide = list(df.columns)
                    st.info(f"Will use all {len(selected_columns_wide)} columns for mapping")
            
            # Optional: Allow user to specify an ID column even in wide format
            id_column_wide = st.selectbox(
                "ID/Identifier Column (optional):",
                options=["None"] + list(df.columns),
                key="wide_id_column",
                help="Select a column that uniquely identifies each row (e.g., patient_id)"
            )
            if id_column_wide == "None":
                id_column_wide = None
                
        elif data_format == "Long":
            st.info("📋 For Long format, specify the key columns")
            
            col1, col2, col3 = st.columns(3)
            
            column_names = list(df.columns)
            
            with col1:
                patient_id_col_long = st.selectbox(
                    "Entity ID Column:", 
                    column_names, 
                    key="auto_patient_id",
                    help="Column containing entity identifiers (e.g., patient_id)"
                )
            
            with col2:
                obs_name_col_long = st.selectbox(
                    "Observation Name Column:", 
                    column_names, 
                    key="auto_obs_name",
                    help="Column containing observation/variable names"
                )
            
            with col3:
                obs_value_col_long = st.selectbox(
                    "Observation Value Column:", 
                    column_names, 
                    key="auto_obs_value",
                    help="Column containing observation values"
                )
    # Model input
    st.subheader("3️⃣ Pydantic Model")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        py_file = st.file_uploader("Upload .py file", type=["py"], key="auto_py_file")
    
    with col2:
        code_text = st.text_area(
            "Or paste code", 
            height=200, 
            key="auto_code_text"
        )
    
    # Load models
    models = None
    schema_module = None
    
    if py_file is not None:
        models, schema_module = load_pydantic_model_from_file(py_file)
    elif code_text.strip():
        models, schema_module = load_pydantic_model_from_code(code_text)
    
    if models:
        model_names = list(models.keys())
        model_name = st.selectbox("Select model", model_names, key="auto_model_select")
    
    # LLM Configuration
    st.subheader("4️⃣ LLM Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        llm_provider = st.selectbox(
            "Provider",
            ["OpenAI", "Azure OpenAI", "OpenAI-Compatible (Local)"],
            key="llm_provider"
        )
        
        api_key = st.text_input(
            "API Key", 
            type="password", 
            key="llm_api_key",
            help="Use 'ollama' for local Ollama installations"
        )
    
    with col2:
        # Model defaults based on provider
        default_models = {
            "OpenAI": "gpt-4o-mini",
            "Azure OpenAI": "gpt-4",
            "OpenAI-Compatible (Local)": "llama3.1:latest"
        }
        
        llm_model = st.text_input(
            "Model Name",
            value=default_models.get(llm_provider, "gpt-4o-mini"),
            key="llm_model_name"
        )
    
    # Additional config for Azure/Local
    if llm_provider == "Azure OpenAI":
        col1, col2 = st.columns(2)
        with col1:
            azure_endpoint = st.text_input("Azure Endpoint", key="azure_endpoint")
        with col2:
            azure_version = st.text_input("API Version", value="2024-02-01", key="azure_version")
    elif llm_provider == "OpenAI-Compatible (Local)":
        base_url = st.text_input(
            "Base URL",
            value="http://localhost:11434/v1",
            key="base_url",
            help="For Ollama: http://localhost:11434/v1"
        )
    
    # Generate data summary
    if data_file and 'df' in locals() and not df.empty:
        with st.expander("📊 Data Summary (will be sent to LLM)", expanded=False):
            # Update data summary generation to use selected columns for wide format
            if data_format == "Wide":
                data_summary = generate_data_summary(
                    df[selected_columns_wide] if selected_columns_wide else df,
                    data_format,
                    id_column=id_column_wide if 'id_column_wide' in locals() else None
                )
            else:
                data_summary = generate_data_summary(
                    df, 
                    data_format, 
                    patient_id_col_long, 
                    obs_name_col_long, 
                    obs_value_col_long
                )
            st.text_area("Summary", value=data_summary, height=300, disabled=True)
    
    # Generate mapping button
    if st.button("🎯 Generate Smart Mapping", key="generate_smart_mapping", type="primary"):
        if not all([models, model_name, data_file, api_key]):
            st.error("Please provide all required inputs")
        else:
            # Validate column configuration
            if data_format == "Wide" and 'selected_columns_wide' in locals() and not selected_columns_wide:
                st.error("Please select at least one column for mapping")
            elif data_format == "Long" and not all([patient_id_col_long, obs_name_col_long, obs_value_col_long]):
                st.error("Please specify all required columns for long format")
            else:
                with st.spinner("Generating mapping with LLM..."):
                    try:
                        # Get model code
                        pydantic_code = code_text if code_text.strip() else py_file.read().decode('utf-8')
                        
                        # Generate prompt
                        prompt = build_llm_prompt(
                            pydantic_code, 
                            model_name, 
                            data_summary if 'data_summary' in locals() else ""
                        )
                        
                        # Call LLM
                        mapping_json = call_llm_for_mapping(
                            prompt,
                            llm_provider,
                            api_key,
                            llm_model,
                            azure_endpoint if llm_provider == "Azure OpenAI" else None,
                            azure_version if llm_provider == "Azure OpenAI" else None,
                            base_url if llm_provider == "OpenAI-Compatible (Local)" else None
                        )
                        
                        # Parse and display results
                        if mapping_json:
                            mapping_df = pd.DataFrame(mapping_json)
                            st.success("✅ Mapping generated successfully!")
                            
                            # Show preview
                            show_data_preview(mapping_df, "Generated Mapping Preview")
                            
                            # Download options
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                csv_data = mapping_df.to_csv(index=False)
                                st.download_button(
                                    "Download as CSV",
                                    data=csv_data,
                                    file_name=f"smart_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            
                            with col2:
                                excel_io = io.BytesIO()
                                with pd.ExcelWriter(excel_io, engine='xlsxwriter') as writer:
                                    mapping_df.to_excel(writer, index=False, sheet_name='mapping')
                                    # Add guide sheet
                                    guide_df = pd.DataFrame({
                                        "Note": ["This mapping was auto-generated. Please review and adjust as needed."]
                                    })
                                    guide_df.to_excel(writer, index=False, sheet_name='guide')
                                excel_io.seek(0)
                                
                                st.download_button(
                                    "Download as Excel",
                                    data=excel_io,
                                    file_name=f"smart_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            
                            st.info("💡 Please review and adjust the generated mapping before using it for transformation")
                    
                    except Exception as e:
                        st.error(f"Error generating mapping: {str(e)}")
                        st.code(traceback.format_exc())

# ===========================
# Helper Functions for Smart Mapping
# ===========================

def generate_data_summary(df: pd.DataFrame, data_format: str, 
                         patient_id_col: str = "", obs_name_col: str = "", 
                         obs_value_col: str = "", id_column: str = None) -> str:
    """Generate a summary of the data for LLM"""
    summaries = []
    
    if data_format == "Wide":
        summaries.append(f"Data Format: Wide (one row per entity)")
        summaries.append(f"Total Rows: {len(df)}")
        summaries.append(f"Total Columns Being Mapped: {len(df.columns)}")
        if id_column:
            summaries.append(f"ID Column: '{id_column}'")
            # Show unique ID count if available
            if id_column in df.columns:
                summaries.append(f"Unique IDs: {df[id_column].nunique()}")
        summaries.append("")
        
        for idx, col in enumerate(df.columns[:50], 1):  # Limit to first 50 columns
            col_data = df[col]
            unique_count = col_data.nunique()
            
            summary_parts = [
                f"{idx}. Column: '{col}'",
                f"   Type: {col_data.dtype}",
                f"   Non-null: {col_data.notna().sum()}/{len(col_data)}",
                f"   Unique values: {unique_count}"
            ]
            
            if unique_count <= AppConfig.MAX_UNIQUE_VALUES_DISPLAY:
                unique_vals = col_data.dropna().unique()[:AppConfig.MAX_UNIQUE_VALUES_DISPLAY]
                summary_parts.append(f"   Values: {list(unique_vals)}")
            else:
                sample_vals = col_data.dropna().sample(min(5, len(col_data.dropna()))).tolist()
                summary_parts.append(f"   Sample: {sample_vals}")
            
            summaries.append("\n".join(summary_parts))
    
    elif data_format == "Long" and all([patient_id_col, obs_name_col, obs_value_col]):
        summaries.append(f"Data Format: Long (multiple rows per entity)")
        summaries.append(f"Entity ID Column: '{patient_id_col}'")
        summaries.append(f"Observation Name Column: '{obs_name_col}'")
        summaries.append(f"Observation Value Column: '{obs_value_col}'")
        summaries.append(f"Total Rows: {len(df)}")
        summaries.append(f"Unique Entities: {df[patient_id_col].nunique()}")
        summaries.append(f"Unique Observations: {df[obs_name_col].nunique()}\n")
        
        obs_names = df[obs_name_col].value_counts().head(50).index
        
        for idx, obs in enumerate(obs_names, 1):
            obs_data = df[df[obs_name_col] == obs][obs_value_col]
            unique_count = obs_data.nunique()
            
            summary_parts = [
                f"{idx}. Observation: '{obs}'",
                f"   Count: {len(obs_data)}",
                f"   Type: {obs_data.dtype}",
                f"   Unique values: {unique_count}"
            ]
            
            if unique_count <= AppConfig.MAX_UNIQUE_VALUES_DISPLAY:
                unique_vals = obs_data.dropna().unique()[:AppConfig.MAX_UNIQUE_VALUES_DISPLAY]
                summary_parts.append(f"   Values: {list(unique_vals)}")
            else:
                sample_vals = obs_data.dropna().sample(min(5, len(obs_data.dropna()))).tolist()
                summary_parts.append(f"   Sample: {sample_vals}")
            
            summaries.append("\n".join(summary_parts))
    
    return "\n\n".join(summaries)

def build_llm_prompt(pydantic_code: str, model_name: str, data_summary: str) -> str:
    """Build prompt for LLM mapping generation"""
    return f"""You are an expert data engineer specializing in medical/healthcare data transformation.

Your task is to create a mapping template that links database observations/columns to Pydantic model fields.

CRITICAL RULES:
1. Output ONLY a valid JSON array - no explanations or text
2. For boolean fields: create 2 rows (True/False)
3. For enum fields: create 1 row per enum value
4. For numeric/string fields: leave pydantic_value blank
5. Match observation names intelligently (consider abbreviations, synonyms)
6. Use evaluation_method = "smart_exact_match" by default
7. Use multiValue_handling_method = "haveBoth" (or "collect_all" for lists)

Pydantic Model ({model_name}):
```python
{pydantic_code}
```

Data Summary:
{data_summary}

OUTPUT FORMAT (JSON array only):
[
  {{
    "pydantic_field": "patient_id",
    "pydantic_type": "str",
    "evaluation_method": "smart_exact_match",
    "multiValue_handling_method": "haveBoth",
    "pydantic_value": "",
    "Observation_ColName": "patient_identifier",
    "Observation_Value": "",
    "Observation_Value2": "",
    "Observation_Value3": "",
    "Observation_Value4": "",
    "notes": "Direct mapping from patient_identifier column"
  }},
  {{
    "pydantic_field": "gender",
    "pydantic_type": "GenderEnum",
    "evaluation_method": "smart_exact_match",
    "multiValue_handling_method": "first",
    "pydantic_value": "male",
    "Observation_ColName": "sex",
    "Observation_Value": "M",
    "Observation_Value2": "Male",
    "Observation_Value3": "MALE",
    "Observation_Value4": "",
    "notes": "Maps various representations to male enum"
  }}
]"""

def call_llm_for_mapping(prompt: str, provider: str, api_key: str, 
                        model: str, azure_endpoint: str = None, 
                        azure_version: str = None, base_url: str = None) -> List[Dict]:
    """Call LLM to generate mapping"""
    try:
        from openai import OpenAI, AzureOpenAI
        
        # Initialize client based on provider
        if provider == "Azure OpenAI":
            client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_version
            )
        elif provider == "OpenAI-Compatible (Local)":
            client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        else:  # OpenAI
            client = OpenAI(api_key=api_key)
        
        # Make API call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a data mapping expert. Output only valid JSON arrays."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=4096
        )
        
        # Parse response
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON from response
        import re
        
        # First try direct parsing
        try:
            return json.loads(content)
        except:
            # Try to find JSON array in the content
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                raise ValueError("No valid JSON array found in LLM response")
    
    except Exception as e:
        st.error(f"LLM call failed: {str(e)}")
        return None

# ===========================
# Main Application
# ===========================

def main():
    """Main application entry point"""
    init_session_state()
    
    # Title and description
    st.title("🔄 Data2Pydantic Mapping Tool - Enhanced")
    st.markdown("""
    Transform your EHR/database data into validated Pydantic models with this enhanced mapping tool.
    
    **Key Features:**
    - 📝 Generate mapping templates from Pydantic models
    - 🔄 Transform data using mappings with validation
    - 🤖 AI-powered mapping suggestions
    - 📊 Data preview and validation
    - 💾 Multiple export formats
    """)
    
    # Navigation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 Generate Template", use_container_width=True, type="primary" if st.session_state["active_section"] == "template" else "secondary"):
            st.session_state["active_section"] = "template"
    
    with col2:
        if st.button("🔄 Transform Data", use_container_width=True, type="primary" if st.session_state["active_section"] == "transform" else "secondary"):
            st.session_state["active_section"] = "transform"
    
    with col3:
        if st.button("🤖 Smart Mapping", use_container_width=True, type="primary" if st.session_state["active_section"] == "smart" else "secondary"):
            st.session_state["active_section"] = "smart"
    
    st.divider()
    
    # Show selected section
    if st.session_state["active_section"] == "template":
        section_generate_template()
    elif st.session_state["active_section"] == "transform":
        section_transform_data()
    elif st.session_state["active_section"] == "smart":
        section_smart_mapping()
    else:
        # Show welcome screen
        st.info("👈 Select an option above to get started")
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.markdown("""
            ### How to use this tool:
            
            1. **Generate Template** 📝
               - Upload or paste your Pydantic model
               - Download the mapping template
               - Fill in the template with your data mappings
            
            2. **Transform Data** 🔄
               - Upload your completed mapping template
               - Upload your source data (CSV/Excel)
               - Transform and download results
            
            3. **Smart Mapping** 🤖 (Optional)
               - Let AI suggest mappings based on your data
               - Review and adjust suggestions
               - Use for complex or large datasets
            
            ### Tips:
            - Start with a small sample of your data
            - Review the examples in the template
            - Use the mapping monitor to fix errors
            - For PHI/PII, use local LLMs or compliant services
            """)

# ===========================
# Run Application
# ===========================

if __name__ == "__main__":
    main()