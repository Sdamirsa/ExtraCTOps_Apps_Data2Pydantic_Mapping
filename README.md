# Data2Pydantic Mapping Tool 🔄

A powerful Streamlit application for mapping and transforming EHR/database data to user-defined Pydantic models. This tool supports mapping template generation, data transformation, and LLM-assisted auto-mapping for seamless data standardization.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.44%2B-red)](https://streamlit.io)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.0%2B-green)](https://pydantic.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Part of ExtraCTOps Project**: This tool is a component of the larger [Awesome Extraction with LLM ExtraCTOps](https://github.com/Sdamirsa/awesome_extraction_with_LLM_ExtraCTOps) project - a comprehensive toolkit for improving LLM performance through structured data extraction and evaluation.

## 🎯 What is Data2Pydantic?

Data2Pydantic is a specialized tool designed to bridge the gap between unstructured or semi-structured healthcare data and well-defined Pydantic models. It enables:

- **🏥 Healthcare Data Standardization**: Transform EHR data from various formats into standardized Pydantic models
- **📊 Flexible Data Mapping**: Support for both wide (one row per patient) and long (multiple rows per patient) data formats
- **🤖 AI-Powered Mapping**: Leverage LLMs to automatically suggest mappings between your data and Pydantic fields
- **✅ Validation & Quality Control**: Ensure data quality through Pydantic's built-in validation

## ✨ Key Features

### 1. **Generate Mapping Templates** 📝
- Automatically create Excel templates from your Pydantic models
- Pre-populated with enum values and field types
- Includes comprehensive instructions and examples

### 2. **Transform Data** 🔄
- Apply mappings to transform raw data into Pydantic-validated structures
- Support for complex nested models and relationships
- Export results as JSON or flattened Excel files

### 3. **Smart AI Mapping** 🤖
- Use LLMs (OpenAI, Azure OpenAI, or local models) to suggest mappings
- Privacy-conscious design with column selection
- Support for HIPAA-compliant LLM services

## 🚀 Quick Start

### For Non-Developers (Easy Setup)

#### Windows Users:

1. **Install Python**
   ```
   - Download Python 3.8+ from python.org
   - During installation, CHECK "Add Python to PATH"
   - Restart your computer
   ```

2. **Download the Tool**
   ```
   - Download this repository as ZIP
   - Extract to your Desktop
   ```

3. **Install & Run**
   ```cmd
   cd Desktop\data2pydantic-tool
   pip install -r requirements.txt
   streamlit run app.py
   ```

#### Mac/Linux Users:

1. **Install Python** (if not already installed)
   ```bash
   # Mac
   brew install python
   
   # Linux
   sudo apt update
   sudo apt install python3 python3-pip
   ```

2. **Download and Run**
   ```bash
   git clone <repository-url>
   cd data2pydantic-tool
   pip3 install -r requirements.txt
   streamlit run app.py
   ```

### For Developers

```bash
# Clone repository
git clone <repository-url>
cd data2pydantic-tool

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py

# For development with auto-reload
streamlit run app.py --server.runOnSave true
```

## 📖 Usage Guide

### Step 1: Generate Mapping Template

1. **Prepare Your Pydantic Model**
   ```python
   from pydantic import BaseModel, Field
   from enum import Enum
   from typing import Optional, List

   class GenderEnum(str, Enum):
       MALE = "male"
       FEMALE = "female"
       OTHER = "other"

   class Patient(BaseModel):
       patient_id: str = Field(description="Unique patient identifier")
       age: Optional[int] = Field(description="Patient age in years")
       gender: GenderEnum = Field(description="Patient gender")
       diagnoses: List[str] = Field(default=[], description="List of diagnoses")
   ```

2. **Generate Template**
   - Upload your `.py` file or paste the code
   - Select your model from the dropdown
   - Click "Generate Template"
   - Download the Excel template

3. **Fill the Template**
   - `Observation_ColName`: Your source data column name
   - `Observation_Value`: Values in your data that map to the pydantic_value
   - For enums/bools: Multiple rows are created for each possible value

### Step 2: Transform Your Data

1. **Upload Required Files**
   - Completed mapping template (Excel/CSV)
   - Your Pydantic model (.py file)
   - Source data file (Excel/CSV)

2. **Configure Data Format**
   - **Wide Format**: One row per entity (e.g., one row per patient)
   - **Long Format**: Multiple rows per entity (e.g., EAV format)

3. **Run Transformation**
   - Click "Transform Data"
   - Review validation results
   - Download transformed data as JSON or Excel

### Step 3: Smart Mapping (Optional)

1. **Upload Data and Model**
   - Upload your source data
   - Upload or paste your Pydantic model

2. **Configure Columns**
   - For Wide: Select which columns to include
   - For Long: Specify ID, observation name, and value columns

3. **Configure LLM**
   - Choose provider (OpenAI, Azure, Local)
   - Enter API credentials
   - Select model (e.g., gpt-4o-mini)

4. **Generate Mapping**
   - Review the data summary
   - Click "Generate Smart Mapping"
   - Download and review suggested mappings

## 🔧 Configuration

### Data Formats

#### Wide Format Example:
```csv
patient_id,age,sex,diagnosis1,diagnosis2
P001,45,M,Diabetes,Hypertension
P002,32,F,Asthma,
```

#### Long Format Example:
```csv
patient_id,observation,value
P001,age,45
P001,sex,M
P001,diagnosis,Diabetes
P001,diagnosis,Hypertension
P002,age,32
P002,sex,F
P002,diagnosis,Asthma
```

### Mapping Template Structure

| Field | Description |
|-------|-------------|
| `pydantic_field` | Target field in your Pydantic model |
| `pydantic_type` | Data type (str, int, enum name, etc.) |
| `evaluation_method` | How to match values (smart_exact_match, contains, regex) |
| `multiValue_handling_method` | How to handle multiple values (first, join_semicolon, haveBoth) |
| `pydantic_value` | For enums/bools: the value to set when matched |
| `Observation_ColName` | Column name in your source data |
| `Observation_Value[1-4]` | Values in source data that trigger this mapping |

## 🛡️ Privacy & Security

### For Healthcare Data:

- **Local Processing**: All data transformation happens locally
- **Smart Mapping Privacy**: Only column names and sample values are sent to LLMs
- **Column Selection**: Choose exactly which columns to include in AI mapping
- **HIPAA Compliance**: Support for compliant LLM services (Azure OpenAI with BAA)
- **Local LLM Support**: Use Ollama or other local models for complete privacy

### Best Practices:

1. **De-identify Data**: Remove PHI before using cloud LLM services
2. **Use Local Models**: For sensitive data, use Ollama or similar
3. **Select Columns**: Only include necessary columns in smart mapping
4. **Review Mappings**: Always review AI-generated mappings before use

## 🎨 Example Use Cases

### Clinical Trial Data Standardization
Transform diverse site data into standardized format:
```python
class ClinicalTrialVisit(BaseModel):
    subject_id: str
    visit_date: date
    vital_signs: VitalSigns
    adverse_events: List[AdverseEvent]
    medications: List[Medication]
```

### EHR Data Integration
Map hospital EHR exports to research database:
```python
class PatientEncounter(BaseModel):
    encounter_id: str
    patient_mrn: str
    admission_date: datetime
    discharge_diagnosis: List[Diagnosis]
    procedures: List[Procedure]
```

### Lab Results Harmonization
Standardize lab results from multiple sources:
```python
class LabResult(BaseModel):
    test_name: str
    value: float
    unit: str
    reference_range: str
    abnormal_flag: Optional[bool]
```

## 📦 Requirements

```txt
streamlit>=1.44.0
pandas>=2.2.0
pydantic>=2.11.0
openpyxl>=3.1.0
xlsxwriter>=3.2.0
openai>=1.79.0
python-dateutil>=2.9.0
```

## 🔧 Advanced Configuration

### Using Local LLMs (Ollama)

1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3.1`
3. In the app:
   - Select "OpenAI-Compatible (Local)"
   - Base URL: `http://localhost:11434/v1`
   - API Key: `ollama`
   - Model: `llama3.1:latest`

### Custom Evaluation Methods

- `smart_exact_match`: Case-insensitive exact matching (default)
- `exact_match`: Case-sensitive exact matching
- `contains`: Source value contains the specified value
- `regex`: Regular expression matching

### Multi-Value Handling

- `haveBoth`: Keep all values (default for lists)
- `first`: Take only the first value
- `join_semicolon`: Join multiple values with `;`

## 🐛 Troubleshooting

### Common Issues

**"No valid Pydantic models found"**
- Ensure your model inherits from `pydantic.BaseModel`
- Check that all imports are included in your .py file

**"Error during transformation"**
- Review the mapping monitor Excel for validation errors
- Ensure required fields have mappings
- Check that enum values match exactly

**"LLM mapping failed"**
- Verify API credentials
- Check internet connection
- Ensure selected columns have data

### Debug Mode

Set Streamlit to development mode for better error messages:
```bash
streamlit run app.py --server.runOnSave true --client.showErrorDetails true
```

## 🤝 Contributing

This tool is part of the [ExtraCTOps project](https://github.com/Sdamirsa/awesome_extraction_with_LLM_ExtraCTOps). Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Part of the **ExtraCTOps Project** for LLM-powered data extraction
- Built with **Streamlit** for easy deployment
- Powered by **Pydantic** for robust data validation
- LLM integration via **OpenAI** API

## 📧 Contact

- **Author**: Seyed Amir Ahmad Safavi-Naini
- **Email**: sdamirsa@gmail.com
- **Project**: [ExtraCTOps](https://github.com/Sdamirsa/awesome_extraction_with_LLM_ExtraCTOps)

---

⭐ **Star this repository** if you find it helpful!

🐛 **Report issues** in the GitHub issue tracker

💡 **Share your use cases** to help improve the tool
