# CT Scan Disease Extraction - Sample Data

This directory contains sample data demonstrating how to extract disease information from CT scan reports using the Data2Pydantic Mapping Tool.

## Overview

This example shows how to transform EHR/database data from CT scan reports into a structured Pydantic model for disease information extraction. The data represents a realistic clinical scenario with three different patient cases showcasing various disease patterns.

## Files Description

### 1. Pydantic Model
**`ct_disease_model.py`** - The target Pydantic model defining the structure for CT scan disease information.

**Key Features Demonstrated:**
- **Nested models** (e.g., `ThoracicFindings` → `PulmonaryNodule` → `LesionCharacteristics`)
- **Enum fields** with clinical terminology (`DiseasePresence`, `SeverityScale`, `TumorType`)
- **Boolean fields** for yes/no clinical findings
- **Numeric fields with units** (`MeasurementWithUnit` for sizes, volumes)
- **Text fields** for free-text descriptions
- **Optional fields** (most clinical findings may not be present)

### 2. Source Data Files (SQL-like format)

**`ct_observations.csv`** - Categorical/text observations from CT reports
- Contains coded observations (OBS001-OBS050) 
- Maps to qualitative findings like presence/absence, severity, location
- Example: OBS001 = "Pulmonary nodule presence" with values like "Present", "Absent"

**`ct_measurements.csv`** - Numeric measurements from CT reports  
- Contains coded measurements (MEAS001-MEAS020) with numeric values
- Maps to quantitative findings like sizes, volumes, scores
- Example: MEAS001 = "Largest nodule size" with value "8.5" (mm)

### 3. Data Dictionaries

**`observations_dict.csv`** - Maps observation codes to human-readable names
- Links codes like "OBS001" to descriptions like "Pulmonary nodule presence"
- Used for understanding what each coded observation represents

**`measurements_dict.csv`** - Maps measurement codes to names and units
- Links codes like "MEAS001" to "Largest nodule size" with unit "mm"
- Includes unit information for proper data interpretation

### 4. Completed Mapping

**`completed_mapping.csv`** - Shows how to map source data to Pydantic fields
- Demonstrates the mapping configuration for transformation
- Shows different mapping scenarios:
  - **Direct mapping**: `patient_id` → `patient_id` 
  - **Enum mapping**: "Present"/"Absent" → `DiseasePresence.Present`/`DiseasePresence.Absent`
  - **Boolean mapping**: "Yes"/"No" → `True`/`False`
  - **Nested field mapping**: Complex paths like `thoracic_findings::pulmonary_nodules::present`
  - **Numeric with units**: Measurements with unit assignment

## Sample Patients

### Patient PAT001 (Age 67)
- **Condition**: Multiple small pulmonary nodules
- **Key findings**: 3 bilateral nodules, largest 8.5mm, mild pleural effusion
- **Assessment**: Possible metastatic disease, follow-up needed

### Patient PAT002 (Age 58) 
- **Condition**: Primary lung cancer with metastases
- **Key findings**: Large left lung mass (45mm), liver metastases, bone lesion
- **Assessment**: Advanced malignancy requiring urgent oncology referral

### Patient PAT003 (Age 72)
- **Condition**: Benign findings 
- **Key findings**: Single small calcified nodule (4.2mm), likely granuloma
- **Assessment**: Benign appearance, annual follow-up

## Data Format Structure

This example uses **Long Format** data:
- **Entity ID**: `patient_id` (identifies each patient)
- **Observation Name**: `observation_code` or `measurement_code` 
- **Value**: `observation_value` or `measurement_value`
- **Date**: `scan_date`

## Key Mapping Concepts Demonstrated

### 1. **Enum Value Mapping**
```csv
# Maps various text representations to standardized enum values
pydantic_field,pydantic_value,Observation_Value,Observation_Value2,Observation_Value3
thoracic_findings::pulmonary_nodules::present,Present,Present,present,Yes,yes
```

### 2. **Nested Object Mapping**  
```csv
# Shows how to map to deeply nested model structures
thoracic_findings::pulmonary_nodules::characteristics::size_qualitative
```

### 3. **Measurements with Units**
```csv
# Demonstrates mapping numeric values and their units
thoracic_findings::pleural_effusion::volume_estimate_ml::value  # The numeric value
thoracic_findings::pleural_effusion::volume_estimate_ml::unit   # The unit (ml)
```

### 4. **Boolean Field Handling**
```csv
# Shows multiple ways to represent boolean values
pydantic_field,pydantic_value,Observation_Value,Observation_Value2
thoracic_findings::pneumothorax_present,True,Yes,yes,Y,positive
thoracic_findings::pneumothorax_present,False,No,no,N,negative
```

## How to Use This Example

1. **Load the Pydantic model** in the Data2Pydantic tool
2. **Upload the mapping file** (`completed_mapping.csv`)
3. **Upload the source data** (use `ct_observations.csv` as your main data file)
4. **Configure as Long format** with:
   - Entity ID column: `patient_id`
   - Observation name column: `observation_code` 
   - Value column: `observation_value`
5. **Transform the data** to generate structured JSON output

## Expected Output Structure

After transformation, each patient will have a structured record like:
```json
{
  "patient_id": "PAT001",
  "scan_date": "2024-01-15",
  "thoracic_findings": {
    "pulmonary_nodules": {
      "present": "Present",
      "count": 3,
      "characteristics": {
        "location_side": "Bilateral",
        "size_qualitative": "Small"
      }
    }
  },
  "age_at_scan": 67
}
```

## Clinical Relevance

This model captures key information needed for:
- **Cancer screening and staging**
- **Disease progression monitoring** 
- **Treatment response assessment**
- **Research data standardization**
- **Quality metrics reporting**

The structured output enables automated analysis, statistical reporting, and integration with clinical decision support systems.