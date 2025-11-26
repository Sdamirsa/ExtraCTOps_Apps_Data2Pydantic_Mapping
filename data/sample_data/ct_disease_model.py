"""
CT Scan Disease Information Extraction - Pydantic Model
Sample data model for extracting disease information from CT scan reports
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class DiseasePresence(str, Enum):
    """Enum for disease presence status"""
    Present = "Present"
    Absent = "Absent"
    Suspected = "Suspected"
    Cannot_be_determined = "Cannot_be_determined"


class SeverityScale(str, Enum):
    """Enum for severity classifications"""
    Minimal = "Minimal"
    Mild = "Mild" 
    Moderate = "Moderate"
    Severe = "Severe"
    Critical = "Critical"
    Unknown = "Unknown"


class LocationSide(str, Enum):
    """Enum for anatomical side"""
    Right = "Right"
    Left = "Left"
    Bilateral = "Bilateral"
    Central = "Central"
    Unknown = "Unknown"


class TumorType(str, Enum):
    """Enum for tumor classifications"""
    Benign = "Benign"
    Malignant = "Malignant"
    Metastatic = "Metastatic"
    Unknown = "Unknown"


class ContrastEnhancement(str, Enum):
    """Enum for contrast enhancement patterns"""
    None_NoEnhancement = "None_NoEnhancement"
    Minimal = "Minimal"
    Mild = "Mild"
    Moderate = "Moderate"
    Marked = "Marked"
    Heterogeneous = "Heterogeneous"
    Rim_enhancing = "Rim_enhancing"


class MeasurementWithUnit(BaseModel):
    """Model for measurements with units"""
    value: Optional[float] = Field(None, description="Numeric measurement value")
    unit: Optional[str] = Field(None, description="Unit of measurement (mm, cm, etc.)")
    comment: Optional[str] = Field(None, description="Additional measurement notes")


class QualitativeSize(str, Enum):
    """Enum for qualitative size descriptions"""
    Very_small = "Very_small"
    Small = "Small"
    Medium = "Medium"
    Large = "Large"
    Very_large = "Very_large"
    Massive = "Massive"


class LesionCharacteristics(BaseModel):
    """Model for lesion/mass characteristics"""
    size_mm: Optional[MeasurementWithUnit] = Field(None, description="Lesion size in mm")
    size_qualitative: Optional[QualitativeSize] = Field(None, description="Qualitative size assessment")
    contrast_enhancement: Optional[ContrastEnhancement] = Field(None, description="Enhancement pattern")
    location_side: Optional[LocationSide] = Field(None, description="Anatomical location side")
    morphology_description: Optional[str] = Field(None, description="Detailed morphological description")


class PulmonaryNodule(BaseModel):
    """Model for pulmonary nodules"""
    present: Optional[DiseasePresence] = Field(None, description="Presence of pulmonary nodules")
    count: Optional[int] = Field(None, description="Number of nodules identified")
    characteristics: Optional[LesionCharacteristics] = Field(None, description="Nodule characteristics")
    largest_nodule_size_mm: Optional[MeasurementWithUnit] = Field(None, description="Size of largest nodule")
    calcification_present: Optional[bool] = Field(None, description="Presence of calcification")
    comment: Optional[str] = Field(None, description="Additional nodule findings")


class PulmonaryMass(BaseModel):
    """Model for pulmonary masses"""
    present: Optional[DiseasePresence] = Field(None, description="Presence of pulmonary mass")
    tumor_type: Optional[TumorType] = Field(None, description="Tumor classification")
    characteristics: Optional[LesionCharacteristics] = Field(None, description="Mass characteristics")
    cavitation_present: Optional[bool] = Field(None, description="Presence of cavitation")
    necrosis_present: Optional[bool] = Field(None, description="Presence of necrosis")
    comment: Optional[str] = Field(None, description="Additional mass findings")


class PleuralEffusion(BaseModel):
    """Model for pleural effusion"""
    present: Optional[DiseasePresence] = Field(None, description="Presence of pleural effusion")
    location_side: Optional[LocationSide] = Field(None, description="Side of effusion")
    severity: Optional[SeverityScale] = Field(None, description="Severity of effusion")
    volume_estimate_ml: Optional[MeasurementWithUnit] = Field(None, description="Estimated volume")
    loculated: Optional[bool] = Field(None, description="Whether effusion is loculated")
    comment: Optional[str] = Field(None, description="Additional effusion findings")


class LiverLesion(BaseModel):
    """Model for liver lesions"""
    present: Optional[DiseasePresence] = Field(None, description="Presence of liver lesions")
    lesion_count: Optional[int] = Field(None, description="Number of liver lesions")
    characteristics: Optional[LesionCharacteristics] = Field(None, description="Lesion characteristics")
    tumor_type: Optional[TumorType] = Field(None, description="Lesion classification")
    segment_location: Optional[str] = Field(None, description="Liver segment location")
    comment: Optional[str] = Field(None, description="Additional liver findings")


class AbdominalFindings(BaseModel):
    """Model for abdominal findings"""
    liver_lesions: Optional[LiverLesion] = Field(None, description="Liver lesion findings")
    ascites_present: Optional[bool] = Field(None, description="Presence of ascites")
    ascites_severity: Optional[SeverityScale] = Field(None, description="Ascites severity if present")
    lymphadenopathy_present: Optional[bool] = Field(None, description="Presence of lymphadenopathy")
    largest_lymph_node_mm: Optional[MeasurementWithUnit] = Field(None, description="Size of largest lymph node")
    comment: Optional[str] = Field(None, description="Additional abdominal findings")


class ThoracicFindings(BaseModel):
    """Model for thoracic findings"""
    pulmonary_nodules: Optional[PulmonaryNodule] = Field(None, description="Pulmonary nodule findings")
    pulmonary_mass: Optional[PulmonaryMass] = Field(None, description="Pulmonary mass findings")
    pleural_effusion: Optional[PleuralEffusion] = Field(None, description="Pleural effusion findings")
    mediastinal_lymphadenopathy: Optional[bool] = Field(None, description="Mediastinal lymphadenopathy")
    pneumothorax_present: Optional[bool] = Field(None, description="Presence of pneumothorax")
    comment: Optional[str] = Field(None, description="Additional thoracic findings")


class BoneFindings(BaseModel):
    """Model for bone/skeletal findings"""
    bone_lesions_present: Optional[DiseasePresence] = Field(None, description="Presence of bone lesions")
    lesion_count: Optional[int] = Field(None, description="Number of bone lesions")
    tumor_type: Optional[TumorType] = Field(None, description="Bone lesion classification")
    fracture_present: Optional[bool] = Field(None, description="Presence of fractures")
    location_description: Optional[str] = Field(None, description="Location of bone findings")
    comment: Optional[str] = Field(None, description="Additional bone findings")


class TechnicalQuality(BaseModel):
    """Model for technical scan quality"""
    contrast_administered: Optional[bool] = Field(None, description="Whether contrast was administered")
    image_quality: Optional[str] = Field(None, description="Overall image quality assessment")
    motion_artifacts: Optional[bool] = Field(None, description="Presence of motion artifacts")
    slice_thickness_mm: Optional[float] = Field(None, description="CT slice thickness in mm")
    reconstruction_algorithm: Optional[str] = Field(None, description="Image reconstruction method")


class CTDiseaseReport(BaseModel):
    """Complete CT scan disease information model"""
    patient_id: str = Field(..., description="Unique patient identifier")
    scan_date: Optional[str] = Field(None, description="Date of CT scan (YYYY-MM-DD)")
    scan_type: Optional[str] = Field(None, description="Type of CT scan (chest, abdomen, etc.)")
    
    # Primary findings sections
    thoracic_findings: Optional[ThoracicFindings] = Field(None, description="Chest/thoracic findings")
    abdominal_findings: Optional[AbdominalFindings] = Field(None, description="Abdominal findings") 
    bone_findings: Optional[BoneFindings] = Field(None, description="Bone/skeletal findings")
    
    # Overall assessment
    overall_impression: Optional[str] = Field(None, description="Radiologist's overall impression")
    primary_diagnosis: Optional[str] = Field(None, description="Primary diagnostic impression")
    recommended_followup: Optional[str] = Field(None, description="Recommended follow-up actions")
    
    # Technical details
    technical_quality: Optional[TechnicalQuality] = Field(None, description="Technical scan parameters")
    
    # Additional information
    age_at_scan: Optional[int] = Field(None, description="Patient age at time of scan")
    clinical_indication: Optional[str] = Field(None, description="Clinical reason for scan")
    radiologist_confidence: Optional[float] = Field(None, description="Radiologist confidence score (0-1)")
    
    # General comment
    additional_notes: Optional[str] = Field(None, description="Any additional clinical notes")