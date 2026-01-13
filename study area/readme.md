# Study Area: Horseshoe Beach, Florida

This folder contains the geographic boundary data and ground-truth annotations used in the **Bi-Temporal Street-View** project for hyperlocal disaster damage assessment.

## 📍 Study Area Overview

The study area focuses on **Horseshoe Beach, Florida**, a coastal community that has experienced multiple hurricane events, making it a suitable case study for evaluating disaster damage using **bi-temporal street-view imagery**.

The spatial extent of the study area is defined by an official city boundary polygon, which is used to:
- Constrain street-view image collection,
- Support spatial filtering and visualization,
- Enable location-aware damage analysis.

## 📂 File Description

### 1. City Boundary (Shapefile)

The following files together define the city boundary of Horseshoe Beach:

- `Horseshoe_Beach_City_Boundary.shp` – Geometry file (polygon)
- `Horseshoe_Beach_City_Boundary.shx` – Shape index
- `Horseshoe_Beach_City_Boundary.dbf` – Attribute table
- `Horseshoe_Beach_City_Boundary.prj` – Coordinate reference system
- `Horseshoe_Beach_City_Boundary.cpg` – Character encoding
- `Horseshoe_Beach_City_Boundary.sbn / .sbx` – Spatial index files
- `Horseshoe_Beach_City_Boundary.shp.xml` – Metadata

These files should be used **together** and can be directly loaded into GIS software such as **ArcGIS Pro** or **QGIS**.

### 2. Ground Truth Annotations

- `groundtruth.csv`

This file contains ground-truth labels used for disaster damage assessment.  
Each record corresponds to a street-view image location within the study area and includes manually curated damage information (e.g., damage presence or severity level), which is used for:
- Model training and evaluation,
- Quantitative comparison between pre- and post-disaster imagery,
- Validation of bi-temporal and dual-channel frameworks.

## 🧭 Coordinate System

The city boundary shapefile includes a `.prj` file specifying the coordinate reference system (CRS).  
Please ensure consistent CRS handling when integrating this data with other spatial layers or street-view metadata.

## 📌 Usage Notes

- This study area is intended **only for research and academic purposes**.
- When using the boundary or ground-truth data in publications, please cite the corresponding paper or repository.
- Do not rename or separate shapefile components, as this may break GIS compatibility.

## 📬 Contact

If you encounter missing files, unclear annotations, or have questions regarding the study area definition, please feel free to contact:

**Yifan Yang**  
Email: yyf990925@gmail.com

