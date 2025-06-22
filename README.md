# SALUSLux

**SALUSLux** is an open-source Python package for simulating and optimizing urban street lighting, with a focus on pedestrian safety and light pollution mitigation. Designed for researchers, engineers, and city planners, SALUSLux brings full programmatic control to the lighting simulation process using standardized IES photometric data.

> **Built by the SALUS Lab at Carnegie Mellon University**  
> [SALUSLab Website](https://www.flanigansaluslab.com) · [IEEE Paper](Coming soon) · License: Apache-2.0

---

## Key Features

- **Parse IES Files** (ANSI/IES LM-63-19 format) to extract luminous intensity data  
- **Compute Illuminance**: Horizontal, vertical, and semi-cylindrical, using vector-based physics  
- **Simulate Luminance**: With realistic Lambertian surface assumptions  
- **Evaluate Glare**: Compute CIE glare rating (GR) from pedestrian or driver perspectives  
- **Intersection Simulation**: Analyze full crosswalk layouts with 16 surfaces and 6 metrics  
- **Modular & Extensible**: Designed for easy integration into smart city, ML, or parametric design tools  

---
### Requirements

SALUSLux uses the following Python packages:

```
numpy, scipy, matplotlib, pandas
```

---

## Quick Start

Import the library:

```python
import saluslux as slux
```

Example: Parse an IES file and compute illumination on a surface:

```python
data = slux.parse_ies('example.ies')
grid = slux.generate_unified_grid(100, [(-10, 10, -10, 10)])
Eh = slux.compute_illuminance(grid, data['sources'], normal_vector=[0, 0, 1])
```



---

## Why SALUSLux?

Mainstream lighting tools like Revit or DIALux are GUI-heavy and proprietary. SALUSLux:

- Empowers **reproducible scientific workflows**
- Enables **large-scale parametric simulations**
- Avoids license limitations with **fully open photometric parsing**

---
## Citation

If you use SALUSLux in your research, please cite:

```
Kavee, K., Flanigan, K. A., & Quick, S. (2025). SALUSLux: Open-source software for optimizing street lighting to improve pedestrian safety and light pollution. IEEE International Smart Cities Conference.
```

---

## Tutorial videos and Workshop

For more complex use cases (e.g., 16-surface 4-way intersection simulations), see the following clips (COMING SOON)
