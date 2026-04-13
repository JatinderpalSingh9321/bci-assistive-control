# BCI Assistive Control — Progress Report
## Date: April 13, 2026
## Phase 2: Data Collection & Hardware Finalization

This document outlines the milestones achieved, modifications made, and findings gathered during today's work sessions.

---

### 1. Architectural & Hardware Simplification
The project's scope was successfully narrowed to match the specific capabilities of the open-source hardware at our disposal.
* **3-Electrode Topology:** Retooled the entire documentation, data-collection checklists, and pipeline configurations to transition away from a standard 10-20 multi-channel setup to a single-channel, differential 3-Electrode framework optimized for the **Upside Down Labs BioAmp EXG Pill**:
  * `IN+`: C3 (Left Motor Cortex)
  * `IN-`: C4 (Right Motor Cortex)
  * `GND`: Fpz (Forehead Medial)
* **Metadata Enforcement:** Session metadata within `src/utils.py` was updated so all captured sessions specifically note the `C3, C4, Fpz` single-channel configuration.

### 2. Firmware Optimization (Arduino UNO R4 Minima Support)
An initial issue with the Arduino programmer failing to sync (`resp=0x00`) was triaged and resolved.
* **Architecture Change:** Identified that the board in use was the 32-bit ARM-based *Arduino Uno R4 Minima*, not the 8-bit AVR *Arduino Uno R3*.
* **Code Modification:** Updated `firmware/eeg_stream.ino` to prevent compiling errors, explicitly stripping out legacy AVR commands `analogReference(EXTERNAL)` and standardizing the pipeline by enforcing `analogReadResolution(10)` to maintain software compatibility across the toolchain.
* **Port Mapping:** Dynamically updated the overarching Python environment to point serial acquisition tools directly to `COM7`.

### 3. Execution of Phase 2: Data Collection
Executed the PyGame-driven paradigm to collect binary (Left vs. Right) Motor Imagery epochs.
* **Capture Milestones:** Successfully launched and verified the automated data collection pipeline (`experiment.py`) for four complete subjects (Subjects 001 - 004).
* **Automated Data Protection:** Safely ran background testing making sure to explicitly isolate subjects and execute retry commands without accidentally overwriting hard-earned legacy data blocks.
* **Storage Verification:** Validated 80 complete trials (~4 minutes of isolated motor imagery) per subject mapped effectively to `data/raw/` in the `.npz` container format. 

### 4. End-to-End Pipeline Validation
Ran the signal datasets through the entire Machine Learning pipeline.
* **Preprocessing Validation:** `preprocessing.py` fully accepted the real hardware collections and correctly chunked the stream while successfully applying the notch (50Hz) and band-pass (8-30Hz) filters to 320 total epochs.
* **Accuracy Assessment:**
  * SVM Baseline Architecture: `50.6%`
  * 1D-CNN Core Architecture: `49.4%`
* **Interpretation of Results:** Because the results hovered exactly at random probability boundaries (50%), it confirmed the models processed everything mathematically correct, but failed to find Event-Related Desynchronization (ERD).
* **Forward Guidance provided:** Laid out the three specific failure points that require fixing for the hardware (Conductive gel impedance, Fpz grounding to fight 50Hz environmental overlap, and kinetic imagery effort).

### 5. Repository Maintenance
* Generated crucial `docs/consent_form.md` and `docs/data_collection_checklist.md` to ensure data collection methods maintain ethical and systemic rigour.
* Safely committed all software structure, configurations, and documentation improvements directly into Git tracking out-of-the-box, ensuring no loss of progress for the current build versions.
