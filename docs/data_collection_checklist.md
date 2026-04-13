# Phase 2: Data Collection Protocol Checklist

This guide ensures proper protocol execution for Phase 2 data collection alongside the motor imagery paradigm. Follow this exact sequence for consistency across all 6 subjects.

## 1. Preparation (Before Subject Arrives)
- [ ] **Hardware Check**: Connect the Upside Down Labs kit to the Arduino and the Arduino to the PC via USB.
- [ ] **Signal Check**: Test the port to ensure the Arduino is streaming `(python src/acquisition.py --check-quality --port COM3)`.
- [ ] **Environment**: Ensure the room is quiet, well-lit, and the computer screen is positioned comfortably for the subject.
- [ ] **Print Forms**: Have the Informed Consent Form (`docs/consent_form.md` equivalent) printed and ready to be signed.

## 2. Subject Onboarding (Mins 0-10)
- [ ] **Explain the Process**: Detail the two tasks (imagining moving left hand, imagining moving right hand).
- [ ] **Consent**: Ask them to read and sign the Informed Consent Form.
- [ ] **Pre-Session Guidance**: Instruct them to limit facial movements and blinks, and stay completely relaxed to avoid muscle artifacts.

## 3. Hardware Attachment (Mins 10-15)
- [ ] **Clean Application Areas**: Clean skin on forehead and earlobes using an alcohol wipe or mild soapy water to remove oils.
- [ ] **Electrode Placement**: Apply gel to the electrodes and secure them based on the 10-20 international system:
    - IN+ (Signal): C3 (Left Motor Cortex)
    - IN- (Signal): C4 (Right Motor Cortex)
    - Reference/Ground: Fpz (Forehead) or Earlobe based on the wiring of the generic bio-amp.
- [ ] **Signal Validation**: View raw stream data once more (`python src/acquisition.py --check-quality --port COM3`) ensuring SNR > 8 dB and low noise. Watch out for 50Hz interference. 

## 4. Practice Run (Mins 15-20)
- [ ] Start a fake session or just explain the 9-second rhythm.
    - 0s-2s: Relax window (Ready screen)
    - 2s-3s: Visual Cue (Left/Right Arrow)
    - 3s-7s: Motor Imagery Time (Screen says IMAGINE)
    - 7s-9s: Rest Interval
- [ ] Answer any final questions before generating real data.

## 5. Main Experiment (Mins 20-35)
- [ ] Launch `src/experiment.py` using command:
  ```bash
  python src/experiment.py --subject [ID] --session [SESSION]
  ```
  *(Be sure to replace [ID] and [SESSION] with the appropriate numbers)*
- [ ] Monitor the script progress on the terminal or the Pygame screen directly.
- [ ] Periodically check in with the subject visually and pause briefly if they are fatigued.

## 6. Wrap Up (Mins 35-40)
- [ ] When the script finishes saving the `.npz` and `.json` files, safely unplug the electrodes from the subject.
- [ ] Provide a damp paper towel to help them remove remaining conductive gel.
- [ ] Save the signed consent form in a secure place.
- [ ] Review `data/raw/` folder to confirm the session metadata and signals are present.
