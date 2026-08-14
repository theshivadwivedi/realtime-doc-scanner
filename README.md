# Real-Time Document Scanner

Detects a document via webcam, verifies it's actually paper (not just
any rectangle), corrects perspective, and saves a clean scan.

## Features
- Contour-based document boundary detection
- Shape + position scoring to reject false positives
- Brightness/saturation check to confirm the object is actually paper
- Perspective correction (warp) to produce a flat, cropped scan

## Usage
```bash
pip install opencv-python numpy
python scanner.py
```

Press `q` to quit manually. Scans save automatically to `output/`.